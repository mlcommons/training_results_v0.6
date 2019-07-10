# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Training script for Mask-RCNN.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_util

import coco_metric
import dataloader
import mask_rcnn_model
import mask_rcnn_params
import mask_rcnn_runner
import runner_utils
from mlp_log import mlp_log

# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'gcp_project', default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_string(
    'tpu_zone', default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_string(
    'tpu', default=None,
    help='Name of the Cloud TPU for Cluster Resolvers.')
flags.DEFINE_string(
    'master',
    default=None,
    help='Name of the Cloud TPU for Cluster Resolvers. You must specify either '
    'this flag or --master.')

# Model specific paramenters
flags.DEFINE_string('tpu_job_name', default=None, help='The tpu worker name.')
flags.DEFINE_string(
    'eval_master', default='',
    help='GRPC URL of the eval master. Set to an appropiate value when running '
    'on CPU/GPU')
flags.DEFINE_bool('use_tpu', True, 'Use TPUs rather than CPUs')
flags.DEFINE_string('hparams', '',
                    'Comma separated k=v pairs of hyperparameters.')
flags.DEFINE_integer(
    'num_cores', default=8, help='Number of TPU cores for training')
flags.DEFINE_multi_integer(
    'input_partition_dims', None,
    'A list that describes the partition dims for all the tensors.')

flags.DEFINE_string('model_dir', None, 'Location of model_dir')
flags.DEFINE_string('resnet_checkpoint', '',
                    'Location of the ResNet50 checkpoint to use for model '
                    'initialization.')
flags.DEFINE_string(
    'training_file_pattern', None,
    'Glob for training data files (e.g., COCO train - minival set)')
flags.DEFINE_string(
    'validation_file_pattern', None,
    'Glob for evaluation tfrecords (e.g., COCO val2017 set)')
flags.DEFINE_string(
    'val_json_file',
    None,
    'COCO validation JSON containing golden bounding boxes.')

flags.DEFINE_string('mode', 'train',
                    'Mode to run: train or eval (default: train)')
flags.DEFINE_bool('eval_after_training', False, 'Run one eval after the '
                  'training finishes.')
flags.DEFINE_bool('use_fake_data', False, 'Use fake input.')

# For Eval mode
flags.DEFINE_integer('min_eval_interval', 180,
                     'Minimum seconds between evaluations.')
flags.DEFINE_integer(
    'eval_timeout', None,
    'Maximum seconds between checkpoints before evaluation terminates.')

FLAGS = flags.FLAGS


def main(argv):
  del argv  # Unused.

  # TODO(b/132208296): remove this workaround that uses control flow v2.
  control_flow_util.ENABLE_CONTROL_FLOW_V2 = True

  tpu = FLAGS.tpu or FLAGS.master
  tpu_cluster_resolver = runner_utils.create_tpu_cluster_resolver(
      FLAGS.use_tpu, tpu, FLAGS.tpu_zone, FLAGS.gcp_project)
  if tpu_cluster_resolver:
    tpu_grpc_url = tpu_cluster_resolver.get_master()
    tf.Session.reset(tpu_grpc_url)

  # Check data path
  run_train = FLAGS.mode in ('train', 'train_and_eval')
  if run_train and FLAGS.training_file_pattern is None:
    raise RuntimeError('You must specify --training_file_pattern for training.')
  run_eval = FLAGS.mode in ('eval', 'train_and_eval') or (
      FLAGS.mode == 'train' and FLAGS.eval_after_training)
  if run_eval:
    if FLAGS.validation_file_pattern is None:
      raise RuntimeError('You must specify --validation_file_pattern '
                         'for evaluation.')
    if FLAGS.val_json_file is None:
      raise RuntimeError('You must specify --val_json_file for evaluation.')

  # Parse hparams
  hparams = mask_rcnn_params.default_hparams()
  hparams.parse(FLAGS.hparams)

  # The following is for spatial partitioning. `features` has one tensor while
  # `labels` has 4 + (`max_level` - `min_level` + 1) * 2 tensors. The input
  # partition is performed on `features` and all partitionable tensors of
  # `labels`, see the partition logic below.
  # Note: In the below code, TPUEstimator uses both `shard` and `replica` (with
  # the same meaning).
  # Note that spatial partition is part of the model-parallelism optimization.
  # See core_assignment_utils.py for more details about model parallelism.
  if FLAGS.input_partition_dims:
    labels_partition_dims = {
        'gt_boxes': None,
        'gt_classes': None,
        'cropped_gt_masks': None,
    }
    for level in range(hparams.get('min_level'), hparams.get('max_level') + 1):
      labels_partition_dims['box_targets_%d' % level] = None
      labels_partition_dims['score_targets_%d' % level] = None
    num_cores_per_replica = int(np.prod(FLAGS.input_partition_dims))
    image_partition_dims = [
        FLAGS.input_partition_dims[i] for i in [1, 0, 2]
    ] if hparams.get('transpose_input') else FLAGS.input_partition_dims
    features_partition_dims = {
        'images': image_partition_dims,
        'source_ids': None,
        'image_info': None,
    }
    input_partition_dims = [features_partition_dims, labels_partition_dims]
    num_shards = FLAGS.num_cores // num_cores_per_replica
  else:
    num_cores_per_replica = None
    input_partition_dims = None
    num_shards = FLAGS.num_cores

  params = dict(
      hparams.values(),
      num_shards=num_shards,
      num_cores_per_replica=num_cores_per_replica,
      use_tpu=FLAGS.use_tpu,
      resnet_checkpoint=FLAGS.resnet_checkpoint,
      val_json_file=FLAGS.val_json_file,
      model_dir=FLAGS.model_dir)

  tpu_config = tf.contrib.tpu.TPUConfig(
      params['iterations_per_loop'],
      num_shards=num_shards,
      num_cores_per_replica=params['num_cores_per_replica'],
      input_partition_dims=input_partition_dims,
      per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig
      .PER_HOST_V2,
      tpu_job_name=FLAGS.tpu_job_name,
  )

  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      log_step_count_steps=params['iterations_per_loop'],
      tpu_config=tpu_config,
      save_checkpoints_steps=params['iterations_per_loop'],
  )

  train_replicas_per_worker = (
      params['cores_per_worker'] // params['num_cores_per_replica']
  ) if params['num_cores_per_replica'] else params['cores_per_worker']
  train_params = dict(
      params,
      replicas_per_worker=train_replicas_per_worker,
  )
  eval_params = dict(
      params,
      input_rand_hflip=False,
      resnet_checkpoint=None,
      is_training_bn=False,
  )

  # MLPerf logging.
  mlp_log.mlperf_print(key='init_start', value=None)
  mlp_log.mlperf_print(key='global_batch_size',
                       value=params['train_batch_size'])
  runner = None
  if run_train and run_eval:
    if params['train_use_tpu_estimator'] or params['eval_use_tpu_estimator']:
      raise RuntimeError('train_and_eval runner does not support TPUEstimator.')
    dist_eval_params = dict(
        eval_params,
        replicas_per_worker=train_replicas_per_worker,
    )
    runner = mask_rcnn_runner.TrainEvalRunner(
        model_fn=mask_rcnn_model.MaskRcnnModelFn(),
        input_fn=dataloader.InputReader(
            FLAGS.training_file_pattern,
            mode=tf.estimator.ModeKeys.TRAIN,
            use_fake_data=FLAGS.use_fake_data),
        eval_input_fn=dataloader.InputReader(
            FLAGS.validation_file_pattern, mode=tf.estimator.ModeKeys.PREDICT,
            distributed_eval=True),
        eval_metric=coco_metric.EvaluationMetric(
            FLAGS.val_json_file, use_cpp_extension=True),
        train_params=train_params,
        eval_params=dist_eval_params,
        run_config=run_config)
  elif run_train:
    # Check low-level train runner compatibility.
    if not params['train_use_tpu_estimator']:
      if FLAGS.mode == 'train_and_eval':
        raise RuntimeError('Low level train runner does not support mode '
                           'train_and_eval yet.')
    train_params = dict(
        params,
        replicas_per_worker=train_replicas_per_worker,
    )
    runner = mask_rcnn_runner.TrainRunner(
        model_fn=mask_rcnn_model.MaskRcnnModelFn(),
        input_fn=dataloader.InputReader(
            FLAGS.training_file_pattern,
            mode=tf.estimator.ModeKeys.TRAIN,
            use_fake_data=FLAGS.use_fake_data),
        params=train_params,
        run_config=run_config,
        use_tpu_estimator=train_params['train_use_tpu_estimator'])
  else:
    sidecar_eval_params = dict(
        eval_params,
        # sidecar eval only uses one worker and does not use spatial partition.
        replicas_per_worker=FLAGS.num_cores,)
    runner = mask_rcnn_runner.EvalRunner(
        mask_rcnn_model.MaskRcnnModelFn(),
        dataloader.InputReader(
            FLAGS.validation_file_pattern,
            mode=tf.estimator.ModeKeys.PREDICT),
        coco_metric.EvaluationMetric(
            FLAGS.val_json_file,
            use_cpp_extension=True),
        sidecar_eval_params,
        run_config,
        use_tpu_estimator=sidecar_eval_params['eval_use_tpu_estimator'])

  if FLAGS.mode == 'train':
    runner.train()
  elif FLAGS.mode == 'eval':

    def terminate_eval():
      tf.logging.info('Terminating eval after %d seconds of no checkpoints' %
                      FLAGS.eval_timeout)
      return True

    run_success = False
    # Run evaluation when there's a new checkpoint
    for ckpt in tf.contrib.training.checkpoints_iterator(
        params['model_dir'],
        min_interval_secs=FLAGS.min_eval_interval,
        timeout=FLAGS.eval_timeout,
        timeout_fn=terminate_eval):

      tf.logging.info('Starting to evaluate.')
      try:

        eval_results = runner.evaluate(ckpt)
        current_step, _ = runner.get_step_and_epoch_number(ckpt)

        if (eval_results['AP'] >= mask_rcnn_params.BOX_EVAL_TARGET and
            eval_results['mask_AP'] >= mask_rcnn_params.MASK_EVAL_TARGET):
          mlp_log.mlperf_print(key='run_stop', metadata={'status': 'success'})
          run_success = True
          break

        if int(current_step) >= params['total_steps']:
          tf.logging.info('Evaluation finished after training step %d' %
                          current_step)
          break

      except tf.errors.NotFoundError:
        # Since the coordinator is on a different job than the TPU worker,
        # sometimes the TPU worker does not finish initializing until long after
        # the CPU job tells it to start evaluating. In this case, the checkpoint
        # file could have been deleted already.
        tf.logging.info('Checkpoint %s no longer exists, skipping checkpoint' %
                        ckpt)
    if not run_success:
      mlp_log.mlperf_print(key='run_stop', metadata={'status': 'aborted'})

  elif FLAGS.mode == 'train_and_eval':
    runner.train_and_eval()
  else:
    tf.logging.info('Mode not found.')


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
