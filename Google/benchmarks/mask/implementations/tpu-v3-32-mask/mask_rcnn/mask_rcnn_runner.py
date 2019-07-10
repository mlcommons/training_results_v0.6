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
"""Runner class for Mask-RCNN."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import os
import time
import tensorflow as tf

import eval_low_level_runner
import eval_multiprocess
import train_and_eval_low_level_runner
import train_low_level_runner
from mlp_log import mlp_log


class EvalRunner(object):
  """A class that supports TPUEstimator or low-level API runner for eval."""

  def __init__(self, model_fn, input_fn, eval_metric, params, run_config,
               use_tpu_estimator=True):
    self.model_fn = model_fn
    self.input_fn = input_fn
    self.eval_batch_size = params['eval_batch_size']
    # We use one extra eval step in addition to the ceiling of
    # eval_samples/eval_batch_size to protect from partial batch due to
    # horizontal/vertical image grouping.
    self.eval_samples = params['eval_samples']
    self.eval_steps = int(math.ceil(self.eval_samples /
                                    self.eval_batch_size)) + 1
    self.model_dir = params['model_dir']
    output_dir = os.path.join(self.model_dir, 'eval')
    tf.gfile.MakeDirs(output_dir)
    # Summary writer writes out eval metrics.
    self.summary_writer = tf.summary.FileWriter(output_dir)
    self.eval_metric = eval_metric
    self.params = params
    self.run_config = run_config
    self.use_tpu = params['use_tpu']
    self.use_tpu_estimator = use_tpu_estimator
    if self.use_tpu_estimator:
      self.runner = tf.contrib.tpu.TPUEstimator(
          model_fn=self.model_fn,
          use_tpu=self.use_tpu,
          train_batch_size=1,  # no effect.
          predict_batch_size=self.eval_batch_size,
          config=self.run_config,
          params=self.params)
    else:
      self.params['batch_size'] = (self.params['eval_batch_size'] //
                                   self.params['num_shards'])

      self.runner = eval_low_level_runner.EvalLowLevelRunner(
          self.run_config.cluster, self.params['eval_batch_size'],
          self.eval_steps, self.params['replicas_per_worker'],
          self.run_config.tpu_config.tpu_job_name)
      input_fn = functools.partial(
          self.input_fn,
          num_examples=self.eval_steps * self.params['eval_batch_size'])
      self.runner.initialize(input_fn, self.model_fn, self.params)

  def __del__(self):
    """Shut down."""
    self.summary_writer.close()
    if not self.use_tpu_estimator:
      self.runner.shutdown()

  def get_step_and_epoch_number(self, ckpt):
    """Calculates step and epoch number."""
    current_step = int(os.path.basename(ckpt).split('-')[1])
    steps_per_epoch = (self.params['num_examples_per_epoch'] //
                       self.params['train_batch_size'])
    # The epoch number is zero-indexed.
    return current_step, (current_step // steps_per_epoch - 1)

  def write_summary(self, eval_results, current_step):
    """Write out eval results for the checkpoint."""
    with tf.Graph().as_default():
      summaries = []
      for metric in eval_results:
        summaries.append(
            tf.Summary.Value(
                tag=metric, simple_value=eval_results[metric]))
      tf_summary = tf.Summary(value=list(summaries))
      self.summary_writer.add_summary(tf_summary, current_step)

  def evaluate(self, ckpt):
    """Performs evaluation against `ckpt` and writes a summary to directory."""
    current_step, num_epochs = self.get_step_and_epoch_number(ckpt)
    mlp_log.mlperf_print(
        'eval_start', None, metadata={'epoch_num': num_epochs})
    eval_begin = time.time()
    if self.use_tpu_estimator:
      input_fn = functools.partial(
          self.input_fn,
          num_examples=self.eval_steps * self.params['eval_batch_size'])
      predictor = self.runner.predict(
          input_fn=input_fn,
          checkpoint_path=ckpt,
          yield_single_examples=False)
    else:
      predictor = self.runner.predict(checkpoint_path=ckpt,
                                      eval_steps=self.eval_steps)

    # Enables multi-processing to accelerate post-processing.
    eval_multiprocess.eval_multiprocessing(self.eval_steps, predictor,
                                           self.eval_metric,
                                           self.params['eval_worker_count'])

    pred_end = time.time()
    tf.logging.info('prediction takes %d seconds.', pred_end - eval_begin)
    num_eval_samples, eval_results = self.eval_metric.evaluate()

    eval_end = time.time()
    tf.logging.info('COCO evaluates %d samples', num_eval_samples)
    assert num_eval_samples == self.params['eval_samples']
    tf.logging.info('one evaluation takes %d seconds', eval_end - eval_begin)
    self.write_summary(eval_results, current_step)
    tf.logging.info('AP: %s' % eval_results['AP'])
    tf.logging.info('mask_AP: %s' % eval_results['mask_AP'])
    mlp_log.mlperf_print(
        'eval_stop', None, metadata={'epoch_num': num_epochs})
    # TODO(b/127959551): use both metrics once the bug is resolved.
    mlp_log.mlperf_print(
        'eval_accuracy', (float(eval_results['AP']),
                          float(eval_results['mask_AP'])),
        metadata={'epoch_num': num_epochs})

    return eval_results


class TrainRunner(object):
  """A class that supports TPUEstimator or low-level API runner for training."""

  def __init__(self,
               model_fn,
               input_fn,
               params,
               run_config,
               use_tpu_estimator=True):
    self.input_fn = input_fn
    self.params = params
    self.use_tpu_estimator = use_tpu_estimator
    if use_tpu_estimator:
      self.runner = tf.contrib.tpu.TPUEstimator(
          model_fn=model_fn,
          use_tpu=self.params['use_tpu'],
          train_batch_size=self.params['train_batch_size'],
          config=run_config,
          params=self.params)
    else:
      self.params['batch_size'] = (
          self.params['train_batch_size'] // self.params['num_shards'])
      self.runner = train_low_level_runner.TrainLowLevelRunner(
          run_config.cluster, self.params,
          run_config.tpu_config.input_partition_dims,
          run_config.tpu_config.tpu_job_name)
      self.runner.initialize(model_fn, input_fn)

  def __del__(self):
    """Shut down."""
    if not self.use_tpu_estimator:
      self.runner.shutdown()

  def train(self):
    """Run the train loops and write a summary to directory."""
    mlp_log.mlperf_print(key='init_stop', value=None)
    mlp_log.mlperf_print(key='run_start', value=None)
    if self.use_tpu_estimator:
      self.runner.train(
          input_fn=self.input_fn, max_steps=self.params['total_steps'])
    else:
      self.runner.train()


class TrainEvalRunner(object):
  """A class that supports low-level API runner for train_and_eval."""

  def __init__(self, model_fn, input_fn, eval_input_fn, eval_metric,
               train_params, eval_params, run_config):
    num_cores_per_replica = (
        run_config.tpu_config.num_cores_per_replica
        if run_config.tpu_config.num_cores_per_replica else 1)
    eval_batch_size = eval_params['eval_batch_size']
    # We use three one eval step in addition to the ceiling of
    # eval_samples/eval_batch_size to protect from partial batch due to
    # horizontal/vertical image bucketizing and make sure distributed eval with
    # spatial partition could use all eval samples.
    eval_steps = int(math.ceil(
        eval_params['eval_samples'] / eval_batch_size)) + 1
    self.runner = train_and_eval_low_level_runner.TrainEvalLowLevelRunner(
        run_config.cluster, train_params, eval_params, eval_steps, eval_metric,
        run_config.tpu_config.input_partition_dims, num_cores_per_replica,
        run_config.tpu_config.tpu_job_name)
    eval_input_fn = functools.partial(
        eval_input_fn, num_examples=eval_steps * eval_batch_size)
    self.runner.initialize(model_fn, input_fn, eval_input_fn)

  def __del__(self):
    """Shut down."""
    self.runner.shutdown()

  def train_and_eval(self):
    """Performs training and distributed eval."""
    self.runner.train_and_eval()
