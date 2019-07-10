# Copyright 2018 Google. All Rights Reserved.
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
"""Training SSD with low level API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import threading
import time
from absl import flags
import six

import tensorflow as tf

from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.data.util import nest as data_nest
from tensorflow.python.framework import graph_io
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_feed
from tensorflow.python.tpu import tpu_function
from tensorflow.python.tpu import training_loop
from tensorflow.python.tpu.ops import tpu_ops
from mlp_log import mlp_log
from utils import low_level_utils

FLAGS = flags.FLAGS

_INITIAL_LOSS = 1e7
_STOP = -1
_ITEM = 1


class LowLevelRunner(object):
  """Run Train and Eval via direct session.run calls."""

  def __init__(self, iterations, eval_steps):
    tf.logging.info("LowLevelRunner: constructor.")

    self.fake_feature_structure = {}
    self.feature_structure = {}
    self.fake_eval_feature_structure = {}
    self.eval_feature_structure = {}
    self.infeed_queue = []
    self.eval_infeed_queue = []
    self.fake_enqueue_ops = []
    self.enqueue_ops = []
    self.fake_eval_enqueue_ops = []
    self.eval_enqueue_ops = []
    self.fake_dataset_initializer = []
    self.dataset_initializer = []
    self.fake_eval_dataset_initializer = []
    self.eval_dataset_initializer = []
    self.outfeed_tensors = []
    self.outfeed_names = []
    self.dequeue_ops = []
    self.train_compile_op = None
    self.eval_compile_op = None
    self.loss = None
    self.eval_op = None
    self.iterations = iterations
    self.eval_steps = eval_steps
    self.num_hosts = FLAGS.tpu_num_shards // FLAGS.tpu_num_shards_per_host
    self.scaffold_fn = None
    self.tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.master or FLAGS.cloud_tpu_name)
    # Disable grappler for better performance.
    self.session_config = tf.ConfigProto(
        allow_soft_placement=True,
        graph_options=tf.GraphOptions(
            rewrite_options=rewriter_config_pb2.RewriterConfig(
                disable_meta_optimizer=True)),
        isolate_session_state=True,
        operation_timeout_in_ms=600 * 60 * 1000)  # 10 hours
    cluster_spec = self.tpu_cluster_resolver.cluster_spec()
    if cluster_spec:
      self.session_config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
    self.input_graph = tf.Graph()
    self.eval_input_graph = tf.Graph()
    # Train and eval share the same session and graph so that the weights
    # can be shared for in memory eval.
    self.graph = tf.Graph()
    self.output_graph = tf.Graph()
    with self.graph.as_default():
      if FLAGS.random_seed:
        tf.random.set_random_seed(FLAGS.random_seed)
      self.num_epochs_tensor = tf.placeholder(
          tf.int32, shape=(), name="epochs")
      self.train_steps_tensor = tf.placeholder(
          tf.int32, shape=(), name="steps_per_train_loop")
      self.eval_steps_tensor = tf.placeholder(
          tf.int32, shape=(), name="steps_per_eval_loop")
      self.tpu_init = [tpu.initialize_system()]
      self.tpu_shutdown = tpu.shutdown_system()
    self.master = self.tpu_cluster_resolver.get_master()
    self.input_sess = tf.Session(
        self.master,
        graph=self.input_graph,
        config=self.session_config)
    self.eval_input_sess = tf.Session(
        self.master,
        graph=self.eval_input_graph,
        config=self.session_config)
    self.sess = tf.Session(
        self.master,
        graph=self.graph,
        config=self.session_config)
    self.output_sess = tf.Session(
        self.master,
        graph=self.output_graph,
        config=self.session_config)
    self.sess.run(self.tpu_init)
    self.infeed_thead = None
    self.train_eval_thead = None

  def shutdown(self):
    """Shut down LowLevelRunner."""
    tf.logging.info("LowLevelRunner: shutdown.")
    self.infeed_thread.join()
    self.train_eval_thread.join()
    self.input_sess.close()
    self.eval_input_sess.close()
    self.sess.close()
    self.output_sess.close()

  def _get_host(self, host_id):
    if self.master in ("", "local"):
      return "/replica:0/task:0"
    job_name = self.tpu_cluster_resolver.get_job_name() or "tpu_worker"
    return "/job:%s/task:%d" % (job_name, host_id)

  def build_enqueue_ops(self,
                        input_fn,
                        dataset_initializer,
                        feature_structure,
                        infeed_queue,
                        enqueue_ops,
                        steps,
                        params,
                        config,
                        host_id,
                        graph,
                        task):
    """Build enqueue ops for training."""
    tf.logging.info(
        "LowLevelRunner: build enqueue ops for %s for host %d.", task, host_id)

    def get_enqueue_ops_fn(host_id):
      """Generate the enqueue ops graph function."""
      with tf.device(low_level_utils.device_for_host(self._get_host(host_id))):
        dataset = input_fn(params, config)
        iterator = dataset.make_initializable_iterator()
        dataset_initializer.append(iterator.initializer)

        def enqueue_ops_fn():
          """Enqueue ops function for one host."""
          per_host_sharded_inputs = []
          control_deps = []
          for _ in range(FLAGS.tpu_num_shards_per_host):
            if "eval" in task:
              with tf.control_dependencies(control_deps):
                features = iterator.get_next()
              feature_structure["features"] = features
            else:
              with tf.control_dependencies(control_deps):
                features, labels = iterator.get_next()
              feature_structure["features"] = features
              feature_structure["labels"] = labels
            flattened_inputs = data_nest.flatten(feature_structure)
            control_deps.extend(flattened_inputs)
            per_host_sharded_inputs.append(flattened_inputs)

          infeed = tpu_feed.InfeedQueue(
              number_of_tuple_elements=len(per_host_sharded_inputs[0]))
          infeed_queue.append(infeed)
          return infeed.generate_enqueue_ops(
              per_host_sharded_inputs,
              tpu_ordinal_function=low_level_utils.tpu_ordinal_fn)

        return enqueue_ops_fn

    with graph.as_default():
      enqueue_ops.append(
          low_level_utils.wrap_computation_in_while_loop(
              get_enqueue_ops_fn(host_id),
              n=steps,
              host_name=self._get_host(host_id)))

  def build_model(self, model_fn, eval_model_fn, params, hparams, config):
    """Build the TPU model for training and eval."""
    tf.logging.info("LowLevelRunner: build_model method for training and eval.")

    def tpu_train_step(loss):
      """Generate the TPU graph."""
      del loss
      values = self.infeed_queue[0].generate_dequeue_op(tpu_device=0)
      unflattened_inputs = data_nest.pack_sequence_as(self.feature_structure,
                                                      values)
      features = unflattened_inputs["features"]
      labels = unflattened_inputs["labels"]
      estimator_spec = model_fn(
          features,
          labels,
          tf.estimator.ModeKeys.TRAIN,
          params=params,
          config=config)
      loss, train_op = estimator_spec.loss, estimator_spec.train_op
      self.scaffold_fn = estimator_spec.scaffold_fn
      with tf.control_dependencies([train_op]):
        return tf.identity(loss)

    @tpu_function.on_device_training_loop
    def train_loop():
      return training_loop.repeat(
          self.train_steps_tensor, tpu_train_step, [_INITIAL_LOSS])

    def tpu_eval_step():
      """Generate the TPU graph."""
      values = self.eval_infeed_queue[0].generate_dequeue_op(tpu_device=0)
      unflattened_inputs = data_nest.pack_sequence_as(
          self.eval_feature_structure, values)
      features = unflattened_inputs["features"]
      estimator_spec = eval_model_fn(
          features,
          None,
          tf.estimator.ModeKeys.PREDICT,
          params=params,
          config=config)
      for k, v in six.iteritems(estimator_spec.predictions):
        self.outfeed_names.append(k)
        self.outfeed_tensors.append(v)

      with tf.device(low_level_utils.device_for_tpu_core(self._get_host(0))):
        outfeed_enqueue_ops = tpu_ops.outfeed_enqueue_tuple(
            self.outfeed_tensors)
      with tf.control_dependencies([outfeed_enqueue_ops]):
        return tf.no_op()

    @tpu_function.on_device_training_loop
    def eval_loop():
      return training_loop.repeat(self.eval_steps_tensor, tpu_eval_step, [])

    def train_eval_step():
      with tf.control_dependencies(train_loop()):
        return eval_loop()

    @tpu_function.on_device_training_loop
    def train_eval_loop():
      return training_loop.repeat(self.num_epochs_tensor, train_eval_step, [])

    with self.graph.as_default():
      (self.compile_op, self.train_eval_op,) = tpu.split_compile_and_shard(
          train_eval_loop,
          inputs=[],
          num_shards=FLAGS.tpu_num_shards,
          outputs_from_all_shards=False,
      )
      if self.scaffold_fn:
        self.scaffold_fn()
      self.sess.run(tf.global_variables_initializer())
      self.sess.run(tf.local_variables_initializer())

      graph_io.write_graph(
          self.graph.as_graph_def(add_shapes=True),
          FLAGS.output_dir,
          "graph.pbtxt")

    def create_dequeue_ops(host_id):
      """Create outfeed dequeue ops."""
      dequeue_ops = []
      tensor_dtypes = []
      tensor_shapes = []
      for v in self.outfeed_tensors:
        dequeue_ops.append([])
        tensor_dtypes.append(v.dtype)
        tensor_shapes.append(v.shape)
      for i in range(FLAGS.tpu_num_shards_per_host):
        with tf.device(
            low_level_utils.device_for_host(self._get_host(host_id))):
          outfeed = tpu_ops.outfeed_dequeue_tuple(
              dtypes=tensor_dtypes, shapes=tensor_shapes, device_ordinal=i)
          for j, item in enumerate(outfeed):
            dequeue_ops[j].append(item)
      for j in range(len(outfeed)):
        dequeue_ops[j] = tf.concat(dequeue_ops[j], axis=0)
      return dequeue_ops

    with self.output_graph.as_default():
      # Get dequeue ops from each hosts.
      for i in range(self.num_hosts):
        tf.logging.info(
            "LowLevelRunner: get dequeue ops for host: %d.", i)
        local_batch_size = hparams.batch_size // self.num_hosts
        local_dequeue_ops = []
        for n in range(local_batch_size):
          local_dequeue_ops.append({})
        for j, dequeue_tensor in enumerate(create_dequeue_ops(i)):
          if self.outfeed_names[j] in ("inputs", "targets", "outputs"):
            dequeue_tensors = tf.split(
                dequeue_tensor, local_batch_size, axis=0)
            for n in range(local_batch_size):
              local_dequeue_ops[n][self.outfeed_names[j]] = dequeue_tensors[n]
        for j, dequeue_dict in enumerate(local_dequeue_ops):
          self.dequeue_ops.append(dequeue_dict)

  def initialize(self,
                 fake_input_fn,
                 fake_eval_input_fn,
                 input_fn,
                 eval_input_fn,
                 model_fn,
                 eval_model_fn,
                 params,
                 hparams,
                 config):
    """Build graph and do initialization."""
    tf.logging.info("LowLevelRunner: initialize method.")

    # Build enqueue ops.
    for i in range(self.num_hosts):
      self.build_enqueue_ops(fake_input_fn,
                             self.fake_dataset_initializer,
                             self.fake_feature_structure,
                             self.infeed_queue,
                             self.fake_enqueue_ops,
                             1,
                             params,
                             config,
                             host_id=i,
                             graph=self.input_graph,
                             task="training warm-up")
      self.build_enqueue_ops(fake_eval_input_fn,
                             self.fake_eval_dataset_initializer,
                             self.fake_eval_feature_structure,
                             self.eval_infeed_queue,
                             self.fake_eval_enqueue_ops,
                             1,
                             params,
                             config,
                             host_id=i,
                             graph=self.eval_input_graph,
                             task="eval warm-up")
      self.build_enqueue_ops(input_fn,
                             self.dataset_initializer,
                             self.feature_structure,
                             self.infeed_queue,
                             self.enqueue_ops,
                             self.iterations,
                             params,
                             config,
                             host_id=i,
                             graph=self.input_graph,
                             task="training")
      self.build_enqueue_ops(eval_input_fn,
                             self.eval_dataset_initializer,
                             self.eval_feature_structure,
                             self.eval_infeed_queue,
                             self.eval_enqueue_ops,
                             self.eval_steps,
                             params,
                             config,
                             host_id=i,
                             graph=self.eval_input_graph,
                             task="eval")

    # Build model.
    self.build_model(model_fn, eval_model_fn, params, hparams, config)

    # Compile.
    self.sess.run([self.compile_op])

    # Warm up 1 step to avoid program starting overheads.
    self.input_sess.run(self.fake_dataset_initializer)
    self.eval_input_sess.run(self.fake_eval_dataset_initializer)

    def infeed_thread_fn(sess, eval_sess, enqueue_ops, eval_enqueue_ops):
      """Build and infeed session.run calls in a background thread."""
      tf.logging.info(
          "Start to infeed %d batches for warmup.", 1)
      sess.run([enqueue_ops])
      eval_sess.run([eval_enqueue_ops])

    infeed_thread = threading.Thread(
        target=infeed_thread_fn,
        args=(self.input_sess,
              self.eval_input_sess,
              self.fake_enqueue_ops,
              self.fake_eval_enqueue_ops))
    infeed_thread.start()
    self.sess.run([self.train_eval_op],
                  feed_dict={self.num_epochs_tensor: 1,
                             self.train_steps_tensor: 1,
                             self.eval_steps_tensor: 1})
    infeed_thread.join()

    self.output_sess.run(self.dequeue_ops)

    # Re-initialize variables.
    with self.graph.as_default():
      self.sess.run(tf.global_variables_initializer())
      self.sess.run(tf.local_variables_initializer())

    # Initialize dataset variables for training.
    self.input_sess.run(self.dataset_initializer)

  def train_and_eval(self,
                     train_steps,
                     local_batch_size,  # pylint: disable=unused-argument
                     num_threads=2):  # pylint: disable=unused-argument
    """Run the training loop on the TPU device."""
    tf.logging.info("LowLevelRunner: train for %d steps in total.",
                    train_steps)

    if train_steps % self.iterations != 0:
      tf.logging.warning(
          "train_steps %d is not divisible by iterations_per_loop %d",
          train_steps, self.iterations)
      train_steps = self.iterations * int(
          math.ceil(train_steps / self.iterations))

    # Train and eval/predict thread.
    def train_eval_thread_fn(sess, train_eval_op, steps):
      sess.run([train_eval_op],
               feed_dict={self.num_epochs_tensor: steps,
                          self.train_steps_tensor: self.iterations,
                          self.eval_steps_tensor: self.eval_steps})

    self.train_eval_thread = threading.Thread(
        target=train_eval_thread_fn,
        args=(self.sess, self.train_eval_op, train_steps // self.iterations))
    self.train_eval_thread.start()

    # Infeed thread.
    def infeed_thread_fn(sess,
                         eval_sess,
                         enqueue_ops,
                         eval_enqueue_ops,
                         eval_dataset_initializer):
      """Build and infeed session.run calls in a background thread."""
      for i in range(train_steps // self.iterations):
        mlp_log.mlperf_print(
            "block_start",
            None,
            metadata={
                "first_epoch_num": i + 1,
                "epoch_count": 1
            })
        tf.logging.info(
            "Start to infeed %d batches for training of epoch %d.",
            self.iterations, i)
        sess.run([enqueue_ops])
        eval_sess.run(eval_dataset_initializer)
        eval_sess.run([eval_enqueue_ops])

    self.infeed_thread = threading.Thread(
        target=infeed_thread_fn,
        args=(self.input_sess,
              self.eval_input_sess,
              self.enqueue_ops,
              self.eval_enqueue_ops,
              self.eval_dataset_initializer))
    time.sleep(240)
    mlp_log.mlperf_print(key="init_stop", value=None)
    mlp_log.mlperf_print(key="run_start", value=None)
    self.infeed_thread.start()

  def dequeue(self, decode_hparams):  # pylint: disable=unused-argument
    """Dequeue the prediction results."""
    ret = []
    for step in range(self.eval_steps):
      tf.logging.info("LowLevelRunner: start eval step: %d.", step)
      for item in self.output_sess.run(self.dequeue_ops):
        ret.append(item)
    return ret
