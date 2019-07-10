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
from six.moves import queue as Queue

import tensorflow as tf

from tensorflow.contrib import tpu
from tensorflow.contrib.tpu.python.tpu import tpu_function
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.data.util import nest as data_nest
from tensorflow.python.framework import graph_io
from utils import low_level_utils

FLAGS = flags.FLAGS

_INITIAL_LOSS = 1e7
_STOP = -1


class TrainLowLevelRunner(object):
  """Run Train via direct session.run calls."""

  def __init__(self, iterations):
    tf.logging.info("TrainLowLevelRunner: constructor")

    self.feature_structure = {}
    self.losses = []
    self.infeed_queue = []
    self.enqueue_ops = []
    self.dataset_initializer = []
    self.iterations = iterations
    self.num_hosts = FLAGS.tpu_num_shards // FLAGS.tpu_num_shards_per_host
    self.scaffold_fn = None
    # Having two separate sessions and graphs to make the initialization faster.
    self.input_sess = None
    self.train_sess = None
    self.input_graph = tf.Graph()
    self.train_graph = None
    self.tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.master or FLAGS.cloud_tpu_name)
    # Disable grappler for better performance.
    self.session_config = tf.ConfigProto(
        allow_soft_placement=True,
        graph_options=tf.GraphOptions(
            rewrite_options=rewriter_config_pb2.RewriterConfig(
                disable_meta_optimizer=True)),
        isolate_session_state=True)
    cluster_spec = self.tpu_cluster_resolver.cluster_spec()
    if cluster_spec:
      self.session_config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
    self.tpu_init = [tpu.initialize_system()]
    self.tpu_shutdown = tpu.shutdown_system()
    self.init_sess = tf.Session(self.tpu_cluster_resolver.get_master(),
                                config=self.session_config)
    self.init_sess.run(self.tpu_init)
    self.queue = Queue.Queue()

  def shutdown(self):
    """Shut down TrainLowLevelRunner."""
    tf.logging.info("TrainLowLevelRunner: shutdown")
    self.queue.put(_STOP)
    self.infeed_thread.join()
    self.input_sess.close()
    self.train_sess.close()

  def _get_host(self, host_id):
    if self.tpu_cluster_resolver.get_master() in ("", "local"):
      return "/replica:0/task:0"
    job_name = self.tpu_cluster_resolver.get_job_name() or "tpu_worker"
    return "/job:%s/task:%d" % (job_name, host_id)

  def build_enqueue_ops(self, input_fn, params, config, host_id):
    """Build enqueue ops."""
    tf.logging.info("TrainLowLevelRunner: build_enqueue_ops")

    def get_enqueue_ops_fn(host_id):
      """Generate the enqueue ops graph function."""
      with tf.device(low_level_utils.device_for_host(self._get_host(host_id))):
        dataset = input_fn(params, config)
        iterator = dataset.make_initializable_iterator()
        self.dataset_initializer.append(iterator.initializer)

        def enqueue_ops_fn():
          """Enqueue ops function for one host."""
          per_host_sharded_inputs = []
          control_deps = []

          if FLAGS.broadcast_input_all_replicas:
            features, labels = iterator.get_next()
            self.feature_structure["features"] = features
            self.feature_structure["labels"] = labels
            flattened_inputs = data_nest.flatten(self.feature_structure)
            for _ in range(FLAGS.tpu_num_shards_per_host):
              per_host_sharded_inputs.append(flattened_inputs)
          else:
            for _ in range(FLAGS.tpu_num_shards_per_host):
              with tf.control_dependencies(control_deps):
                features, labels = iterator.get_next()
              self.feature_structure["features"] = features
              self.feature_structure["labels"] = labels
              flattened_inputs = data_nest.flatten(self.feature_structure)
              control_deps.extend(flattened_inputs)
              per_host_sharded_inputs.append(flattened_inputs)

          infeed = tpu.InfeedQueue(
              number_of_tuple_elements=len(per_host_sharded_inputs[0]))
          self.infeed_queue.append(infeed)
          return infeed.generate_enqueue_ops(
              per_host_sharded_inputs,
              tpu_ordinal_function=low_level_utils.tpu_ordinal_fn)

        return enqueue_ops_fn

    with self.input_graph.as_default():
      if FLAGS.random_seed:
        tf.random.set_random_seed(FLAGS.random_seed)
      self.enqueue_ops.append(
          low_level_utils.wrap_computation_in_while_loop(
              get_enqueue_ops_fn(host_id),
              n=self.iterations,
              host_name=self._get_host(host_id)))

  def initialize(self, input_fn, model_fn, params, hparams, config):  # pylint: disable=unused-argument
    """Build graph and do initialization for training."""
    tf.logging.info("TrainLowLevelRunner: initialize method")

    self.build_enqueue_ops(input_fn, params, config, host_id=0)

    def infeed_thread_fn():
      """Build and infeed session.run calls in a background thread."""
      for i in range(1, self.num_hosts):
        self.build_enqueue_ops(input_fn, params, config, host_id=i)
      # Build infeed sesssion
      self.input_sess = tf.Session(
          self.tpu_cluster_resolver.get_master(),
          graph=self.input_graph,
          config=self.session_config)
      # Initialize dataset variables
      self.input_sess.run(self.dataset_initializer)
      # Run infeed session.run calls
      while True:
        iterations = self.queue.get(block=True)
        if iterations == _STOP:
          return
        tf.logging.info("Start to infeed %d batches", iterations)
        self.input_sess.run([self.enqueue_ops])

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
      return tpu.repeat(self.iterations, tpu_train_step, [_INITIAL_LOSS])

    self.train_graph = tf.Graph()
    with self.train_graph.as_default():
      if FLAGS.random_seed:
        tf.random.set_random_seed(FLAGS.random_seed)
      self.losses = tpu.shard(
          train_loop,
          inputs=[],
          num_shards=FLAGS.tpu_num_shards,
          outputs_from_all_shards=True,
      )
      if self.scaffold_fn:
        self.scaffold_fn()
      global_initializer = tf.global_variables_initializer()
      local_initializer = tf.local_variables_initializer()
      graph_io.write_graph(
          self.input_graph.as_graph_def(add_shapes=True), FLAGS.output_dir,
          "input_graph.pbtxt")
      graph_io.write_graph(
          self.train_graph.as_graph_def(add_shapes=True), FLAGS.output_dir,
          "graph.pbtxt")
      self.saver = tf.train.Saver()

    # Build tpu train model session and initialize graph
    self.train_sess = tf.Session(
        self.tpu_cluster_resolver.get_master(),
        graph=self.train_graph,
        config=self.session_config)

    self.train_sess.run(global_initializer)
    self.train_sess.run(local_initializer)

    # Complete infeed graph generation and session.run calls
    self.infeed_thread = threading.Thread(target=infeed_thread_fn)
    self.infeed_thread.start()

  def train(self, train_steps, local_batch_size, base_step=0, num_threads=2):
    """Run the Train loop on the TPU device."""
    tf.logging.info("TrainLowLevelRunner: train for %d steps in total",
                    train_steps)
    if train_steps % self.iterations != 0:
      tf.logging.warning(
          "train_steps %d is not divisible by iterations_per_loop %d",
          train_steps, self.iterations)
      train_steps = self.iterations * int(
          math.ceil(train_steps / self.iterations))

    def checkpoint_thread_fn(saver, sess):
      saver.save(sess,
                 FLAGS.output_dir + "/model.ckpt-%d" % (cur_step + base_step))

    cur_step = 0
    thread_id = 0
    checkpoint_threads = []
    for i in range(num_threads):
      checkpoint_threads.append(None)

    while cur_step < train_steps:
      start = time.time()
      tf.logging.info("TrainLowLevelRunner: start train step:%d", cur_step)
      self.queue.put(self.iterations)
      cur_step += self.iterations

      losses = self.train_sess.run(self.losses)
      tf.logging.info("TrainLowLevelRunner: sess run loss: %s", losses)

      if checkpoint_threads[thread_id] is not None:
        checkpoint_threads[thread_id].join()
      checkpoint_threads[thread_id] = threading.Thread(
          target=checkpoint_thread_fn, args=(self.saver, self.train_sess))
      checkpoint_threads[thread_id].start()
      thread_id += 1
      if thread_id >= num_threads:
        thread_id = 0

      end = time.time()
      tf.logging.info(
          "TrainLowLevelRunner: step time {} sec {} examples/sec".format(
              end - start,
              (self.iterations * local_batch_size * FLAGS.tpu_num_shards /
               (end - start))))

    for i in range(num_threads):
      if checkpoint_threads[i] is not None:
        checkpoint_threads[i].join()
        checkpoint_threads[i] = None
