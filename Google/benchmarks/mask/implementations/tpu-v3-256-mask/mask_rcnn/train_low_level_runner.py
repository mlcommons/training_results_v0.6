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
"""Training with low level API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import threading
import time
from six.moves import queue as Queue

import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import device_assignment as tpu_device_assignment
from tensorflow.contrib.tpu.python.tpu import tpu
from tensorflow.contrib.tpu.python.tpu import tpu_feed
from tensorflow.contrib.tpu.python.tpu import tpu_function
from tensorflow.python.framework import graph_io
import runner_utils


_INITIAL_LOSS = 1e7
_STOP = -1
_MAX_NUM_CHECKPOINT_THREADS = 1
# for spatial partition
_NUM_CORES_TO_COMPUTATION_SHAPE = {
    1: [1, 1, 1],
    2: [1, 1, 2],
    4: [1, 2, 2],
    8: [2, 2, 2],
    16: [4, 2, 2],
}


class TrainLowLevelRunner(object):
  """Run Train via direct session.run calls."""

  def __init__(self, tpu_cluster_resolver, params, input_partition_dims=None,
               tpu_job_name=None):
    tf.logging.info("TrainLowLevelRunner: constructor")

    self.tpu_cluster_resolver = tpu_cluster_resolver
    self.params = params
    self.tpu_job_name = tpu_job_name

    self.model_dir = params["model_dir"]
    self.iterations_per_loop = params["iterations_per_loop"]
    self.num_shards = self.params["num_shards"]
    self.input_flattener = runner_utils.InputsFlattener()
    self.feature_structure = {}
    self.train_compile_op = None
    self.train_op = None
    self.infeed_queue = []
    self.enqueue_ops = []
    self.dataset_initializer = []
    self.scaffold_fn = None
    # Having two separate sessions and graphs to make the initialization faster.
    self.input_sess = None
    self.train_sess = None
    self.input_graph = tf.Graph()
    self.train_graph = None
    self.session_config = tf.ConfigProto(
        allow_soft_placement=True, isolate_session_state=True,
        operation_timeout_in_ms=600 * 60 * 1000)  # 10 hours
    cluster_spec = self.tpu_cluster_resolver.cluster_spec()
    if cluster_spec:
      self.session_config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
    self.tpu_init = tf.contrib.tpu.initialize_system()
    self.tpu_shutdown = tf.contrib.tpu.shutdown_system()
    self.init_sess = tf.Session(self.tpu_cluster_resolver.get_master(),
                                config=self.session_config)
    self.device_topology = self.init_sess.run(self.tpu_init)
    self.input_partition_dims = input_partition_dims
    self.use_spatial_partition = input_partition_dims is not None
    self.num_cores_per_replica = (
        self.params["num_cores_per_replica"]
        if self.params["num_cores_per_replica"] else 1)
    if self.use_spatial_partition:
      computation_shape = _NUM_CORES_TO_COMPUTATION_SHAPE[
          self.num_cores_per_replica]
      self.device_assignment = tpu_device_assignment.device_assignment(
          topology=self.device_topology,
          computation_shape=computation_shape,
          num_replicas=self.num_shards)
      tf.logging.info("num_cores_per_replica: %d", self.num_cores_per_replica)
      tf.logging.info("computation_shape: %s", str(computation_shape))
      tf.logging.info("num_replicas: %d", self.num_shards)
      tf.logging.info("device_assignment.topology.device_coordinates: %s",
                      str(self.device_assignment.topology.device_coordinates))
      tf.logging.info("device_assignment.core_assignment: %s",
                      str(self.device_assignment.core_assignment))
      self.input_dims_flattener = runner_utils.InputDimsFlattener(
          self.input_partition_dims)
    else:
      self.device_assignment = None
      self.input_dims_flattener = None
    self.queue = Queue.Queue()
    # Summary writer writes out train metrics.
    self.summary_writer = tf.summary.FileWriter(self.model_dir)
    self.infeed_thread = None

  def shutdown(self):
    """Shut down TrainLowLevelRunner."""
    tf.logging.info("TrainLowLevelRunner: shutdown")
    self.queue.put(_STOP)
    if self.infeed_thread:
      self.infeed_thread.join()
    if self.input_sess:
      self.input_sess.close()
    if self.train_sess:
      self.train_sess.close()
    self.summary_writer.close()

  def _get_host(self, host_id):
    if self.tpu_cluster_resolver.get_master() in ("", "local"):
      return "/replica:0/task:0"
    job_name = (
        self.tpu_job_name or self.tpu_cluster_resolver.get_job_name() or
        "tpu_worker")
    return "/job:%s/task:%d" % (job_name, host_id)

  def build_enqueue_ops(self, input_fn, params, num_hosts, host_id):
    """Build enqueue ops."""
    tf.logging.info("TrainLowLevelRunner: build_enqueue_ops for %d", host_id)

    def get_enqueue_ops_fn(host_id):
      """Generate the enqueue ops graph function."""
      #  TODO(b/129084726): make dataset sharding also work for TPU Estimator.
      params["dataset_num_shards"] = num_hosts
      params["dataset_shard_id"] = host_id
      with tf.device(runner_utils.device_for_host(self._get_host(host_id))):
        dataset = input_fn(params)
        iterator = dataset.make_initializable_iterator()
        self.dataset_initializer.append(iterator.initializer)

        def enqueue_ops_fn():
          """Enqueue ops function for one host."""
          per_host_sharded_inputs = []
          control_deps = []
          for _ in range(self.params["replicas_per_worker"]):
            with tf.control_dependencies(control_deps):
              features, labels = iterator.get_next()
            if self.use_spatial_partition:
              self.input_dims_flattener.validate_and_flatten_input_dims(
                  features, labels)
            flattened_inputs = self.input_flattener.flatten_features_and_labels(
                features, labels)
            control_deps.extend(flattened_inputs)
            per_host_sharded_inputs.append(flattened_inputs)

          if self.use_spatial_partition:
            flattened_input_dims = (
                self.input_dims_flattener.flattened_input_dims)
            # pylint: disable=protected-access
            infeed = tpu_feed._PartitionedInfeedQueue(
                number_of_tuple_elements=len(per_host_sharded_inputs[0]),
                host_id=host_id,
                input_partition_dims=flattened_input_dims,
                device_assignment=self.device_assignment)
            self.infeed_queue.append(infeed)
            return infeed.generate_enqueue_ops(per_host_sharded_inputs)

          infeed = tf.contrib.tpu.InfeedQueue(
              number_of_tuple_elements=len(per_host_sharded_inputs[0]))
          self.infeed_queue.append(infeed)
          return infeed.generate_enqueue_ops(
              per_host_sharded_inputs,
              tpu_ordinal_function=functools.partial(
                  runner_utils.tpu_ordinal_fn,
                  replicas_per_worker=self.params["replicas_per_worker"]))

        return enqueue_ops_fn

    with self.input_graph.as_default():
      self.enqueue_ops.append(
          runner_utils.wrap_computation_in_while_loop(
              get_enqueue_ops_fn(host_id),
              n=self.iterations_per_loop,
              host_name=self._get_host(host_id)))

  def initialize(self, model_fn, input_fn):
    """Build graph and do initialization for training."""
    tf.logging.info("TrainLowLevelRunner: initialize method")

    num_hosts = (
        self.num_shards * self.num_cores_per_replica //
        self.params["cores_per_worker"])
    for i in range(num_hosts):
      self.build_enqueue_ops(input_fn, self.params, num_hosts, i)

    def infeed_thread_fn():
      """Build and infeed session.run calls in a background thread."""
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
      features, labels = self.input_flattener.unflatten_features_and_labels(
          values)
      estimator_spec = model_fn(features, labels, tf.estimator.ModeKeys.TRAIN,
                                self.params)
      loss, train_op = estimator_spec.loss, estimator_spec.train_op
      self.scaffold_fn = estimator_spec.scaffold_fn
      with tf.control_dependencies([train_op]):
        return tf.identity(loss)

    @tpu_function.on_device_training_loop
    def train_loop():
      return tf.contrib.tpu.repeat(self.iterations_per_loop, tpu_train_step,
                                   [_INITIAL_LOSS])

    self.train_graph = tf.Graph()
    with self.train_graph.as_default():
      (self.train_compile_op, self.train_op) = tpu.split_compile_and_shard(
          train_loop,
          inputs=[],
          num_shards=self.num_shards,
          outputs_from_all_shards=False,
          device_assignment=self.device_assignment
      )
      if self.scaffold_fn:
        self.scaffold_fn()
      global_initializer = tf.global_variables_initializer()
      local_initializer = tf.local_variables_initializer()
      graph_io.write_graph(
          self.input_graph.as_graph_def(add_shapes=True), self.model_dir,
          "input_graph.pbtxt")
      graph_io.write_graph(
          self.train_graph.as_graph_def(add_shapes=True), self.model_dir,
          "graph.pbtxt")
      self.saver = tf.train.Saver()

    # Build tpu train model session and initialize graph
    self.train_sess = tf.Session(
        self.tpu_cluster_resolver.get_master(),
        graph=self.train_graph,
        config=self.session_config)

    self.train_sess.run(global_initializer)
    self.train_sess.run(local_initializer)
    # Compiles the train program.
    self.train_sess.run([self.train_compile_op])

    # Complete infeed graph generation and session.run calls
    self.infeed_thread = threading.Thread(target=infeed_thread_fn)
    self.infeed_thread.start()

  def write_summary(self, summary_writer, graph, loss, global_step,
                    elapsed_time, elapsed_steps, trained_examples):
    """Write a per-epoch summary of loss, epoch time, etc."""
    with graph.as_default():
      global_step_per_sec = elapsed_steps / elapsed_time
      examples_per_sec = trained_examples / elapsed_time
      if summary_writer is not None:
        loss_summary = tf.Summary(
            value=[tf.Summary.Value(tag="loss", simple_value=loss)])
        global_step_summary = tf.Summary(value=[
            tf.Summary.Value(
                tag="global_step/sec", simple_value=global_step_per_sec)
        ])
        example_summary = tf.Summary(value=[
            tf.Summary.Value(
                tag="examples/sec", simple_value=examples_per_sec)
        ])
        summary_writer.add_summary(loss_summary, global_step)
        summary_writer.add_summary(global_step_summary, global_step)
        summary_writer.add_summary(example_summary, global_step)
      tf.logging.info("loss = %g, step = %d (%.3f sec)", loss, global_step,
                      elapsed_time)
      tf.logging.info("global_step/sec: %g", global_step_per_sec)
      tf.logging.info("examples/sec: %g", examples_per_sec)

  def train(self):
    """Run the Train loop on the TPU device."""
    train_steps = self.params["total_steps"]
    num_examples_per_epoch = self.params["num_examples_per_epoch"]
    tf.logging.info("TrainLowLevelRunner: train for %d steps in total",
                    train_steps)
    if train_steps % self.iterations_per_loop != 0:
      tf.logging.warning(
          "train_steps %d is not divisible by iterations_per_loop %d",
          train_steps, self.iterations_per_loop)
      train_steps = self.iterations_per_loop * int(
          math.ceil(train_steps / self.iterations_per_loop))

    ckpt_saver = runner_utils.AsyncCheckpointSaver(_MAX_NUM_CHECKPOINT_THREADS,
                                                   self.saver, self.model_dir,
                                                   self.train_sess)
    cur_step = 0
    while cur_step < train_steps:
      start = time.time()
      tf.logging.info("TrainLowLevelRunner: start train step:%d", cur_step)
      self.queue.put(self.iterations_per_loop)
      cur_step += self.iterations_per_loop
      loss = self.train_sess.run(self.train_op)
      end = time.time()

      # checkpoint every epoch.
      ckpt_saver.checkpoint(cur_step)
      self.write_summary(
          summary_writer=self.summary_writer,
          graph=self.train_graph,
          loss=loss[0],
          global_step=cur_step,
          elapsed_time=end - start,
          elapsed_steps=self.iterations_per_loop,
          trained_examples=num_examples_per_epoch)
