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
"""Train_and_eval MaskRcnn with low level API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import threading
import time
import six

import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import device_assignment as tpu_device_assignment
from tensorflow.contrib.tpu.python.tpu import tpu
from tensorflow.contrib.tpu.python.tpu import tpu_feed
from tensorflow.python.framework import graph_io
import eval_multiprocess
import mask_rcnn_params
import runner_utils
from mlp_log import mlp_log


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


# Decorator function for tpu computation func that was passed to tpu.rewrite()
# if there are embedded train and eval loops in this func, trace tools will
# generate step markers for each iteration.
def on_device_train_and_eval_loops(func):
  # Value for this attribute is from xla.DebugOptions.StepMarkerLocation.
  setattr(func, "step_marker_location", "STEP_MARK_AT_SECOND_LEVEL_WHILE_LOOP")
  return func


class TrainEvalLowLevelRunner(object):
  """Run Train via direct session.run calls."""

  def __init__(self, tpu_cluster_resolver, train_params, eval_params,
               eval_steps, eval_metric, input_partition_dims=None,
               num_cores_per_replica=None, tpu_job_name=None):
    tf.logging.info("TrainLowLevelRunner: constructor")

    self.tpu_cluster_resolver = tpu_cluster_resolver
    self.eval_metric = eval_metric
    self.train_params = train_params
    self.eval_params = eval_params
    self.train_params["batch_size"] = (
        train_params["train_batch_size"] // train_params["num_shards"])
    self.eval_params["batch_size"] = (
        eval_params["eval_batch_size"] // eval_params["num_shards"])
    self.tpu_job_name = tpu_job_name

    self.model_dir = train_params["model_dir"]
    self.iterations_per_loop = train_params["iterations_per_loop"]
    self.eval_steps = eval_steps
    self.num_shards = self.train_params["num_shards"]
    self.input_flattener = runner_utils.InputsFlattener()
    self.eval_input_flattener = runner_utils.InputsFlattener()
    self.num_hosts = None
    self.train_eval_compile_op = None
    self.train_eval_op = None
    self.infeed_queue = []
    self.eval_infeed_queue = []
    self.outfeed_names = []
    self.outfeed_tensors = []
    self.enqueue_ops = []
    self.eval_enqueue_ops = []
    self.dequeue_ops = []
    self.dataset_initializer = []
    self.eval_dataset_initializer = []
    self.scaffold_fn = None
    # Having two separate sessions and graphs to make the initialization faster.
    self.input_sess = None
    self.train_eval_sess = None
    self.input_graph = tf.Graph()
    self.train_eval_graph = tf.Graph()
    self.session_config = tf.ConfigProto(
        allow_soft_placement=True, isolate_session_state=True,
        operation_timeout_in_ms=600 * 60 * 1000)  # 10 hours
    cluster_spec = self.tpu_cluster_resolver.cluster_spec()
    if cluster_spec:
      self.session_config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
    self.tpu_init = tf.contrib.tpu.initialize_system()
    self.tpu_shutdown = tf.contrib.tpu.shutdown_system()
    self.master = self.tpu_cluster_resolver.get_master()
    self.init_sess = tf.Session(self.master, config=self.session_config)
    self.device_topology = self.init_sess.run(self.tpu_init)
    self.input_partition_dims = input_partition_dims
    self.use_spatial_partition = input_partition_dims is not None
    self.num_cores_per_replica = num_cores_per_replica
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
      eval_input_partition_dims = [dict(self.input_partition_dims[0]), None]
      # don't need to partition the "is_padding" dimension
      if eval_params["eval_samples"] % eval_params["eval_batch_size"] != 0:
        eval_input_partition_dims[0][mask_rcnn_params.IS_PADDING] = None
      self.eval_input_dims_flattener = runner_utils.InputDimsFlattener(
          eval_input_partition_dims)
    else:
      self.device_assignment = None
      self.input_dims_flattener = None
      self.eval_input_dims_flattener = None
    # Summary writer writes out train metrics.
    self.summary_writer = tf.summary.FileWriter(self.model_dir)
    # Summary writer writes out eval metrics.
    eval_output_dir = os.path.join(self.model_dir, "eval")
    tf.gfile.MakeDirs(eval_output_dir)
    self.eval_summary_writer = tf.summary.FileWriter(eval_output_dir)
    self.infeed_thread = None
    self.total_epoch = self.train_params[
        "total_steps"] // self.iterations_per_loop

  def shutdown(self):
    """Shut down TrainLowLevelRunner."""
    tf.logging.info("TrainLowLevelRunner: shutdown")
    if self.infeed_thread:
      self.infeed_thread.join()
    if self.input_sess:
      self.input_sess.close()
    if self.train_eval_sess:
      self.train_eval_sess.close()
    self.summary_writer.close()
    self.eval_summary_writer.close()

  def _get_host(self, host_id):
    if self.master in ("", "local"):
      return "/replica:0/task:0"
    job_name = (
        self.tpu_job_name or self.tpu_cluster_resolver.get_job_name() or
        "tpu_worker")
    return "/job:%s/task:%d" % (job_name, host_id)

  def build_enqueue_ops(self, input_fn, params, num_hosts, host_id, iterations,
                        is_training=True):
    """Build enqueue ops."""
    tf.logging.info("TrainLowLevelRunner: build_enqueue_ops for %d, train=%g",
                    host_id, is_training)

    def get_enqueue_ops_fn(host_id):
      """Generate the enqueue ops graph function for training."""
      #  TODO(b/129084726): make dataset sharding also work for TPU Estimator.
      params["dataset_num_shards"] = num_hosts
      params["dataset_shard_id"] = host_id
      with tf.device(runner_utils.device_for_host(self._get_host(host_id))):
        dataset = input_fn(params)
        iterator = dataset.make_initializable_iterator()
        if is_training:
          self.dataset_initializer.append(iterator.initializer)
        else:
          self.eval_dataset_initializer.append(iterator.initializer)

        def enqueue_ops_fn():
          """Enqueue ops function for one host."""
          per_host_sharded_inputs = []
          control_deps = []
          for _ in range(self.train_params["replicas_per_worker"]):
            with tf.control_dependencies(control_deps):
              features, labels = iterator.get_next()
            if self.use_spatial_partition:
              self.input_dims_flattener.validate_and_flatten_input_dims(
                  features, labels)
            flattened_inputs = (
                self.input_flattener.flatten_features_and_labels(
                    features, labels))
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
                  replicas_per_worker=self.train_params["replicas_per_worker"]))

        return enqueue_ops_fn

    def get_eval_enqueue_ops_fn(host_id):
      """Generate the enqueue ops graph function for eval."""
      #  TODO(b/129084726): make dataset sharding also work for TPU Estimator.
      params["dataset_num_shards"] = num_hosts
      params["dataset_shard_id"] = host_id
      with tf.device(runner_utils.device_for_host(self._get_host(host_id))):
        dataset = input_fn(params)
        iterator = dataset.make_initializable_iterator()
        self.eval_dataset_initializer.append(iterator.initializer)

        def eval_enqueue_ops_fn():
          """Enqueue ops function for one host."""
          per_host_sharded_inputs = []
          control_deps = []
          for _ in range(self.train_params["replicas_per_worker"]):
            with tf.control_dependencies(control_deps):
              features = iterator.get_next()
            if self.use_spatial_partition:
              self.eval_input_dims_flattener.validate_and_flatten_input_dims(
                  features, None)
            flattened_inputs = (
                self.eval_input_flattener.flatten_features_and_labels(
                    features, None))
            control_deps.extend(flattened_inputs)
            per_host_sharded_inputs.append(flattened_inputs)

          if self.use_spatial_partition:
            flattened_input_dims = (
                self.eval_input_dims_flattener.flattened_input_dims)
            # pylint: disable=protected-access
            infeed = tpu_feed._PartitionedInfeedQueue(
                number_of_tuple_elements=len(per_host_sharded_inputs[0]),
                host_id=host_id,
                input_partition_dims=flattened_input_dims,
                device_assignment=self.device_assignment)
            self.eval_infeed_queue.append(infeed)
            return infeed.generate_enqueue_ops(per_host_sharded_inputs)

          infeed = tf.contrib.tpu.InfeedQueue(
              number_of_tuple_elements=len(per_host_sharded_inputs[0]))
          self.eval_infeed_queue.append(infeed)
          return infeed.generate_enqueue_ops(
              per_host_sharded_inputs,
              tpu_ordinal_function=functools.partial(
                  runner_utils.tpu_ordinal_fn,
                  replicas_per_worker=self.train_params["replicas_per_worker"]))

        return eval_enqueue_ops_fn

    with self.input_graph.as_default():
      enqueue_op = runner_utils.wrap_computation_in_while_loop(
          get_enqueue_ops_fn(host_id)
          if is_training else get_eval_enqueue_ops_fn(host_id),
          n=iterations,
          host_name=self._get_host(host_id))
      if is_training:
        self.enqueue_ops.append(enqueue_op)
      else:
        self.eval_enqueue_ops.append(enqueue_op)

  def initialize(self, model_fn, input_fn, eval_input_fn):
    """Build graph and do initialization for training."""
    tf.logging.info("TrainAndEvalLowLevelRunner: initialize method")

    self.num_hosts = (
        self.num_shards * self.num_cores_per_replica //
        self.train_params["cores_per_worker"])
    for i in range(self.num_hosts):
      self.build_enqueue_ops(input_fn, self.train_params, self.num_hosts, i,
                             self.iterations_per_loop, True)
      self.build_enqueue_ops(eval_input_fn, self.eval_params, self.num_hosts, i,
                             self.eval_steps, False)

    def infeed_thread_fn():
      """Build and infeed session.run calls in a background thread."""
      # Starts the clock.
      time.sleep(60)
      mlp_log.mlperf_print(key="init_stop", value=None)
      mlp_log.mlperf_print(key="run_start", value=None)
      mlp_log.mlperf_print(
          "block_start", None, metadata={"first_epoch_num": 0,
                                         "epoch_count": 1})
      for cur_epoch in range(self.total_epoch):
        tf.logging.info("Start to infeed train batches for epoch %d", cur_epoch)
        self.input_sess.run([self.enqueue_ops])
        tf.logging.info("Start to infeed eval batches for epoch %d", cur_epoch)
        self.input_sess.run([self.eval_enqueue_ops])
      tf.logging.info("infeed thread exited.")

    def tpu_train_step(loss):
      """Generate the TPU graph."""
      del loss
      values = self.infeed_queue[0].generate_dequeue_op(tpu_device=0)
      features, labels = self.input_flattener.unflatten_features_and_labels(
          values)
      estimator_spec = model_fn(features, labels, tf.estimator.ModeKeys.TRAIN,
                                self.train_params)
      loss, train_op = estimator_spec.loss, estimator_spec.train_op
      self.scaffold_fn = estimator_spec.scaffold_fn
      with tf.control_dependencies([train_op]):
        return tf.identity(loss)

    def train_loop():
      return tf.contrib.tpu.repeat(self.iterations_per_loop, tpu_train_step,
                                   [_INITIAL_LOSS])

    def tpu_eval_step():
      """Generate the TPU graph."""
      values = self.eval_infeed_queue[0].generate_dequeue_op(tpu_device=0)
      (features,
       _) = self.eval_input_flattener.unflatten_features_and_labels(values)
      estimator_spec = model_fn(features, None, tf.estimator.ModeKeys.PREDICT,
                                self.eval_params)
      for k, v in six.iteritems(estimator_spec.predictions):
        self.outfeed_names.append(k)
        self.outfeed_tensors.append(v)

      with tf.device(runner_utils.device_for_tpu_core(self._get_host(0))):
        outfeed_enqueue_ops = tf.contrib.tpu.outfeed_enqueue_tuple(
            self.outfeed_tensors)
      with tf.control_dependencies([outfeed_enqueue_ops]):
        return tf.no_op()

    def eval_loop():
      return tf.contrib.tpu.repeat(self.eval_steps, tpu_eval_step, [])

    def train_eval_step():
      with tf.control_dependencies(train_loop()):
        return eval_loop()

    @on_device_train_and_eval_loops
    def train_eval_loop():
      return tf.contrib.tpu.repeat(
          self.total_epoch if self.train_params["all_in_one_session"] else 1,
          train_eval_step, [])

    def create_dequeue_ops(host_id):
      """Create outfeed dequeue ops."""
      dequeue_ops = []
      tensor_dtypes = []
      tensor_shapes = []
      for v in self.outfeed_tensors:
        dequeue_ops.append([])
        tensor_dtypes.append(v.dtype)
        tensor_shapes.append(v.shape)
      for i in range(self.eval_params["replicas_per_worker"]):
        with tf.device(runner_utils.device_for_host(self._get_host(host_id))):
          if self.use_spatial_partition:
            replica_id = self.device_assignment.lookup_replicas(host_id, 0)[i]
            ordinal = self.device_assignment.tpu_ordinal(
                replica=replica_id, logical_core=0)
          else:
            ordinal = i
          outfeed_tensors = tf.contrib.tpu.outfeed_dequeue_tuple(
              dtypes=tensor_dtypes,
              shapes=tensor_shapes,
              device_ordinal=ordinal)
          for j, item in enumerate(outfeed_tensors):
            dequeue_ops[j].append(item)
      for j in range(len(outfeed_tensors)):
        dequeue_ops[j] = tf.concat(dequeue_ops[j], axis=0)
      return dequeue_ops

    with self.train_eval_graph.as_default():
      (self.train_eval_compile_op,
       self.train_eval_op) = tpu.split_compile_and_shard(
           train_eval_loop,
           inputs=[],
           num_shards=self.train_params["num_shards"],
           outputs_from_all_shards=False,
           device_assignment=self.device_assignment
       )
      for i in range(self.num_hosts):
        self.dequeue_ops.append({})
        tf.logging.info(
            "TrainAndEvalLowLevelRunner: get dequeue ops for host:%d", i)
        host_dequeue_ops = create_dequeue_ops(i)
        for j, dequeue_tenor in enumerate(host_dequeue_ops):
          self.dequeue_ops[i][self.outfeed_names[j]] = dequeue_tenor
      if self.scaffold_fn:
        self.scaffold_fn()
      global_initializer = tf.global_variables_initializer()
      local_initializer = tf.local_variables_initializer()
      graph_io.write_graph(
          self.train_eval_graph.as_graph_def(add_shapes=True), self.model_dir,
          "graph.pbtxt")
      self.saver = tf.train.Saver()

    # Build tpu train model session and initialize graph
    self.train_eval_sess = tf.Session(
        self.master,
        graph=self.train_eval_graph,
        config=self.session_config)

    self.train_eval_sess.run(global_initializer)
    self.train_eval_sess.run(local_initializer)
    # Compiles the train program.
    self.train_eval_sess.run([self.train_eval_compile_op])

    # Complete infeed graph generation and session.run calls
    self.input_sess = tf.Session(
        self.master,
        graph=self.input_graph,
        config=self.session_config)
    self.input_sess.run(self.dataset_initializer)
    self.input_sess.run(self.eval_dataset_initializer)
    self.infeed_thread = threading.Thread(target=infeed_thread_fn)
    self.infeed_thread.start()

  def write_summary(self, summary_writer, graph, global_step,
                    elapsed_time, elapsed_steps, trained_examples):
    """Write a per-epoch summary of loss, epoch time, etc."""
    with graph.as_default():
      global_step_per_sec = elapsed_steps / elapsed_time
      examples_per_sec = trained_examples / elapsed_time
      if summary_writer is not None:
        global_step_summary = tf.Summary(value=[
            tf.Summary.Value(
                tag="global_step/sec", simple_value=global_step_per_sec)
        ])
        example_summary = tf.Summary(value=[
            tf.Summary.Value(
                tag="examples/sec", simple_value=examples_per_sec)
        ])
        summary_writer.add_summary(global_step_summary, global_step)
        summary_writer.add_summary(example_summary, global_step)
      tf.logging.info("step = %d (%.3f sec)", global_step, elapsed_time)
      tf.logging.info("global_step/sec: %g", global_step_per_sec)
      tf.logging.info("examples/sec: %g", examples_per_sec)

  def write_eval_summary(self, summary_writer, eval_results, current_step):
    """Write out eval results for the checkpoint."""
    with tf.Graph().as_default():
      summaries = []
      for metric in eval_results:
        summaries.append(
            tf.Summary.Value(
                tag=metric, simple_value=eval_results[metric]))
      tf_summary = tf.Summary(value=list(summaries))
      summary_writer.add_summary(tf_summary, current_step)

  def get_predict_results(self, cur_epoch):
    """Run the predict loop on the TPU device."""
    for step in range(self.eval_steps):
      tf.logging.info(
          "TrainAndEvalLowLevelRunner: reading eval step %d results", step)
      predictions = {name: [] for name in self.outfeed_names}
      for outfeed_dict in self.train_eval_sess.run(self.dequeue_ops):
        for name, tensors in six.iteritems(outfeed_dict):
          predictions[name].extend(tensors)
      if step == self.eval_steps - 1:
        # all predictions is read from device, async eval post-process starts.
        # next train on device also starts.
        mlp_log.mlperf_print(
            "block_stop", None, metadata={"first_epoch_num": cur_epoch,
                                          "epoch_count": 1})
        mlp_log.mlperf_print(
            "eval_start", None, metadata={"epoch_num": cur_epoch})
        tf.logging.info("TrainAndEvalLowLevelRunner: start eval epoch %d.",
                        cur_epoch)
        mlp_log.mlperf_print(
            "block_start", None, metadata={"first_epoch_num": cur_epoch + 1,
                                           "epoch_count": 1})
      yield predictions

  def train_and_eval(self):
    """Performs distributed model eval and writes a summary to directory."""
    self.run_success = False
    self.continue_train = True

    # queues for predictions post-processing.
    def post_processing_thread_fn():
      """Run post-processing on CPU for predictions."""
      for cur_epoch in range(self.total_epoch):

        eval_begin = time.time()
        # Enables multi-processing to accelerate post-processing.
        eval_multiprocess.eval_multiprocessing(
            self.eval_steps, self.get_predict_results(cur_epoch),
            self.eval_metric, self.eval_params["eval_worker_count"])

        pred_end = time.time()
        tf.logging.info("prediction takes %d seconds.", pred_end - eval_begin)

        num_eval_samples, eval_results = self.eval_metric.evaluate()
        eval_end = time.time()
        tf.logging.info("COCO evaluates %d samples", num_eval_samples)
        if num_eval_samples != self.eval_params["eval_samples"]:
          tf.logging.info("COCO fails to evaluate all %d samples, exit!" %
                          self.eval_params["eval_samples"])
          self.run_success = False
          self.continue_train = False
          return
        tf.logging.info("one evaluation takes %d seconds",
                        eval_end - eval_begin)
        self.write_eval_summary(self.eval_summary_writer, eval_results,
                                cur_epoch * self.iterations_per_loop)
        tf.logging.info("AP: %s" % eval_results["AP"])
        tf.logging.info("mask_AP: %s" % eval_results["mask_AP"])
        # Eval epoch is 0-indexed (for MLPerf log parsing).
        mlp_log.mlperf_print(
            "eval_stop", None, metadata={"epoch_num": cur_epoch})
        # TODO(b/127959551): use both metrics once the bug is resolved.
        mlp_log.mlperf_print(
            "eval_accuracy", (float(eval_results["AP"]),
                              float(eval_results["mask_AP"])),
            metadata={"epoch_num": cur_epoch})

        if (eval_results["AP"] >= mask_rcnn_params.BOX_EVAL_TARGET and
            eval_results["mask_AP"] >= mask_rcnn_params.MASK_EVAL_TARGET):
          mlp_log.mlperf_print("run_stop", None, metadata={"status": "success"})
          self.run_success = True
          self.continue_train = False
          return

    # Run predict post processing thread on the background.
    post_processing_thread = threading.Thread(target=post_processing_thread_fn)
    post_processing_thread.start()
    if self.train_params["all_in_one_session"]:
      tf.logging.info("TrainAndEvalLowLevelRunner: start train_eval sessions")
      self.train_eval_sess.run(self.train_eval_op)
    else:
      if self.train_params["train_and_eval_save_checkpoint"]:
        ckpt_saver = runner_utils.AsyncCheckpointSaver(
            _MAX_NUM_CHECKPOINT_THREADS, self.saver, self.model_dir,
            self.train_eval_sess)
      cur_epoch = 0
      while cur_epoch < self.total_epoch and self.continue_train:
        tf.logging.info("TrainAndEvalLowLevelRunner: start train epoch: %d",
                        cur_epoch)
        start = time.time()
        self.train_eval_sess.run(self.train_eval_op)
        end = time.time()
        self.write_summary(
            summary_writer=self.summary_writer,
            graph=self.train_eval_graph,
            global_step=cur_epoch * self.iterations_per_loop,
            elapsed_time=end - start,
            elapsed_steps=self.iterations_per_loop,
            trained_examples=self.train_params["num_examples_per_epoch"])
        if self.train_params["train_and_eval_save_checkpoint"]:
          ckpt_saver.checkpoint(cur_epoch * self.iterations_per_loop)
        if self.run_success or not self.continue_train:
          break
        cur_epoch += 1

    post_processing_thread.join()
    if not self.run_success:
      mlp_log.mlperf_print("run_stop", None, metadata={"status": "abort"})
