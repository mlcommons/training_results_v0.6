"""Library for training. See t2t_trainer.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import re
import numpy as np

import tensorflow as tf

from mlp_log import mlp_log
from data_generators import text_encoder
from data_generators.problem import Problem
from utils import decoding
from utils import eval_low_level_runner
from utils import low_level_runner
from utils import registry
from utils import t2t_model
from utils import train_low_level_runner


def next_checkpoint(model_dir, timeout_mins=240):
  """Yields successive checkpoints from model_dir.

  Args:
    model_dir: The directory in which checkpoints are saved.
    timeout_mins: The maximum amount of time in minutes to wait
                  between checkpoints. Set this to -1 to wait indefinitely.
  Yields:
    last_ckpt: a new checkpoint path, or None if the timeout was reached.
  """
  last_ckpt = None
  timeout_secs = None
  if timeout_mins != -1:
    timeout_secs = timeout_mins * 60
  while True:
    last_ckpt = tf.contrib.training.wait_for_new_checkpoint(
        model_dir, last_ckpt, seconds_to_sleep=60, timeout=timeout_secs)

    if last_ckpt is None:
      tf.logging.info(
          "Eval timeout: no new checkpoints within %dm" % timeout_mins)
      break

    yield last_ckpt


def next_undecoded_checkpoint(model_dir, timeout_mins=240):
  """Yields successive checkpoints from model_dir."""
  last_ckpt = None
  last_step = 0
  while True:
    # Get the latest checkpoint.
    last_ckpt = tf.contrib.training.wait_for_new_checkpoint(
        model_dir, last_ckpt, seconds_to_sleep=60, timeout=60 * timeout_mins)
    # Get all the checkpoint from the model dir.
    ckpt_path = tf.train.get_checkpoint_state(model_dir)
    all_model_checkpoint_paths = ckpt_path.all_model_checkpoint_paths
    ckpt_step = np.inf
    next_ckpt = None
    # Find the next checkpoint to eval based on last_step.
    for ckpt in all_model_checkpoint_paths:
      step = int(os.path.basename(ckpt).split("-")[1])
      if step > last_step and step < ckpt_step:
        ckpt_step = step
        next_ckpt = ckpt

    # If all the checkpoints have been evaluated.
    if last_ckpt is None and next_ckpt is None:
      tf.logging.info(
          "Eval timeout: no new checkpoints within %dm" % timeout_mins)
      break

    if next_ckpt is not None:
      last_step = ckpt_step
      last_ckpt = next_ckpt

    yield last_ckpt


def create_hparams(hparams_set,
                   hparams_overrides_str="",
                   data_dir=None,
                   problem_name=None):
  """Create HParams with data_dir and problem hparams, if kwargs provided."""
  hparams = registry.hparams(hparams_set)
  if data_dir:
    hparams.add_hparam("data_dir", data_dir)
  if hparams_overrides_str:
    tf.logging.info("Overriding hparams in %s with %s", hparams_set,
                    hparams_overrides_str)
    hparams = hparams.parse(hparams_overrides_str)
  if problem_name:
    add_problem_hparams(hparams, problem_name)
  return hparams


def create_run_config(
    model_name,  # pylint: disable=unused-argument
    master="",
    model_dir=None,
    iterations_per_loop=1000,
    num_shards=8,
    save_checkpoints_steps=1000,
    keep_checkpoint_max=20,
    keep_checkpoint_every_n_hours=10000,
    schedule="continuous_train_and_eval",  # pylint: disable=unused-argument
    random_seed=None,
    use_tpu=False,
    log_step_count_steps=100,
    cloud_tpu_name="",
    tpu_config_extra_kwargs=None):
  """Create RunConfig, TPUConfig, and Parallelism object."""
  session_config = tf.ConfigProto(
      allow_soft_placement=True,
      graph_options=tf.GraphOptions())
  run_config_args = {
      "master": master,
      "evaluation_master": master,
      "model_dir": model_dir,
      "session_config": session_config,
      "save_summary_steps": 100,
      "save_checkpoints_steps": save_checkpoints_steps,
      "keep_checkpoint_max": keep_checkpoint_max,
      "keep_checkpoint_every_n_hours": keep_checkpoint_every_n_hours,
      "tf_random_seed": random_seed,
      "log_step_count_steps": log_step_count_steps
  }
  run_config_cls = tf.contrib.learn.RunConfig

  if use_tpu:
    # If using TPUEstimator, use TPU RunConfig, add TPUConfig, and add
    # additional args.
    tpu_config_kwargs = {
        "iterations_per_loop": iterations_per_loop,
        "num_shards": num_shards,
        "per_host_input_for_training": True,
    }
    if tpu_config_extra_kwargs is not None:
      tpu_config_kwargs.update(tpu_config_extra_kwargs)
    run_config_cls = tf.contrib.tpu.RunConfig
    tpu_config = tf.contrib.tpu.TPUConfig(**tpu_config_kwargs)
    run_config_args["tpu_config"] = tpu_config
    if not master and cloud_tpu_name:
      # Update run_config to use cluster instead of master/evaluation_master
      # as we need the cluster spec to use Cloud Pods
      tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
          cloud_tpu_name)
      run_config_args["cluster"] = tpu_cluster_resolver
      del run_config_args["master"]
      del run_config_args["evaluation_master"]

  config = run_config_cls(**run_config_args)
  config.use_tpu = use_tpu

  return config


def create_estimator(model_name,
                     hparams,
                     run_config,
                     schedule="train_and_evaluate",
                     decode_hparams=None,
                     use_tpu=False):
  """Create a T2T Estimator."""
  model_fn = t2t_model.T2TModel.make_estimator_model_fn(
      model_name, hparams, decode_hparams=decode_hparams, use_tpu=use_tpu)

  if use_tpu:
    problem = hparams.problem
    batch_size = (
        problem.tpu_batch_size_per_shard(hparams) *
        run_config.tpu_config.num_shards)
    predict_batch_size = batch_size
    if decode_hparams and decode_hparams.batch_size:
      predict_batch_size = decode_hparams.batch_size
    if decode_hparams and run_config.tpu_config:
      decode_hparams.add_hparam("iterations_per_loop",
                                run_config.tpu_config.iterations_per_loop)
    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        model_dir=run_config.model_dir,
        config=run_config,
        use_tpu=use_tpu,
        train_batch_size=batch_size,
        eval_batch_size=batch_size if "eval" in schedule else None,
        predict_batch_size=predict_batch_size)
  else:
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=run_config.model_dir,
        config=run_config,
    )
  return estimator


class T2TExperiment(object):
  """Custom Experiment class for running distributed experiments."""

  def __init__(self,
               estimator,
               hparams,
               train_spec,
               eval_spec,
               decode_hparams=None,
               trunner=None,
               erunner=None,
               runner=None):
    self._train_spec = train_spec
    self._eval_spec = eval_spec
    self._hparams = hparams
    self._decode_hparams = decode_hparams
    self._estimator = estimator
    self._trunner = trunner
    self._erunner = erunner
    self._runner = runner

  @property
  def estimator(self):
    return self._estimator

  @property
  def train_steps(self):
    return self._train_spec.max_steps

  @property
  def eval_steps(self):
    return self._eval_spec.steps

  def continuous_train_and_eval(self, continuous_eval_predicate_fn=None):
    del continuous_eval_predicate_fn
    tf.estimator.train_and_evaluate(self._estimator, self._train_spec,
                                    self._eval_spec)
    return self.evaluate()

  def train(self, max_steps=None):
    """Train for max_steps."""
    mlp_log.mlperf_print(key="init_stop", value=None)
    mlp_log.mlperf_print(key="run_start", value=None)
    mlp_log.mlperf_print(
        "block_start", None, metadata={
            "first_epoch_num": 1,
            "epoch_count": 1
        })

    if self._hparams.train_with_low_level_api:
      self._trunner.train(self._hparams.train_steps, self._hparams.batch_size)
      self._trunner.shutdown()
    else:
      self._estimator.train(
          self._train_spec.input_fn,
          hooks=self._train_spec.hooks,
          max_steps=max_steps or self._train_spec.max_steps)

  def train_and_decode(self):
    """Does decode after training every eval_freq_in_steps."""
    eval_steps = self._hparams.eval_freq_in_steps
    if self._hparams.train_and_decode_with_low_level_api:
      self._runner.train_and_eval(
          self._train_spec.max_steps, self._hparams.batch_size)
      for i in range(0, self._train_spec.max_steps, eval_steps):
        if self._hparams.mlperf_mode:
          self._decode_hparams.mlperf_decode_step = i + eval_steps
        output_dir = os.path.join(self._estimator.model_dir, "decode")
        tf.gfile.MakeDirs(output_dir)
        output_dirs = [output_dir]
        result = list(self._runner.dequeue(self._decode_hparams))

        mlp_log.mlperf_print(
            "eval_start", None, metadata={"epoch_num": (i // eval_steps + 1)})
        predictions = []
        inputs_vocab = self._hparams.problem_hparams.vocabulary["inputs"]
        targets_vocab = self._hparams.problem_hparams.vocabulary["targets"]
        for prediction in result:
          inputs = prediction.get("inputs")
          targets = prediction.get("targets")
          outputs = prediction.get("outputs")
          if not re.match(
              "^({})+$".format(text_encoder.PAD),
              inputs_vocab.decode(decoding.save_until_eos(inputs))):
            predictions.append(
                (targets_vocab.decode(decoding.save_until_eos(outputs)),
                 targets_vocab.decode(decoding.save_until_eos(targets))))
        decoding.run_postdecode_hooks(decoding.DecodeHookArgs(
            estimator=self._estimator,
            problem=self._hparams.problem,
            output_dirs=output_dirs,
            hparams=self._hparams,
            decode_hparams=self._decode_hparams,
            predictions=predictions
        ), tf.estimator.ModeKeys.EVAL)

        mlp_log.mlperf_print(
            "block_stop",
            None,
            metadata={
                "first_epoch_num": (i // eval_steps + 1),
                "epoch_count": 1
            })
        if self._hparams.mlperf_mode and self._decode_hparams.mlperf_success:
          break

      self._runner.shutdown()
    else:
      mlp_log.mlperf_print(key="init_stop", value=None)
      mlp_log.mlperf_print(key="run_start", value=None)
      packed_dataset = "_packed" in self._hparams.problem.name
      for i in range(0, self._train_spec.max_steps, eval_steps):
        mlp_log.mlperf_print(
            "block_start",
            None,
            metadata={
                "first_epoch_num": (i // eval_steps + 1),
                "epoch_count": 1
            })
        if packed_dataset and i > 0:
          problem = registry.problem(self._hparams.problem.name + "_packed")
          p_hparams = problem.get_hparams(self._hparams)
          self._hparams.problem = problem
          self._hparams.problem_hparams = p_hparams
        self._estimator.train(
            self._train_spec.input_fn,
            steps=eval_steps,
            hooks=self._train_spec.hooks)
        if packed_dataset:
          problem = registry.problem(
              self._hparams.problem.name.replace("_packed", ""))
          p_hparams = problem.get_hparams(self._hparams)
          self._hparams.problem = problem
          self._hparams.problem_hparams = p_hparams
        if self._hparams.mlperf_mode:
          self._decode_hparams.mlperf_decode_step = i + eval_steps
        predictions = self.decode(dataset_split=tf.estimator.ModeKeys.EVAL)
        mlp_log.mlperf_print(
            "block_stop",
            None,
            metadata={
                "first_epoch_num": (i // eval_steps + 1),
                "epoch_count": 1
            })
        if self._hparams.mlperf_mode and self._decode_hparams.mlperf_success:
          break

    if self._hparams.mlperf_mode and not self._decode_hparams.mlperf_success:
      mlp_log.mlperf_print("run_stop", None, metadata={"status": "abort"})
    return predictions, self._train_spec.max_steps

  def evaluate(self):
    return self._estimator.evaluate(
        self._eval_spec.input_fn,
        steps=self._eval_spec.steps,
        hooks=self._eval_spec.hooks)

  def evaluate_on_train_data(self):
    self._estimator.evaluate(
        self._train_spec.input_fn,
        steps=self._eval_spec.steps,
        hooks=self._eval_spec.hooks,
        name="eval_train")

  def continuous_eval(self):
    """Evaluate until checkpoints stop being produced."""
    for _ in next_checkpoint(self._hparams.model_dir,
                             self._hparams.eval_timeout_mins):
      self.evaluate()

  def continuous_eval_on_train_data(self):
    """Evaluate on train data until checkpoints stop being produced."""
    for _ in next_checkpoint(self._hparams.model_dir,
                             self._hparams.eval_timeout_mins):
      self.evaluate_on_train_data()

  def test(self):
    """Perform 1 step of train and 2 step of eval."""
    self._estimator.train(
        self._train_spec.input_fn, hooks=self._train_spec.hooks, max_steps=1)

    self._estimator.evaluate(
        self._eval_spec.input_fn, steps=1, hooks=self._eval_spec.hooks)

  def decode(self,
             dataset_split=None,
             decode_from_file=False,
             checkpoint_path=None):
    """Decodes from dataset or file."""
    if decode_from_file:
      decoding.decode_from_file(self._estimator,
                                self._decode_hparams.decode_from_file,
                                self._hparams,
                                self._decode_hparams,
                                self._decode_hparams.decode_to_file)
    else:
      decoding.decode_from_dataset(
          self._estimator,
          self._hparams.problem.name,
          self._hparams,
          self._decode_hparams,
          dataset_split=dataset_split,
          checkpoint_path=checkpoint_path,
          erunner=self._erunner,
          runner=self._runner)

  def continuous_decode_on_eval_data(self):
    """Decode from dataset on new checkpoint."""
    if self._hparams.mlperf_mode:
      ckpt_generator = next_undecoded_checkpoint(self._hparams.model_dir)
    else:
      ckpt_generator = next_checkpoint(self._hparams.model_dir)

    for ckpt in ckpt_generator:
      current_step = int(os.path.basename(ckpt).split("-")[1])
      tf.logging.info("Decoding step %d" % current_step)
      # Skip checkpoint 0.
      if current_step == 0:
        continue
      # Decode the latest checkpoint by default.
      checkpoint_path = None
      if self._hparams.mlperf_mode:
        self._decode_hparams.mlperf_decode_step = current_step
        checkpoint_path = ckpt

      mlp_log.mlperf_print(
          "eval_start",
          None,
          metadata={
              "epoch_num": max(
                  current_step // self._decode_hparams.iterations_per_loop, 1)
          })
      self.decode(
          dataset_split=tf.estimator.ModeKeys.EVAL,
          checkpoint_path=checkpoint_path)
      if self._hparams.mlperf_mode and self._decode_hparams.mlperf_success:
        mlp_log.mlperf_print("run_stop", None, metadata={"status": "success"})
        break

    if self._hparams.mlperf_mode and not self._decode_hparams.mlperf_success:
      mlp_log.mlperf_print("run_stop", None, metadata={"status": "abort"})

  def continuous_decode_from_file(self):
    """Decode from file on new checkpoint."""
    for _ in next_checkpoint(self._hparams.model_dir):
      self.decode(decode_from_file=True)


def create_experiment(run_config,
                      hparams,
                      model_name,
                      problem_name,
                      data_dir,
                      train_steps,
                      eval_steps,
                      min_eval_frequency=2000,
                      schedule="train_and_evaluate",
                      decode_hparams=None,
                      eval_timeout_mins=240,
                      use_tpu=False,
                      train_with_low_level_api=False,
                      decode_with_low_level_api=False,
                      train_and_decode_with_low_level_api=False,
                      tpu_num_hosts=1,
                      iterations_per_loop=1000,
                      decode_from_file=None,
                      decode_to_file=None,
                      decode_reference=None):
  """Create Experiment."""
  # HParams
  hparams.add_hparam("model_dir", run_config.model_dir)
  hparams.add_hparam("data_dir", data_dir)
  hparams.add_hparam("train_steps", train_steps)
  hparams.add_hparam("eval_steps", eval_steps)
  hparams.add_hparam("schedule", schedule)
  hparams.add_hparam("eval_freq_in_steps", min_eval_frequency)
  hparams.add_hparam("eval_timeout_mins", eval_timeout_mins)
  hparams.add_hparam("train_with_low_level_api", train_with_low_level_api)
  hparams.add_hparam("decode_with_low_level_api", decode_with_low_level_api)
  hparams.add_hparam("train_and_decode_with_low_level_api",
                     train_and_decode_with_low_level_api)
  if decode_hparams is not None:
    decode_hparams.add_hparam("decode_from_file", decode_from_file)
    decode_hparams.add_hparam("decode_to_file", decode_to_file)
    decode_hparams.add_hparam("decode_reference", decode_reference)
  add_problem_hparams(hparams, problem_name)

  # Input fns from Problem
  problem = hparams.problem
  train_input_fn = problem.make_estimator_input_fn(tf.estimator.ModeKeys.TRAIN,
                                                   hparams)
  eval_input_fn = problem.make_estimator_input_fn(tf.estimator.ModeKeys.EVAL,
                                                  hparams)

  if train_with_low_level_api:
    params = {}
    params["batch_size"] = problem.tpu_batch_size_per_shard(hparams)
    params["tpu_num_hosts"] = tpu_num_hosts
    mlp_log.mlperf_print(
        key="global_batch_size",
        value=params["batch_size"] * run_config.tpu_config.num_shards)
    trunner = train_low_level_runner.TrainLowLevelRunner(
        iterations=iterations_per_loop)
    model_fn = t2t_model.T2TModel.make_estimator_model_fn(
        model_name, hparams, decode_hparams=decode_hparams, use_tpu=use_tpu)
    trunner.initialize(train_input_fn, model_fn, params, hparams, run_config)

  if decode_with_low_level_api:
    if decode_hparams.batch_size:
      hparams.batch_size = decode_hparams.batch_size
      hparams.use_fixed_batch_size = True
    dataset_kwargs = {
        "shard": decode_hparams.shard_id if decode_hparams.shards > 1 else None,
        "dataset_split": tf.estimator.ModeKeys.EVAL,
        "max_records": decode_hparams.num_samples
    }
    infer_input_fn = problem.make_estimator_input_fn(
        tf.estimator.ModeKeys.PREDICT, hparams, dataset_kwargs=dataset_kwargs)

    params = {}
    # Currently, the decoding part runs on a donut, will change this for
    # distibuted eval.
    params["batch_size"] = int(decode_hparams.batch_size * tpu_num_hosts /
                               run_config.tpu_config.num_shards)
    erunner = eval_low_level_runner.EvalLowLevelRunner(
        eval_steps=int(
            math.ceil(decode_hparams.num_samples / decode_hparams.batch_size)))
    model_fn = t2t_model.T2TModel.make_estimator_model_fn(
        model_name, hparams, decode_hparams=decode_hparams, use_tpu=use_tpu)
    erunner.initialize(infer_input_fn, params, run_config)
    erunner.build_model(model_fn, params, run_config)

  if train_and_decode_with_low_level_api:
    mlp_log.mlperf_print(key="max_sequence_length", value=hparams.max_length)
    fake_train_input_fn = problem.make_estimator_input_fn(
        tf.estimator.ModeKeys.TRAIN, hparams, fake_data=True)
    params = {}
    params["batch_size"] = problem.tpu_batch_size_per_shard(hparams)
    params["tpu_num_hosts"] = tpu_num_hosts
    mlp_log.mlperf_print(
        key="global_batch_size",
        value=params["batch_size"] * run_config.tpu_config.num_shards)
    runner = low_level_runner.LowLevelRunner(
        iterations=iterations_per_loop,
        eval_steps=int(
            math.ceil(decode_hparams.num_samples / decode_hparams.batch_size)))
    model_fn = t2t_model.T2TModel.make_estimator_model_fn(
        model_name, hparams, decode_hparams=decode_hparams, use_tpu=use_tpu)

    # Changed the problem to unpacked one for decoding.
    if "_packed" in hparams.problem.name:
      problem = registry.problem(hparams.problem.name.replace("_packed", ""))
      p_hparams = problem.get_hparams(hparams)
      hparams.problem = problem
      hparams.problem_hparams = p_hparams

    # Hard-coded based on the current wmt14 en-de eval dataset.
    hparams.max_length = 97

    if decode_hparams.batch_size:
      hparams.batch_size = decode_hparams.batch_size
      hparams.use_fixed_batch_size = True
    dataset_kwargs = {
        "shard": decode_hparams.shard_id if decode_hparams.shards > 1 else None,
        "dataset_split": tf.estimator.ModeKeys.EVAL,
        "max_records": decode_hparams.num_samples
    }
    fake_infer_input_fn = problem.make_estimator_input_fn(
        tf.estimator.ModeKeys.PREDICT, hparams, fake_data=True,
        dataset_kwargs=dataset_kwargs)
    infer_input_fn = problem.make_estimator_input_fn(
        tf.estimator.ModeKeys.PREDICT, hparams, dataset_kwargs=dataset_kwargs)
    infer_model_fn = t2t_model.T2TModel.make_estimator_model_fn(
        model_name, hparams, decode_hparams=decode_hparams, use_tpu=use_tpu)
    runner.initialize(
        fake_train_input_fn,
        fake_infer_input_fn,
        train_input_fn,
        infer_input_fn,
        model_fn,
        infer_model_fn,
        params,
        hparams,
        run_config)

  # Estimator
  estimator = create_estimator(
      model_name,
      hparams,
      run_config,
      schedule=schedule,
      decode_hparams=decode_hparams,
      use_tpu=use_tpu)

  # Eval on TPU Pods is not supported yet
  if use_tpu and run_config.tpu_config.num_shards > 8 and "eval" in schedule:
    raise ValueError("Eval is not currently supported on a TPU Pod")

  train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=train_steps)
  eval_spec = tf.estimator.EvalSpec(
      eval_input_fn,
      steps=eval_steps,
      start_delay_secs=0 if hparams.schedule == "evaluate" else 120,
      exporters=None)

  return T2TExperiment(estimator, hparams, train_spec, eval_spec,
                       decode_hparams,
                       trunner if train_with_low_level_api else None,
                       erunner if decode_with_low_level_api else None,
                       runner if train_and_decode_with_low_level_api else None)


def create_experiment_fn(*args, **kwargs):
  """Wrapper for canonical experiment_fn. See create_experiment."""

  def experiment_fn(run_config, hparams):
    return create_experiment(run_config, hparams, *args, **kwargs)

  return experiment_fn


def add_problem_hparams(hparams, problem_name_or_instance):
  """Add problem hparams for the problems."""
  if isinstance(problem_name_or_instance, Problem):
    problem = problem_name_or_instance
  else:
    problem = registry.problem(problem_name_or_instance)
  p_hparams = problem.get_hparams(hparams)
  hparams.problem = problem
  hparams.problem_hparams = p_hparams


def set_random_seed(seed):
  tf.set_random_seed(seed)
  random.seed(seed)
  np.random.seed(seed)


def restore_checkpoint(ckpt_dir, saver, sess, must_restore=False):
  """Restore from a checkpoint."""
  ckpt = tf.train.get_checkpoint_state(ckpt_dir)
  if must_restore and not ckpt:
    raise ValueError("No checkpoint found in %s" % ckpt_dir)
  if not ckpt:
    return 0

  path = ckpt.model_checkpoint_path
  tf.logging.info("Restoring checkpoint %s", path)
  saver.restore(sess, path)
  step = int(path.split("-")[-1])
  return step
