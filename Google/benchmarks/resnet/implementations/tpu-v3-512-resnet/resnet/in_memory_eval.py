from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import six

import numpy as np

from tensorflow.contrib.tpu.python.tpu import error_handling
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import monitored_session
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import training
from tensorflow.python.training import training_util
from tensorflow.python.estimator import estimator as estimator_lib
from tensorflow.python.estimator import model_fn as model_fn_lib
from mlp_log import mlp_log

_EVAL_METRIC = 'top_1_accuracy'


def _initialize_in_memory_eval(estimator):
  """Setup TPUEstimator for in-memory evaluation."""

  # estimator.evaluate calls _create_global_step unconditionally, override this.
  def _global_step(_):
    with variable_scope.variable_scope('', use_resource=True):
      return training_util.get_or_create_global_step()

  estimator._create_global_step = _global_step  #  pylint: disable=protected-access
  estimator._rendezvous[  # pylint: disable=protected-access
      model_fn_lib.ModeKeys.EVAL] = error_handling.ErrorRendezvous(3)
  estimator._rendezvous[  # pylint: disable=protected-access
      model_fn_lib.ModeKeys.PREDICT] = error_handling.ErrorRendezvous(3)


# pylint: disable=protected-access
class TPUInMemoryEvalHook(training.SessionRunHook):
  """Hook to run evaluation in training without a checkpoint.

  Example:

  ```python
  def train_input_fn():
    ...
    return train_dataset

  def eval_input_fn():
    ...
    return eval_dataset

  estimator = tf.estimator.DNNClassifier(...)

  evaluator = tf.contrib.estimator.InMemoryEvalHook(
      estimator, eval_input_fn)
  estimator.train(train_input_fn, hooks=[evaluator])
  ```

  Current limitations of this approach are:

  * It doesn't support multi-node distributed mode.
  * It doesn't support saveable objects other than variables (such as boosted
    tree support)
  * It doesn't support custom saver logic (such as ExponentialMovingAverage
    support)

  """

  def __init__(self,
               estimator,
               input_fn,
               steps_per_epoch,
               stop_threshold=0.749,
               steps=None,
               hooks=None,
               name=None,
               every_n_iter=100,
               eval_every_epoch_from=61):
    """Initializes a `InMemoryEvalHook`.

    Args:
      estimator: A `tf.estimator.Estimator` instance to call evaluate.
      input_fn:  Equivalent to the `input_fn` arg to `estimator.evaluate`. A
        function that constructs the input data for evaluation. See [Createing
        input functions](
        https://tensorflow.org/guide/premade_estimators#create_input_functions)
          for more information. The function should construct and return one of
        the following:
          * A 'tf.data.Dataset' object: Outputs of `Dataset` object must be a
            tuple (features, labels) with same constraints as below.
          * A tuple (features, labels): Where `features` is a `Tensor` or a
            dictionary of string feature name to `Tensor` and `labels` is a
            `Tensor` or a dictionary of string label name to `Tensor`. Both
            `features` and `labels` are consumed by `model_fn`. They should
            satisfy the expectation of `model_fn` from inputs.
      steps_per_epoch: steps_per_epoch for training.
      stop_threshold: stop threshold for top 1 accuracy.
      steps: Equivalent to the `steps` arg to `estimator.evaluate`.  Number of
        steps for which to evaluate model. If `None`, evaluates until `input_fn`
        raises an end-of-input exception.
      hooks: Equivalent to the `hooks` arg to `estimator.evaluate`. List of
        `SessionRunHook` subclass instances. Used for callbacks inside the
        evaluation call.
      name:  Equivalent to the `name` arg to `estimator.evaluate`. Name of the
        evaluation if user needs to run multiple evaluations on different data
        sets, such as on training data vs test data. Metrics for different
        evaluations are saved in separate folders, and appear separately in
        tensorboard.
      every_n_iter: `int`, runs the evaluator once every N training iteration.
      eval_every_epoch_from: `int`, eval every epoch after this epoch.

    Raises:
      ValueError: if `every_n_iter` is non-positive or it's not a single machine
        training
    """
    if every_n_iter is None or every_n_iter <= 0:
      raise ValueError('invalid every_n_iter=%s.' % every_n_iter)
    if (estimator.config.num_ps_replicas > 0 or
        estimator.config.num_worker_replicas > 1):
      raise ValueError(
          'InMemoryEval supports only single machine (aka Local) setting.')
    self._estimator = estimator
    self._input_fn = input_fn
    self._steps = steps
    self._name = name
    self._every_n_iter = every_n_iter
    self._eval_dir = os.path.join(self._estimator.model_dir,
                                  'eval' if not name else 'eval_' + name)

    self._graph = None
    self._hooks = estimator_lib._check_hooks_type(hooks)
    self._hooks.extend(self._estimator._convert_eval_steps_to_hooks(steps))
    self._timer = training.SecondOrStepTimer(every_steps=every_n_iter)
    self._steps_per_epoch = steps_per_epoch
    self._stop_threshold = stop_threshold
    self._run_success = False
    self._eval_every_epoch_from = eval_every_epoch_from

    _initialize_in_memory_eval(estimator)

  def begin(self):
    """Build eval graph and restoring op."""
    self._timer.reset()
    self._graph = ops.Graph()
    self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
    with self._graph.as_default():
      (self._scaffold, self._update_op, self._eval_dict,
       self._all_hooks) = self._estimator._evaluate_build_graph(
           self._input_fn, self._hooks, checkpoint_path=None)

      for h in self._all_hooks:
        if isinstance(h, tpu_estimator.TPUInfeedOutfeedSessionHook):
          h._should_initialize_tpu = False  # pylint: disable=protected-access

      if self._scaffold.saver is not None:
        raise ValueError('InMemoryEval does not support custom saver')
      if self._scaffold.init_fn is not None:
        raise ValueError('InMemoryEval does not support custom init_fn')

      self._var_name_to_eval_var = {
          v.name: v for v in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      }
      self._var_name_to_placeholder = {
          v.name: array_ops.placeholder(v.dtype)
          for v in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      }

  def after_create_session(self, session, coord):  # pylint: disable=unused-argument
    """Does first run which shows the eval metrics before training."""
    if ops.get_collection(ops.GraphKeys.SAVEABLE_OBJECTS):
      raise ValueError(
          'InMemoryEval does not support saveables other than global '
          'variables.')
    logging.info('Eval: Building var map')
    self._var_name_to_train_var = {
        v.name: v for v in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    }
    logging.info('Eval: Building transfer set')
    var_names_to_transfer = set(self._var_name_to_placeholder.keys()) & set(
        self._var_name_to_train_var.keys())
    logging.info('Eval: Building filtering evaluation vars')
    # Filter training var names that are not exist in evaluation
    self._var_name_to_train_var = {
        v_name: self._var_name_to_train_var[v_name]
        for v_name in var_names_to_transfer
    }
    logging.info('Eval: Building filtering training vars')
    # Filter eval var names that are not exist in training
    self._var_name_to_eval_var = {
        v_name: self._var_name_to_eval_var[v_name]
        for v_name in var_names_to_transfer
    }

    logging.info('Eval: Building feed op')
    with self._graph.as_default():
      self._var_feed_op = control_flow_ops.group([
          state_ops.assign(self._var_name_to_eval_var[v_name],
                           self._var_name_to_placeholder[v_name])
          for v_name in var_names_to_transfer
      ])
    logging.info('Eval: Done building.')

  def _evaluate(self, session, step):
    var_name_to_value = session.run(self._var_name_to_train_var)
    logging.info('Building placeholders.')
    placeholder_to_value = {
        self._var_name_to_placeholder[v_name]: var_name_to_value[v_name]
        for v_name in var_name_to_value
    }

    def feed_variables(scaffold, session):
      del scaffold
      session.run(self._var_feed_op, feed_dict=placeholder_to_value)

    logging.info('Building scaffold.')
    scaffold = training.Scaffold(
        init_fn=feed_variables, copy_from_scaffold=self._scaffold)

    with self._graph.as_default():
      eval_results = self._estimator._evaluate_run(
          checkpoint_path=None,
          scaffold=scaffold,
          update_op=self._update_op,
          eval_dict=self._eval_dict,
          all_hooks=self._all_hooks,
          output_dir=self._eval_dir)
      logging.info('Eval done.')

    self._timer.update_last_triggered_step(step)
    return eval_results

  def _get_step(self):
    ckpt = checkpoint_management.latest_checkpoint(self._estimator.model_dir)
    if ckpt:
      return int(os.path.basename(ckpt).split('-')[1])
    else:
      return 0

  def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
    """Runs evaluator."""
    step = np.asscalar(run_context.session.run(self._global_step_tensor))

    if self._timer.should_trigger_for_step(step):
      logging.info('Starting eval.')
      eval_results = self._evaluate(run_context.session, step)
      mlp_log.mlperf_print(
          'eval_accuracy',
          float(eval_results[_EVAL_METRIC]),
          metadata={'epoch_num': max(step // self._steps_per_epoch - 1, 0)})

      # The ImageNet eval size is hard coded.
      if eval_results[_EVAL_METRIC] >= self._stop_threshold:
        self._run_success = True
        mlp_log.mlperf_print('run_stop', None, metadata={'status': 'success'})
        mlp_log.mlperf_print('run_final', None)
        run_context.request_stop()

    if step // self._steps_per_epoch == self._eval_every_epoch_from:
      self._timer = training.SecondOrStepTimer(
          every_steps=self._steps_per_epoch)
      self._timer.reset()

  def end(self, session):  # pylint: disable=unused-argument
    """Runs evaluator for final model."""
    # Only runs eval at the end if highest accuracy so far
    # is less than self._stop_threshold.
    if not self._run_success:
      step = np.asscalar(session.run(self._global_step_tensor))
      logging.info('Starting eval.')
      eval_results = self._evaluate(session, step)
      mlp_log.mlperf_print(
          'eval_accuracy',
          float(eval_results[_EVAL_METRIC]),
          metadata={'epoch_num': max(step // self._steps_per_epoch - 1, 0)})
      if eval_results[_EVAL_METRIC] >= self._stop_threshold:
        mlp_log.mlperf_print('run_stop', None, metadata={'status': 'success'})
      else:
        mlp_log.mlperf_print('run_stop', None, metadata={'status': 'abort'})

      mlp_log.mlperf_print('run_final', None)


class TPUInMemoryPredictHook(training.SessionRunHook):
  """Hook to run predictions + postprocessing in-memory

  ```
  evaluator = TPUInMemoryPredictEvalHook(
      estimator, predict_input_fn, predict_processing_fn)

  ```

  `predict_processing_fn` should take one argument: the output of each
  prediction.
  """

  def __init__(self,
               estimator,
               input_fn,
               handler,
               steps=None,
               hooks=None,
               name=None,
               every_n_iter=100):
    """Initializes a `InMemoryEvalHook`.

    Args:
      estimator: A `tf.estimator.Estimator` instance to call evaluate.
      input_fn:  Equivalent to the `input_fn` arg to `estimator.evaluate`. A
        function that constructs the input data for evaluation. See [Createing
        input functions](
        https://tensorflow.org/guide/premade_estimators#create_input_functions)
          for more information. The function should construct and return one of
        the following:
          * A 'tf.data.Dataset' object: Outputs of `Dataset` object must be a
            tuple (features, labels) with same constraints as below.
          * A tuple (features, labels): Where `features` is a `Tensor` or a
            dictionary of string feature name to `Tensor` and `labels` is a
            `Tensor` or a dictionary of string label name to `Tensor`. Both
            `features` and `labels` are consumed by `model_fn`. They should
            satisfy the expectation of `model_fn` from inputs.
      hooks: Equivalent to the `hooks` arg to `estimator.evaluate`. List of
        `SessionRunHook` subclass instances. Used for callbacks inside the
        evaluation call.
      name:  Equivalent to the `name` arg to `estimator.evaluate`. Name of the
        evaluation if user needs to run multiple evaluations on different data
        sets, such as on training data vs test data. Metrics for different
        evaluations are saved in separate folders, and appear separately in
        tensorboard.
      every_n_iter: `int`, runs the evaluator once every N training iteration.

    Raises:
      ValueError: if `every_n_iter` is non-positive or it's not a single machine
        training
    """
    if every_n_iter is None or every_n_iter <= 0:
      raise ValueError('invalid every_n_iter=%s.' % every_n_iter)
    if (estimator.config.num_ps_replicas > 0 or
        estimator.config.num_worker_replicas > 1):
      raise ValueError(
          'InMemoryEval supports only single machine (aka Local) setting.')
    self._estimator = estimator
    self._input_fn = input_fn
    self._handler = handler
    self._name = name
    self._every_n_iter = every_n_iter
    self._eval_dir = os.path.join(self._estimator.model_dir,
                                  'eval' if not name else 'eval_' + name)

    self._graph = None
    self._hooks = estimator_lib._check_hooks_type(hooks)
    self._timer = training.SecondOrStepTimer(every_steps=every_n_iter)

    self._var_name_to_eval_var = None
    self._var_name_to_placeholder = None
    self._var_name_to_train_var = None
    self._var_feed_op = None

    _initialize_in_memory_eval(estimator)

    self._predictions = None
    self._global_step_tensor = None

  def begin(self):
    """Build eval graph and restoring op."""
    self._timer.reset()
    self._graph = ops.Graph()
    self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
    with self._graph.as_default():
      with variable_scope.variable_scope('', use_resource=True):
        training_util.get_or_create_global_step()
      features, input_hooks = self._estimator._get_features_from_input_fn(  # pylint: disable=protected-access
          self._input_fn, model_fn_lib.ModeKeys.PREDICT)
      estimator_spec = self._estimator._call_model_fn(  # pylint: disable=protected-access
          features, None, model_fn_lib.ModeKeys.PREDICT, self._estimator.config)

      self._all_hooks = list(input_hooks) + list(estimator_spec.prediction_hooks)
      self._predictions = self._estimator._extract_keys(  # pylint: disable=protected-access
          estimator_spec.predictions,
          predict_keys=None)
      self._var_name_to_eval_var = {
          v.name: v for v in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      }
      self._var_name_to_placeholder = {
          v.name: array_ops.placeholder(v.dtype)
          for v in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      }
      logging.info('Placeholders: %s', self._var_name_to_placeholder)

      for h in self._all_hooks:
        logging.info('Hook: %s', h)
        if isinstance(h, tpu_estimator.TPUInfeedOutfeedSessionHook):
          h._should_initialize_tpu = False  # pylint: disable=protected-access

  def after_create_session(self, session, coord):  # pylint: disable=unused-argument
    """Does first run which shows the eval metrics before training."""
    if ops.get_collection(ops.GraphKeys.SAVEABLE_OBJECTS):
      raise ValueError(
          'InMemoryEval does not support saveables other than global '
          'variables.')
    logging.info('Predict: Building var map')
    self._var_name_to_train_var = {
        v.name: v for v in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    }
    logging.info('Predict: Building transfer set')
    var_names_to_transfer = set(self._var_name_to_placeholder.keys()) & set(
        self._var_name_to_train_var.keys())
    logging.info('Predict: Building filtering evaluation vars')
    # Filter training var names that are not exist in evaluation
    self._var_name_to_train_var = {
        v_name: self._var_name_to_train_var[v_name]
        for v_name in var_names_to_transfer
    }
    logging.info('Predict: Building filtering training vars')
    # Filter eval var names that are not exist in training
    self._var_name_to_eval_var = {
        v_name: self._var_name_to_eval_var[v_name]
        for v_name in var_names_to_transfer
    }

    logging.info('Predict: Building feed op')
    with self._graph.as_default():
      self._var_feed_op = control_flow_ops.group([
          state_ops.assign(self._var_name_to_eval_var[v_name],
                           self._var_name_to_placeholder[v_name])
          for v_name in var_names_to_transfer
      ])
    logging.info('Predict: Done building.')

  def _predict(self, run_ctx, step):
    var_name_to_value = run_ctx.session.run(self._var_name_to_train_var)
    logging.info('Building placeholders.')
    placeholder_to_value = {
        self._var_name_to_placeholder[v_name]: var_name_to_value[v_name]
        for v_name in var_name_to_value
    }

    def feed_variables(scaffold, session):
      del scaffold
      session.run(self._var_feed_op, feed_dict=placeholder_to_value)

    logging.info('Building scaffold.')
    scaffold = training.Scaffold(init_fn=feed_variables)

    with self._graph.as_default():
      session_creator = monitored_session.ChiefSessionCreator(
          scaffold=scaffold,
          checkpoint_filename_with_path=None,
          master=run_ctx.session.sess_str)

      self._handler.setup(step)
      logging.info('Setup done.')
      with monitored_session.MonitoredSession(
          session_creator=session_creator,
          hooks=self._all_hooks) as predict_session:
        while not predict_session.should_stop():
          logging.info('Predicting.... %s', self._predictions)
          preds_evaluated = predict_session.run(self._predictions)
          if not isinstance(self._predictions, dict):
            for pred in preds_evaluated:
              self._handler.handle_prediction(pred)
          else:
            for i in range(self._estimator._extract_batch_length(preds_evaluated)):
              self._handler.handle_prediction({
                  key: value[i]
                  for key, value in six.iteritems(preds_evaluated)
              })

      logging.info('Finalizing.')
      self._handler.finalize(step)

    logging.info('Done with prediction.')
    self._timer.update_last_triggered_step(step)

  def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
    """Runs evaluator."""
    step = run_context.session.run(self._global_step_tensor)
    if self._timer.should_trigger_for_step(step):
      self._predict(run_context, step)

  def end(self, session):  # pylint: disable=unused-argument
    """Runs evaluator for final model."""
    step = session.run(self._global_step_tensor)
    run_ctx = session_run_hook.SessionRunContext({}, session)
    self._predict(run_ctx, step)
