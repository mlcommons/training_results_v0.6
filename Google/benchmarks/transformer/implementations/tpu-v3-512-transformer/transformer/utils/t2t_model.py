"""T2TModel Base Class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import functools
import math
import six

import tensorflow as tf

from tensorflow.python.layers import base
from tensorflow.python.ops import inplace_ops
from data_generators import text_encoder
from data_generators.problem import problem_hparams_to_features
from layers import common_layers
from utils import decoding
from utils import learning_rate
from utils import metrics
from utils import modality
from utils import optimize
from utils import quantization
from utils import registry


_no_problem_err_str = (
    "The default implementation of %s requires that the "
    "model be used with a Problem. If using a Problem, augment the "
    "hparams object with trainer_lib.add_problem_hparams. If not, "
    "override %s.")
_no_problem_err = (
    lambda method_name: _no_problem_err_str % (method_name, method_name))


class T2TModel(base.Layer):
  """Abstract base class for models.

  `T2TModel` has three typical usages:

  1. Estimator: The method `make_estimator_model_fn` builds a `model_fn` for
     the tf.Estimator workflow of training, evaluation, and prediction.
     It performs the method `call`, which performs the core computation,
     followed by `estimator_spec_train`, `estimator_spec_eval`, or
     `estimator_spec_predict` depending on the tf.Estimator mode.
  2. Layer: The method `call` enables `T2TModel` to be used a callable by
     itself. It calls the following methods:

     * `bottom`, which transforms features according to `problem_hparams`' input
       and target `Modality`s;
     * `body`, which takes features and performs the core model computation to
        return output and any auxiliary loss terms;
     * `top`, which takes features and the body output, and transforms them
       according to `problem_hparams`' input and target `Modality`s to return
       the final logits;
     * `loss`, which takes the logits, forms any missing training loss, and sums
       all loss terms.
  3. Inference: The method `infer` enables `T2TModel` to make sequence
     predictions by itself.

  Subclasses generally only need to override `body`.
  """
  REGISTERED_NAME = None  # Updated on registration.

  def __init__(self,
               hparams,
               mode=tf.estimator.ModeKeys.TRAIN,
               problem_hparams=None,
               decode_hparams=None,
               **kwargs):
    """Creates a T2TModel.

    Args:
      hparams: tf.contrib.training.HParams, model hyperparameters.
      mode: tf.estimator.ModeKeys, the execution mode.
      problem_hparams: tf.contrib.training.HParams, hyperparameters for the
        Problem. If provided here or in hparams.problem_hparams, the model will
        automatically determine bottom, top, and loss methods. If not provided,
        calling the model will only invoke body.
      decode_hparams: a hyperparameter object with decoding parameters.
        See decoding.decode_hparams.
      **kwargs: arguments to pass to base.Layer constructor.
    """
    # Determine name first: use registered name if possible, class name else.
    default_name = registry.default_name(type(self))
    name = self.REGISTERED_NAME or default_name
    super(T2TModel, self).__init__(
        trainable=mode == tf.estimator.ModeKeys.TRAIN, name=name, **kwargs)

    if not problem_hparams and hasattr(hparams, "problem_hparams"):
      problem_hparams = hparams.problem_hparams
    self._problem_hparams = problem_hparams

    # Setup hparams
    hparams = copy.copy(hparams)
    if self._problem_hparams and hparams.shared_embedding_and_softmax_weights:
      # If vocabularies differ, unset shared_embedding_and_softmax_weights.
      input_modality = self._problem_hparams.modality.get("inputs")
      target_modality = self._problem_hparams.modality.get("targets")
      if (isinstance(input_modality, modality.Modality) and
          isinstance(target_modality, modality.Modality) and
          input_modality.top_dimensionality !=
          target_modality.top_dimensionality):
        log_info("Unsetting shared_embedding_and_softmax_weights.")
        hparams.shared_embedding_and_softmax_weights = 0

    self._original_hparams = hparams
    self.set_mode(mode)

    self._decode_hparams = copy.copy(decode_hparams or
                                     decoding.decode_hparams())
    self._variable_scopes = {}

  def _add_variable_scope(self, key, vs):
    if key not in self._variable_scopes:
      self._variable_scopes[key] = vs

  @property
  def hparams(self):
    return self._hparams

  @property
  def is_training(self):
    return self._hparams.mode == tf.estimator.ModeKeys.TRAIN

  @property
  def is_predicting(self):
    return self._hparams.mode == tf.estimator.ModeKeys.PREDICT

  @property
  def has_input(self):
    if self._problem_hparams:
      return "inputs" in self._problem_hparams.modality
    else:
      return True

  @property
  def _custom_getter(self):
    if self.hparams.weight_dtype == "bfloat16":
      if self.hparams.optimizer != "Adafactor":
        raise NotImplementedError(
            "weight_dtype=bfloat16 only implemented with Adafactor optimizer")
      return quantization.EighthPowerEncoding().custom_getter(
          activation_dtype=tf.bfloat16
          if self.hparams.activation_dtype == "bfloat16" else tf.float32)
    elif self.hparams.activation_dtype == "bfloat16":
      return quantization.bfloat16_activations_var_getter
    else:
      return None

  @property
  def _target_modality_is_real(self):
    """Whether the target modality is real-valued."""
    target_modality = self._problem_hparams.modality["targets"]
    return target_modality.name.startswith("real_")

  def call(self, inputs, **kwargs):
    del kwargs
    features = inputs
    set_custom_getter_compose(self._custom_getter)
    tf.get_variable_scope().set_initializer(
        optimize.get_variable_initializer(self.hparams))
    self._fill_problem_hparams_features(features)
    logits, losses = self.model_fn(features)

    new_losses = {}
    for loss_name in sorted(losses):
      if isinstance(losses[loss_name], tuple):
        loss_num, loss_den = losses[loss_name]
        real_loss = loss_num / tf.maximum(
            tf.cast(1.0, loss_den.dtype), loss_den)
      else:
        real_loss = losses[loss_name]
      new_losses[loss_name] = real_loss

    return logits, new_losses

  def model_fn(self, features):
    with tf.variable_scope(tf.get_variable_scope(), use_resource=True) as vs:
      self._add_variable_scope("model_fn", vs)
      transformed_features = self.bottom(features)

      if self.hparams.activation_dtype == "bfloat16":
        for k, v in sorted(six.iteritems(transformed_features)):
          if v.dtype == tf.float32:
            transformed_features[k] = tf.cast(v, tf.bfloat16)

      with tf.variable_scope("body") as body_vs:
        self._add_variable_scope("body", body_vs)
        log_info("Building model body")
        body_out = self.body(transformed_features)
      output, losses = self._normalize_body_output(body_out)

      if "training" in losses:
        log_info("Skipping T2TModel top and loss because training loss "
                 "returned from body")
        logits = output
      else:
        logits = self.top(output, features)
        losses["training"] = 0.0
        if self._hparams.mode != tf.estimator.ModeKeys.PREDICT:
          losses["training"] = self.loss(logits, features)

      return logits, losses

  def bottom(self, features):
    """Transforms features to feed into body.

    Args:
      features: dict of str to Tensor. Typically it is the preprocessed data
        batch after Problem's preprocess_example().

    Returns:
      transformed_features: dict of same key-value pairs as features. The value
        Tensors are newly transformed.
    """
    if not self._problem_hparams:
      log_warn("Without a Problem, T2TModel.bottom is a passthrough.")
      return features

    transformed_features = collections.OrderedDict()
    all_previous_modalities = []
    target_modality = _create_target_modality(self._problem_hparams.modality)

    # Transform features via its corresponding modality.
    for feature_name, modality_obj in sorted(
        six.iteritems(self._problem_hparams.modality)):
      if feature_name not in features:
        tf.logging.warning("Missing feature %s - ignoring." % feature_name)
        continue
      # Use if-else clauses to preserve behavior of previous changes: namely,
      # the variable scope name for the targets feature if there is only one
      # target modality; and to reuse variable scopes for only input modalities.
      if feature_name in target_modality:
        if len(target_modality) > 1:
          variable_scope_name = "%s/%s" % (modality_obj.name, feature_name)
        else:
          variable_scope_name = modality_obj.name
        # TODO(aidangomez): share variables?
        with tf.variable_scope(variable_scope_name) as vs:
          self._add_variable_scope(variable_scope_name, vs)
          log_info("Transforming feature '%s' with %s.targets_bottom",
                   feature_name,
                   modality_obj.name)
          transformed_features[feature_name] = modality_obj.targets_bottom(
              features[feature_name])
      else:
        do_reuse = modality_obj.name in all_previous_modalities
        with tf.variable_scope(modality_obj.name, reuse=do_reuse) as vs:
          self._add_variable_scope(modality_obj.name, vs)
          log_info("Transforming feature '%s' with %s.bottom",
                   feature_name,
                   modality_obj.name)
          transformed_features[feature_name] = modality_obj.bottom(
              features[feature_name])
        all_previous_modalities.append(modality_obj.name)

    for key in features:
      if key not in transformed_features:
        # For features without a modality, we pass them along as is
        transformed_features[key] = features[key]
      else:
        # Other features get passed along with the "raw" suffix
        transformed_features[key + "_raw"] = features[key]

    return transformed_features

  def body(self, features):
    """Computes the targets' pre-logit activations given transformed inputs.

    Most `T2TModel` subclasses will override this method.

    Args:
      features: dict of str to Tensor, where each Tensor has shape [batch_size,
        ..., hidden_size]. It typically contains keys `inputs` and `targets`.

    Returns:
      output: Tensor of pre-logit activations with shape [batch_size, ...,
              hidden_size].
      losses: Either single loss as a scalar, a list, a Tensor (to be averaged),
              or a dictionary of losses. If losses is a dictionary with the key
              "training", losses["training"] is considered the final training
              loss and output is considered logits; self.top and self.loss will
              be skipped.
    """
    raise NotImplementedError("Abstract Method")

  def _top_single(self, body_output, target_modality, features):
    if not target_modality:
      log_warn("Without a Problem, T2TModel.top is a passthrough.")
      return body_output

    with tf.variable_scope(target_modality.name) as tm_vs:
      self._add_variable_scope(tm_vs.name, tm_vs)
      log_info("Transforming body output with %s.top", target_modality.name)
      last_only = (
          target_modality.top_is_pointwise and
          self.hparams.mode == tf.estimator.ModeKeys.PREDICT and
          not self.hparams.force_full_predict)
      if not last_only:
        logits = target_modality.top(body_output, features.get("targets"))
      else:
        # Take body outputs for the last position only, and targets too.
        if "decode_loop_step" not in features:
          last_position_body_output = tf.expand_dims(
              body_output[:, -1, :, :], axis=[1])
          last_position_targets = tf.expand_dims(
              features["targets"][:, -1, :, :], axis=[1])
        else:
          body_output_shape = body_output.shape.as_list()
          last_position_body_output = tf.slice(
              body_output, [0, features["decode_loop_step"][0], 0, 0], [
                  body_output_shape[0], 1, body_output_shape[2],
                  body_output_shape[3]
              ])
          target_shape = features["targets"].shape.as_list()
          last_position_targets = tf.slice(
              features["targets"], [0, features["decode_loop_step"][0], 0, 0],
              [target_shape[0], 1, target_shape[2], target_shape[3]])
        logits = target_modality.top(last_position_body_output,
                                     last_position_targets)
    return logits

  def top(self, body_output, features):
    """Computes logits given body output and features.

    Args:
      body_output: dict of str to Tensor, comprising one key-value pair for each
        target. Each value denotes the target's pre-logit activations.
        Alternatively, it may be a single Tensor denoting the pre-logits for
        that target.
      features: dict of str to Tensor. Typically it is the preprocessed data
        batch after Problem's preprocess_example().

    Returns:
      logits: dict of str to Tensor, denoting each logits for each target; or
        a single Tensor denoting the logits for that target.
        When targets are generated at training time:
          logits == {
            "self_generated_targets": <generated targets tensor>
            "logits": <original logits Tensor or dict>
          }
    """
    if isinstance(body_output, dict):
      if self._problem_hparams:
        target_modality = _create_target_modality(
            self._problem_hparams.modality)
      else:
        target_modality = {k: None for k in body_output.keys()}
      for k in body_output.keys():
        assert k in target_modality.keys(), (
            "The key %s of model_body's returned logits dict must be in "
            "problem_hparams.modality's dict." % k)
      logits = {}
      for k, v in six.iteritems(body_output):
        # TODO(aidangomez): share variables here?
        with tf.variable_scope(k) as top_vs:
          self._add_variable_scope("top_%s" % k, top_vs)
          logits[k] = self._top_single(v, target_modality[k], features)
      return logits
    else:
      if self._problem_hparams:
        target_modality = _create_target_modality(
            self._problem_hparams.modality)
      else:
        target_modality = None
      if isinstance(target_modality, dict):
        assert "targets" in target_modality, (
            "model_body returned single logits so 'targets' must be a key "
            "since problem_hparams.modality is a dict.")
        target_modality = target_modality["targets"]
      return self._top_single(body_output, target_modality, features)

  def _loss_single(self, logits, target_modality, feature):
    # The current bfloat16 version still uses float32 for most parts of backward
    # propagation to keep model quality, so cast back before computing the loss
    # value.
    if not target_modality:
      log_warn(_no_problem_err("loss"))
      return (tf.constant(0., dtype=tf.float32),
              tf.constant(1., dtype=tf.float32))

    loss_num, loss_den = target_modality.loss(logits, feature)
    loss_num *= self._problem_hparams.loss_multiplier

    return loss_num, loss_den

  def loss(self, logits, features):
    if isinstance(logits, dict):
      if self._problem_hparams:
        target_modality = _create_target_modality(
            self._problem_hparams.modality)
      else:
        target_modality = {k: None for k in logits.keys()}
      for k in logits.keys():
        assert k in target_modality.keys(), (
            "The key %s of model_body's returned logits dict must be in "
            "problem_hparams.modality's dict." % k)
      losses = {}
      for k, v in six.iteritems(logits):
        losses[k] = self._loss_single(v, target_modality[k], features[k])

        n, d = losses[k]
        if common_layers.should_generate_summaries():
          tf.summary.scalar(k + "_loss", n / d)
          tf.summary.scalar(k + "_loss_num", n)
          tf.summary.scalar(k + "_loss_den", d)
          if getattr(self.hparams, "visualize_logits_histogram", False):
            hist = tf.summary.histogram
            hist(k + "_predict", tf.argmax(tf.squeeze(v), axis=-1))
            hist(k + "_targets", features[k])

      return tf.add_n([n / d for n, d in losses.values()])
    else:
      if self._problem_hparams:
        target_modality = _create_target_modality(
            self._problem_hparams.modality)
      else:
        target_modality = None
      if isinstance(target_modality, dict):
        assert "targets" in target_modality, (
            "model_body returned single logits so 'targets' must be a key "
            "since problem_hparams.modality is a dict.")
        target_modality = target_modality["targets"]
      return self._loss_single(logits, target_modality, features["targets"])

  def optimize(self, loss, num_async_replicas=1, use_tpu=False):
    """Return a training op minimizing loss."""
    lr = learning_rate.learning_rate_schedule(self.hparams)
    if num_async_replicas > 1:
      log_info("Dividing learning rate by num_async_replicas: %d",
               num_async_replicas)
    lr /= math.sqrt(float(num_async_replicas))
    train_op = optimize.optimize(loss, lr, self.hparams, use_tpu=use_tpu)
    return train_op

  def set_mode(self, mode):
    """Set hparams with the given mode."""
    log_info("Setting T2TModel mode to '%s'", mode)
    hparams = copy.copy(self._original_hparams)
    hparams.add_hparam("mode", mode)
    # When not in training mode, set all forms of dropout to zero.
    if mode != tf.estimator.ModeKeys.TRAIN:
      for key in hparams.values():
        if key.endswith("dropout") or key == "label_smoothing":
          log_info("Setting hparams.%s to 0.0", key)
          setattr(hparams, key, 0.0)
    self._hparams = hparams

    if self._problem_hparams:
      # Set model hparams in problem_hparams' modalities, which also store them.
      for modality_obj in six.itervalues(self._problem_hparams.modality):
        if modality_obj is not None:
          modality_obj._model_hparams = self._hparams  # pylint: disable=protected-access

  def _fill_problem_hparams_features(self, features):
    if features is not None:
      for k, v in sorted(
          six.iteritems(problem_hparams_to_features(self._problem_hparams))):
        if k not in features:
          features[k] = tf.constant(v, name=k)

  def infer(self,
            features=None,
            decode_length=50,
            beam_size=1,
            top_beams=1,
            alpha=0.0,
            use_tpu=False):
    """A inference method.

    Quadratic time in decode_length.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for longer translations.
      use_tpu: bool, whether to build the inference graph for TPU.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, top_beams, <= decode_length].
          "scores": decoding log probs from the beam search.
      }

    Raises:
      NotImplementedError: If beam_size is one.
    """
    set_custom_getter_compose(self._custom_getter)
    if not self.has_input and beam_size > 1:
      log_warn("Beam searching for a model with no inputs.")
    self._fill_problem_hparams_features(features)

    if self._problem_hparams:
      target_modality = self._problem_hparams.modality["targets"]
    if beam_size == 1:
      raise NotImplementedError(
          "Greedy Decoding is not supported in this MLPerf version.")
    else:
      log_info("Beam Decoding with beam size %d" % beam_size)
      results = self._beam_decode(features, decode_length, beam_size,
                                  top_beams, alpha, use_tpu)

    return results

  def _beam_decode(self,
                   features,
                   decode_length,
                   beam_size,
                   top_beams,
                   alpha,
                   use_tpu=False):
    """Beam search decoding.

    Models should ideally implement a more efficient version of this function.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for longer translations.
      use_tpu: A bool, whether to do beam decode on TPU.

    Returns:
       samples: an integer `Tensor`. Top samples from the beam search
    """
    raise NotImplementedError("Abstract Method")

  @staticmethod
  def make_estimator_model_fn(model_name,
                              hparams,
                              decode_hparams=None,
                              use_tpu=False):
    model_cls = registry.model(model_name)

    def wrapping_model_fn(features, labels, mode, params=None, config=None):
      return model_cls.estimator_model_fn(
          hparams,
          features,
          labels,
          mode,
          config=config,
          params=params,
          decode_hparams=decode_hparams,
          use_tpu=use_tpu)

    return wrapping_model_fn

  @classmethod
  def estimator_model_fn(cls,
                         hparams,
                         features,
                         labels,
                         mode,
                         config=None,
                         params=None,
                         decode_hparams=None,
                         use_tpu=False):
    """Model fn for Estimator.

    Args:
      hparams: HParams, model hyperparameters
      features: dict<str name, Tensor feature>
      labels: Tensor
      mode: tf.estimator.ModeKeys
      config: RunConfig
      params: dict, may include batch_size, use_tpu
      decode_hparams: HParams, used when mode == PREDICT.
      use_tpu: A bool, whether to build the inference graph for TPU.

    Returns:
      TPUEstimatorSpec if use tpu else EstimatorSpec
    """
    hparams = copy.deepcopy(hparams)

    # Instantiate model
    reuse = tf.get_variable_scope().reuse
    model = cls(
        hparams,
        mode,
        decode_hparams=decode_hparams,
        _reuse=reuse)

    # PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
      if use_tpu:
        inputs = features["inputs"]
        shape = inputs.get_shape().as_list()
        if shape[0] is None:
          shape[0] = decode_hparams.batch_size or hparams.batch_size
        if shape[1] is None:
          shape[1] = hparams.max_input_seq_length or hparams.max_length
        inputs.set_shape(shape)
      return model.estimator_spec_predict(features, use_tpu=use_tpu)

    # TRAIN and EVAL modes
    logits, losses_dict = model(features)  # pylint: disable=not-callable

    # Support model-generated labels by overriding features["targets"] with
    # logits["self_generated_targets"].
    if isinstance(logits, dict) and "self_generated_targets" in logits:
      # Overwrite 'features["targets"]' and 'labels'
      # by logits["self_generated_targets"].
      tf.logging.info("Replacing targets with model-provided targets.")
      features["targets"] = labels = logits.pop("self_generated_targets")
      assert logits.keys() == ["logits"], (
          # See "Returns" in the "top" method docstring for the expected
          # "logits" format when targets are generated at training time.
          "Expect only key 'logits' when there is 'self_generated_targets'. "
          "Found {}".format(logits.keys())
      )
      # Recover the original logits tensor from the logits dict.
      logits = logits["logits"]  # Can be a tf.Tensor or a dict.

    # Set known shapes
    if isinstance(logits, dict):
      for k, v in sorted(six.iteritems(logits)):
        if "scalar/" in k:
          continue

        shape = v.get_shape().as_list()
        if shape[0] is None:
          shape[0] = params["batch_size"]
        if shape[1] is None:
          shape[1] = hparams.max_length
        v.set_shape(shape)
    else:
      shape = logits.get_shape().as_list()
      if shape[0] is None:
        shape[0] = params["batch_size"]
      if shape[1] is None:
        shape[1] = hparams.max_length
      logits.set_shape(shape)

    assert "training" in losses_dict

    # Summarize losses
    model._summarize_losses(losses_dict)  # pylint: disable=protected-access

    # Accumulate losses
    loss = sum(losses_dict[key] for key in sorted(losses_dict.keys()))

    # EVAL mode
    if mode == tf.estimator.ModeKeys.EVAL:
      return model.estimator_spec_eval(features, logits, labels, loss,
                                       losses_dict)

    # TRAIN mode
    assert mode == tf.estimator.ModeKeys.TRAIN
    num_async_replicas = (1 if (use_tpu or not config) else
                          config.t2t_device_info["num_async_replicas"])
    return model.estimator_spec_train(
        loss, num_async_replicas=num_async_replicas, use_tpu=use_tpu)

  def estimator_spec_train(self, loss, num_async_replicas=1, use_tpu=False):
    """Constructs `tf.estimator.EstimatorSpec` for TRAIN (training) mode."""
    train_op = self.optimize(loss, num_async_replicas=num_async_replicas,
                             use_tpu=use_tpu)

    if use_tpu:
      # Note: important to call this before remove_summaries()
      if self.hparams.tpu_enable_host_call:
        host_call = create_host_call(self.hparams.model_dir)
      else:
        host_call = None

      remove_summaries()

      return tf.contrib.tpu.TPUEstimatorSpec(
          tf.estimator.ModeKeys.TRAIN,
          loss=loss,
          train_op=train_op,
          host_call=host_call,
          scaffold_fn=None)
    else:
      return tf.estimator.EstimatorSpec(
          tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op)

  def estimator_spec_eval(self, features, logits, labels, loss, losses_dict):
    """Constructs `tf.estimator.EstimatorSpec` for EVAL (evaluation) mode."""
    del losses_dict
    hparams = self.hparams

    if not hasattr(hparams, "problem"):
      raise NotImplementedError(_no_problem_err("estimator_spec_eval"))

    problem = hparams.problem

    remove_summaries()
    if isinstance(logits, dict):
      eval_metrics_fn = create_tpu_eval_metrics_fn(problem, hparams)
      # For TPU, logits dict will be passed as keyword arguments to
      # eval_metrics_fn. Here we add the labels to those arguments.
      logits.update({"labels": labels})
      return tf.contrib.tpu.TPUEstimatorSpec(
          tf.estimator.ModeKeys.EVAL,
          eval_metrics=(eval_metrics_fn, logits),
          loss=loss)
    else:
      eval_metrics_fn = create_tpu_eval_metrics_fn(problem, hparams)
      return tf.contrib.tpu.TPUEstimatorSpec(
          tf.estimator.ModeKeys.EVAL,
          eval_metrics=(eval_metrics_fn, [logits, labels]),
          loss=loss)

  def estimator_spec_predict(self, features, use_tpu=False):
    """Constructs `tf.estimator.EstimatorSpec` for PREDICT (inference) mode."""
    decode_hparams = self._decode_hparams
    infer_out = self.infer(
        features,
        beam_size=decode_hparams.beam_size,
        top_beams=1,
        alpha=decode_hparams.alpha,
        decode_length=decode_hparams.extra_length,
        use_tpu=use_tpu)
    if isinstance(infer_out, dict):
      outputs = infer_out["outputs"]
      scores = infer_out["scores"]
    else:
      outputs = infer_out
      scores = None

    inputs = features.get("inputs")
    if inputs is None:
      inputs = features["targets"]

    predictions = {
        "outputs": outputs,
        "scores": scores,
        "inputs": inputs,
        "targets": features.get("infer_targets"),
    }

    # Pass through remaining features
    for name, feature in features.items():
      if name not in list(predictions.keys()) + ["infer_targets"]:
        if name == "decode_loop_step":
          continue
        if not feature.shape.as_list():
          # All features must have a batch dimension
          batch_size = common_layers.shape_list(outputs)[0]
          feature = tf.tile(tf.expand_dims(feature, 0), [batch_size])
        predictions[name] = feature

    _del_dict_non_tensors(predictions)

    export_out = {"outputs": predictions["outputs"]}
    if "scores" in predictions:
      export_out["scores"] = predictions["scores"]

    # Necessary to rejoin examples in the correct order with the Cloud ML Engine
    # batch prediction API.
    if "batch_prediction_key" in predictions:
      export_out["batch_prediction_key"] = predictions["batch_prediction_key"]

    remove_summaries()

    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            tf.estimator.export.PredictOutput(export_out)
    }
    if use_tpu:
      return tf.contrib.tpu.TPUEstimatorSpec(
          tf.estimator.ModeKeys.PREDICT,
          predictions=predictions,
          export_outputs=export_outputs)
    else:
      return tf.estimator.EstimatorSpec(
          tf.estimator.ModeKeys.PREDICT,
          predictions=predictions,
          export_outputs=export_outputs)

  def _normalize_body_output(self, body_out):
    if isinstance(body_out, tuple):
      output, losses = body_out
      if not isinstance(losses, dict):
        losses = {"extra": tf.reduce_mean(losses)}
    else:
      output = body_out
      losses = {"extra": 0.0}

    return output, losses

  def _summarize_losses(self, losses_dict):
    """Adds `tf.summary`s to all terms in the losses dictionary."""
    if common_layers.should_generate_summaries():
      with tf.name_scope("losses"):
        for loss_name, loss_val in sorted(losses_dict.items()):
          tf.summary.scalar(loss_name, loss_val)


def create_tpu_eval_metrics_fn(problem, model_hparams):
  """Create the metrics_fn that TPUEstimatorSpec expects."""

  metric_fns = []
  eval_metrics = problem.eval_metrics()

  tm = _create_target_modality(problem.get_hparams(model_hparams).modality)
  if isinstance(tm, dict):
    for k, v in six.iteritems(tm):
      weights_fn = v.targets_weights_fn

      def make_metric_fn(metric_fn):

        def wrapped_metric_fn(logits, labels, weights_fn=weights_fn):
          num, den = metric_fn(logits, labels, weights_fn=weights_fn)
          return tf.metrics.mean(num, den)

        return wrapped_metric_fn

      for metric in eval_metrics:
        name = "%s/metrics-%s/%s" % (k, problem.name, metric)
        metric_fns.append((name, make_metric_fn(metrics.METRICS_FNS[metric])))
  else:
    weights_fn = tm.targets_weights_fn

    def make_metric_fn(metric_fn):

      def wrapped_metric_fn(logits, labels):
        num, den = metric_fn(logits, labels, weights_fn=weights_fn)
        return tf.metrics.mean(num, den)

      return wrapped_metric_fn

    for metric in eval_metrics:
      name = "metrics-%s/%s" % (problem.name, metric)
      metric_fns.append((name, make_metric_fn(metrics.METRICS_FNS[metric])))

  def all_metrics_fn(logits=None, labels=None, **kwargs):
    """Construct metrics dictionary."""
    metrics_dict = {}

    if logits is None:
      logits = kwargs

    for name, fn in metric_fns:
      if isinstance(logits, dict) and isinstance(labels, dict):
        for k, v in six.iteritems(logits):
          metrics_dict["%s/%s" % (k, name)] = fn(v, labels[k])
      elif isinstance(logits, dict):
        tf.logging.warning("Logits is a dict, but labels is not; only "
                           "evaluating logits['targets'] against labels.")
        metrics_dict["%s/%s" % ("targets", name)] = fn(logits["targets"],
                                                       labels)
      else:
        metrics_dict[name] = fn(logits, labels)

    return metrics_dict

  return all_metrics_fn


def remove_summaries():
  """Remove summaries from the default graph."""
  g = tf.get_default_graph()
  key = tf.GraphKeys.SUMMARIES
  log_debug("Remove summaries %s" % str(g.get_collection(key)))
  del g.get_collection_ref(key)[:]
  assert not g.get_collection(key)


def create_host_call(model_dir):
  """Construct a host_call writing scalar summaries.

  Args:
    model_dir: String containing path to train

  Returns:
    (fn, args) Pair to be called by TPUEstimator as the host_call.
  """
  graph = tf.get_default_graph()
  summaries = graph.get_collection(tf.GraphKeys.SUMMARIES)
  gs_t = tf.reshape(tf.to_int32(tf.train.get_global_step()), [1])
  summary_kwargs = collections.OrderedDict()
  for t in summaries:
    # TODO(aidangomez): enable ImageSummary support when we have a faster method
    # see @shibow's comment in cl/202344570
    if t.op.type not in ["ScalarSummary"]:
      tf.logging.warn("Ignoring unsupported tf.Summary type %s" % t.op.type)
      continue

    name = t.op.name
    tensor = t.op.inputs[1]
    if t.op.type == "ScalarSummary":
      assert tensor.shape.is_compatible_with([])
      if tensor.dtype == tf.int64:
        tensor = tf.to_int32(tensor)
      summary_kwargs["ScalarSummary" + name] = tf.reshape(tensor, [1])
    elif t.op.type == "ImageSummary":
      # TODO(aidangomez): as we move to support more types, update
      # common_layers.tpu_safe_image_summary
      if tensor.dtype != tf.float32:
        tf.logging.warn(
            "Currently T2T on TPU only supports ImageSummary of "
            "tf.float32-type Tensors. Skipping Tensor "
            "%s with dtype %s..." % (tensor.name, tensor.dtype))
        continue
      # tensor = tf.to_float(tensor)
      summary_kwargs["ImageSummary" + name] = tensor
  # When no supported summaries are found, don't create host_call. Otherwise,
  # TPU outfeed queue would enqueue global_step while host_call doesn't dequeue
  # it, eventually causing hang.
  if not summary_kwargs:
    return None
  summary_kwargs["global_step"] = gs_t
  log_info("summary_kwargs %s" % str(summary_kwargs))

  def host_call_fn(**kwargs):
    """Training host call. Creates summaries for training metrics.

    Args:
      **kwargs: Dict of {str: Tensor} , with `Tensor` of shape `[batch]`. Must
        contain key "global_step" with value of current global_step Tensor.

    Returns:
      List of summary ops to run on the CPU host.
    """
    gs = tf.to_int64(kwargs.pop("global_step")[0])
    with tf.contrib.summary.create_file_writer(model_dir).as_default():
      with tf.contrib.summary.always_record_summaries():
        # We need to use tf.contrib.summary in order to feed the `step`.
        for name, value in sorted(six.iteritems(kwargs)):
          if name.startswith("ScalarSummary"):
            name = name[len("ScalarSummary"):]
            tf.contrib.summary.scalar(
                name, tf.reduce_mean(tf.to_float(value)), step=gs)
          elif name.startswith("ImageSummary"):
            name = name[len("ImageSummary"):]
            tf.contrib.summary.image(name, value, step=gs)

        return tf.contrib.summary.all_summary_ops()

  return (host_call_fn, summary_kwargs)


def _del_dict_non_tensors(d):
  for k in list(d.keys()):
    if not isinstance(d[k], tf.Tensor):
      del d[k]


_already_logged = set()


def _eager_log(level, *args):
  if tf.contrib.eager.in_eager_mode() and args in _already_logged:
    return
  _already_logged.add(args)
  getattr(tf.logging, level)(*args)


def log_debug(*args):
  _eager_log("debug", *args)


def log_info(*args):
  _eager_log("info", *args)


def log_warn(*args):
  _eager_log("warn", *args)


def _compose_custom_getters(getter_a, getter_b):
  """Compose two custom getters.

  Example use:
  tf.get_variable_scope().set_custom_getter(
    compose_custom_getters(tf.get_variable_scope().custom_getter, new_getter))

  This composes getters in the same way as creating a new variable scope with
  the new_getter, but it does not actually create a new variable scope.

  Args:
    getter_a: a custom getter - generally from the existing variable scope.
    getter_b: a custom getter

  Returns:
    a custom getter
  """
  if not getter_a:
    return getter_b
  if not getter_b:
    return getter_a

  def getter_fn(getter, *args, **kwargs):
    return getter_b(functools.partial(getter_a, getter), *args, **kwargs)

  return getter_fn


def set_custom_getter_compose(custom_getter):
  """Set a custom getter in the current variable scope.

  Do not overwrite the existing custom getter - rather compose with it.

  Args:
    custom_getter: a custom getter.
  """
  tf.get_variable_scope().set_custom_getter(
      _compose_custom_getters(tf.get_variable_scope().custom_getter,
                              custom_getter))


def _create_target_modality(modality_dict):
  # TODO(trandustin): We require this in order to apply methods utilized
  # differently for modalities which are "targets"
  # (e.g., modality.target_bottom). In the future, remove need for this
  # behavior.
  return {k: v for k, v in six.iteritems(modality_dict) if "target" in k
          and k != "targets_segmentation" and k != "targets_position"}
