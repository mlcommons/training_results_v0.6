"""Hyperparameters and ranges common to multiple models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import zip  # pylint: disable=redefined-builtin

import tensorflow as tf
from utils import registry


@registry.register_hparams("basic_1")
def basic_params1():
  """A set of basic hyperparameters."""
  return tf.contrib.training.HParams(
      # If the problem consists of variable-length sequences
      # (see problem.batch_size_means_tokens()), then this is the number
      # of tokens per batch per GPU or per TPU core.  Otherwise, this is
      # the number of examples per GPU or per TPU core.
      batch_size=4096,
      batch_shuffle_size=0,
      # If True, then if the features are of variable length, the batch_size is
      # used as the actual batch size (and not tokens per batch).
      use_fixed_batch_size=False,
      num_hidden_layers=4,
      hidden_size=64,
      # All hyperparameters ending in "dropout" are automatically set to 0.0
      # when not in training mode.
      dropout=0.2,
      clip_grad_norm=2.0,
      grad_noise_scale=0.0,
      # Flag for whether mlperf mode is on
      mlperf_mode=False,
      initializer="orthogonal",
      initializer_gain=1.5,
      label_smoothing=0.1,
      optimizer="Adam",
      optimizer_adam_epsilon=1e-6,
      optimizer_adam_beta1=0.85,
      optimizer_adam_beta2=0.997,
      optimizer_momentum_momentum=0.9,
      optimizer_momentum_nesterov=False,
      optimizer_adafactor_beta1=0.0,
      optimizer_adafactor_beta2=0.999,
      optimizer_adafactor_factored=True,
      optimizer_adafactor_decay_type="pow",
      optimizer_adafactor_memory_exponent=0.8,
      optimizer_adafactor_clipping_threshold=1.0,
      optimizer_adafactor_multiply_by_parameter_scale=True,
      weight_decay=1e-6,
      weight_noise=0.0,
      # Defines the learning rate as a product of named functions.
      # Available functions are listed in learning_rate._LEARNING_RATE_FUNCTIONS
      # e.g. "constant*linear_warmup*rsqrt_decay*rsqrt_hidden_size"
      learning_rate_schedule="legacy",
      learning_rate_constant=1.0,
      # If learning_rate_schedule=="legacy",
      # then we specify decay scheme here.  Warmup is always exponential,
      # except with "noam" learning rate decay scheme.
      # see optimize.legacy_learning_rate_schedule()
      # TODO(noam): migrate everyone away from this.
      learning_rate_decay_scheme="none",
      # decay_steps and decay_staircase for learning_rate_decay_scheme=="exp"
      learning_rate_decay_steps=5000,
      learning_rate_decay_staircase=False,
      learning_rate_minimum=None,
      learning_rate_decay_rate=1.0,
      learning_rate_warmup_steps=100,
      learning_rate_cosine_cycle_steps=250000,
      learning_rate=0.1,
      multiply_embedding_mode="sqrt_depth",
      # Sequences of operations to perform on layer input and layer output.
      # Used by common_layers.layer_preprocess, common_layers.layer_postprocess
      # Each character represents an operation:
      # none: no preprocessing
      #    d: apply dropout
      #    n: apply normalization (see norm_type and norm_epsilon)
      #    a: add layer input (residual connection - only during postprocess)
      # The special string "none" is used instead of the empty string
      # to indicate no pre/postprocessing, since the empty string causes
      # trouble for hyperparameter tuning.
      # TODO(noam): The current settings ("", "dan") are the published version
      # of the transformer.  ("n", "da") seems better for harder-to-learn
      # models, so it should probably be the default.
      layer_preprocess_sequence="none",
      layer_postprocess_sequence="dan",
      # dropout rate to use during layer_preprocess and layer_postprocess
      layer_prepostprocess_dropout=0.1,
      # broadcast dimensions for layer_prepostprocess_dropout
      # a comma-separated list of integers.
      # see common_layers.dropout_with_broadcast_dims()
      # Change this to "1" to save memory.
      layer_prepostprocess_dropout_broadcast_dims="",
      # dropout some symbols (set them to 0) before embedding.
      symbol_dropout=0.0,
      # What type of normalization to use
      norm_type="layer",  # "batch", layer", "noam", "none".
      # epsilon parameter to normalization function
      norm_epsilon=1e-6,
      symbol_modality_num_shards=1,
      # pad vocabularies so that this value divides the vocabulary size.
      vocab_divisor=1,
      # During training, we drop sequences whose inputs and targets are shorter
      # than min_length
      min_length=0,
      # During training, we drop sequences whose inputs or targets are longer
      # than max_length.
      # If max_length==0, we use hparams.batch_size instead.
      max_length=0,
      # Maximum length in the smallest length bucket.  Setting this
      # flag too high will result in wasteful padding of short
      # sequences.  Due to some (hopefully) temporary hacks in the
      # data reading and batching code, setting this flag too low
      # results in a very long batch-shuffling queue.
      # TODO(noam): change this once the Datasets API changes.
      min_length_bucket=8,
      # This flag controls the number of length buckets in the data
      # reader.  The buckets have maximum lengths from
      # min_bucket_length to (max_length or batch_size), increasing
      # (approximately) by factors of length_bucket_step.
      length_bucket_step=1.1,
      # If set to True, drop sequences longer than max_length during eval.
      # This affects the validity of the evaluation metrics.
      eval_drop_long_sequences=False,
      # TODO(lukaszkaiser): these parameters should probably be set elsewhere.
      # (SymbolModality) - If this flag is on, we try to share all of the input
      # embeddings, the target embeddings and the softmax weights.
      shared_embedding_and_softmax_weights=False,
      # (SymbolModality) - If this flag is on, we try to share the input
      # embeddings and the target embeddings.
      # You can also share the input embeddings with the target embeddings
      # by using a problem_hparams that uses the same modality object for
      # the input modality and target modality.
      shared_embedding=False,
      # In SymbolModality, skip the top layer, assume we're providing logits.
      symbol_modality_skip_top=False,
      # Modalities used to map from features to a space compatible with
      # chosen model architecture. It comprises key-value pairs of a feature
      # name (str) and its modality class.
      modality={},
      # The maximum length of "input" sequence.
      # Sequences longer than this value will be truncated. 0 or negative values
      # mean there is no maximum or truncation.
      # You can change this behavior by overriding preprocess_example() method
      # in your problem class.
      max_input_seq_length=0,
      # The maximum length of "target" sequence.
      # Sequences longer than this value will be truncated. 0 or negative values
      # mean there is no maximum or truncation.
      # You can change this behavior by overriding preprocess_example() method
      # in your problem class.
      max_target_seq_length=0,
      # If True in PREDICT mode, then last-position-only optimizations are not
      # used.
      force_full_predict=False,
      # dtype used for activations. - "float32" or "bfloat16"
      # activation_dtype="bfloat16" currently only works on TPU.
      #    It lowers activation-memory usage
      #    and does not appear to affect quality.
      #    You can train on TPU with activation_dtype="bfloat16" and evaluate
      #    on CPU/GPU with activation_dtype="float32"
      activation_dtype="float32",
      # dtype used for parameters: "float32" or "bfloat16"
      # bfloat16 currently only works with optimizer="adafactor".
      #   The savings in memory allow for training larger models.
      #   Weights are encoded as (w*128)^8, using pseudostochastic
      #   roundoff.  Initial experiments show that model quality is similar
      #   to baseline for about 3M training steps, but worse thereafter.
      weight_dtype="float32",
      # If enable the host_call which is executed every training step.
      # There could be a performance drop if host_call function is slow and
      # cannot keep up with the TPU-side computation.
      tpu_enable_host_call=False,
      # Pad batch dim of inputs to nearest multiple of batch multiple.
      pad_batch=False,
      # Use bfloat16 for gradients cross-replica-sum on TPUs. Setting this flag
      # to true may hurt the model quality, especially on larger slices.
      bfloat16_grads_all_reduce=False,
      # Write tf.summary.
      write_summary=False,
  )


class RangedHParams(object):
  """Defines parameter ranges for tuning."""

  # From ParameterConfig proto
  LINEAR_SCALE = 1
  LOG_SCALE = 2
  REVERSE_LOG_SCALE = 3

  SCALES_STR = {
      LINEAR_SCALE: "UNIT_LINEAR_SCALE",
      LOG_SCALE: "UNIT_LOG_SCALE",
      REVERSE_LOG_SCALE: "UNIT_REVERSE_LOG_SCALE",
  }

  def __init__(self):
    self._categorical_params = {}
    self._discrete_params = {}
    self._float_params = {}
    self._int_params = {}

  def _check_reset_and_type_change(self, name, orig_ctr):
    """Check if name is in orig_ctr or in one of the other type containers."""
    # Resetting a hyperparameter
    if name in orig_ctr:
      tf.logging.warning("Overwriting hparam %s", name)

    ctr_names = [
        (self._categorical_params, "categorical"),
        (self._discrete_params, "discrete"),
        (self._float_params, "float"),
        (self._int_params, "int"),
    ]
    ctrs, names = list(zip(*ctr_names))
    orig_name = names[ctrs.index(orig_ctr)]

    for ctr, ctr_name in ctr_names:
      if ctr is orig_ctr:
        continue

      # Using a different type for the same hyperparameter name
      if name in ctr:
        raise ValueError("Setting hyperparameter %s as type %s, but a "
                         "hyperparemeter of the same name was originally "
                         "registered as type %s" % (name, ctr_name, orig_name))

  def set_categorical(self, name, categories, length=None):
    self._check_reset_and_type_change(name, self._categorical_params)
    self._categorical_params[name] = (name, categories, length)

  def set_discrete(self, name, feasible_points, scale=None, length=None):
    self._check_reset_and_type_change(name, self._discrete_params)
    self._discrete_params[name] = (name, feasible_points, scale, length)

  def set_float(self, name, min_val, max_val, scale=None, length=None):
    self._check_reset_and_type_change(name, self._float_params)
    self._float_params[name] = (name, min_val, max_val, scale, length)

  def set_int(self, name, min_val, max_val, scale=None, length=None):
    self._check_reset_and_type_change(name, self._int_params)
    self._int_params[name] = (name, min_val, max_val, scale, length)

  def fix_select_params(self, hp):
    ctrs = [
        self._categorical_params, self._discrete_params, self._float_params,
        self._int_params
    ]
    for key, val in hp.values().iteritems():
      for ctr in ctrs:
        if key in ctr:
          del ctr[key]
      self.set_discrete(key, [val])

  def to_parameter_specs(self, name_prefix=""):
    """To list of dicts suitable for Cloud ML Engine hyperparameter tuning."""
    specs = []
    for name, categories, _ in self._categorical_params.values():
      spec = {
          "parameterName": name_prefix + name,
          "type": "CATEGORICAL",
          "categoricalValues": categories,
      }
      specs.append(spec)

    for name, feasible_points, scale, _ in self._discrete_params.values():
      spec = {
          "parameterName": name_prefix + name,
          "type": "DISCRETE",
          "discreteValues": feasible_points,
      }
      if scale:
        spec["scaleType"] = self.SCALES_STR[scale]
      specs.append(spec)

    for name, min_val, max_val, scale, _ in self._float_params.values():
      spec = {
          "parameterName": name_prefix + name,
          "type": "DOUBLE",
          "minValue": min_val,
          "maxValue": max_val,
      }
      if scale:
        spec["scaleType"] = self.SCALES_STR[scale]
      specs.append(spec)

    for name, min_val, max_val, scale, _ in self._int_params.values():
      spec = {
          "parameterName": name_prefix + name,
          "type": "INTEGER",
          "minValue": min_val,
          "maxValue": max_val,
      }
      if scale:
        spec["scaleType"] = self.SCALES_STR[scale]
      specs.append(spec)

    return specs


@registry.register_ranged_hparams("basic1")
def basic_range1(ranged_hparams):
  """A basic range of hyperparameters."""
  rhp = ranged_hparams
  rhp.set_discrete("batch_size", [256, 512, 1024, 2048])
  rhp.set_float("dropout", 0.0, 0.3)
  rhp.set_float("learning_rate_constant", 0.005, 8.0, scale=rhp.LOG_SCALE)
  rhp.set_int("learning_rate_warmup_steps", 500, 2000)
  rhp.set_categorical("initializer",
                      ["uniform", "orthogonal", "uniform_unit_scaling"])
  rhp.set_float("initializer_gain", 0.5, 3.5)
  rhp.set_categorical("learning_rate_decay_scheme",
                      ["none", "sqrt", "noam", "exp"])
  rhp.set_float("optimizer_adam_epsilon", 1e-7, 1e-2, scale=rhp.LOG_SCALE)
  rhp.set_float("optimizer_adam_beta1", 0.8, 0.9)
  rhp.set_float("optimizer_adam_beta2", 0.995, 0.999)
  rhp.set_categorical(
      "optimizer",
      ["Adam", "SM3"])

