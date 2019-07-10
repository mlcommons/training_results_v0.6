"""Transformer model from "Attention Is All You Need".

The Transformer model consists of an encoder and a decoder. Both are stacks
of self-attention layers followed by feed-forward layers. This model yields
good results on a number of problems, especially in NLP and machine translation.

See "Attention Is All You Need" (https://arxiv.org/abs/1706.03762) for the full
description of the model and the results obtained with its early version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from six.moves import range  # pylint: disable=redefined-builtin

import tensorflow as tf

from tensorflow.python.ops import inplace_ops
from tensorflow.python.util import nest
from layers import common_attention
from layers import common_hparams
from layers import common_layers
from layers import transformer_layers
from utils import beam_search
from utils import registry
from utils import t2t_model


# Alias some commonly reused layers, here and elsewhere.
transformer_prepare_encoder = transformer_layers.transformer_prepare_encoder
transformer_encoder = transformer_layers.transformer_encoder
transformer_ffn_layer = transformer_layers.transformer_ffn_layer


@registry.register_model
class Transformer(t2t_model.T2TModel):
  """Attention net.  See file docstring."""

  def __init__(self, *args, **kwargs):
    super(Transformer, self).__init__(*args, **kwargs)

  def encode(self, inputs, target_space, hparams, features=None):
    """Encode transformer inputs.

    Args:
      inputs: Transformer inputs [batch_size, input_length, 1, hidden_dim] which
        will be flattened along the two spatial dimensions.
      target_space: scalar, target space ID.
      hparams: hyperparameters for model.
      features: optionally pass the entire features dictionary as well.
        This is needed now for "packed" datasets.

    Returns:
      Tuple of:
          encoder_output: Encoder representation.
              [batch_size, input_length, hidden_dim]
          encoder_decoder_attention_bias: Bias and mask weights for
              encoder-decoder attention. [batch_size, input_length]
    """
    inputs = common_layers.flatten4d3d(inputs)

    encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
        transformer_prepare_encoder(
            inputs, target_space, hparams, features=features))

    encoder_input = tf.nn.dropout(encoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    encoder_output = transformer_encoder(
        encoder_input,
        self_attention_bias,
        hparams,
        nonpadding=features_to_nonpadding(features, "inputs"))

    return encoder_output, encoder_decoder_attention_bias

  def decode(self,
             decoder_input,
             encoder_output,
             encoder_decoder_attention_bias,
             decoder_self_attention_bias,
             hparams,
             cache=None,
             kv_encdecs=None,
             decode_loop_step=None,
             nonpadding=None):
    """Decode Transformer outputs from encoder representation.

    Args:
      decoder_input: inputs to bottom of the model.
        [batch_size, decoder_length, hidden_dim]
      encoder_output: Encoder representation.
        [batch_size, input_length, hidden_dim]
      encoder_decoder_attention_bias: Bias and mask weights for
        encoder-decoder attention. [batch_size, input_length]
      decoder_self_attention_bias: Bias and mask weights for decoder
        self-attention. [batch_size, decoder_length]
      hparams: hyperparameters for model.
      cache: A dict, containing tensors which are the results of previous
        attentions, used for fast decoding.
      kv_encdecs: A dict, representing the keys and values for encoder-decoder
        attention used by decoding (inference).
      decode_loop_step: An integer, step number of the decoding loop.
        Only used for inference on TPU.
      nonpadding: optional Tensor with shape [batch_size, decoder_length]

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    decoder_input = tf.nn.dropout(decoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    decoder_output = transformer_decoder(
        decoder_input,
        encoder_output,
        decoder_self_attention_bias,
        encoder_decoder_attention_bias,
        hparams,
        cache=cache,
        kv_encdecs=kv_encdecs,
        decode_loop_step=decode_loop_step,
        nonpadding=nonpadding)

    if hparams.mode == tf.estimator.ModeKeys.TRAIN:
      # TPU does not react kindly to extra dimensions.
      # TODO(noam): remove this once TPU is more forgiving of extra dims.
      return decoder_output
    else:
      # Expand since t2t expects 4d tensors.
      return tf.expand_dims(decoder_output, axis=2)

  def body(self, features):
    """Transformer main model_fn.

    Args:
      features: Map of features to the model. Should contain the following:
          "inputs": Transformer inputs.
              [batch_size, input_length, 1, hidden_dim].
          "targets": Target decoder outputs.
              [batch_size, decoder_length, 1, hidden_dim]
          "target_space_id": A scalar int from data_generators.problem.SpaceID.

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    hparams = self._hparams

    if self.has_input:
      inputs = features["inputs"]
      target_space = features["target_space_id"]
      encoder_output, encoder_decoder_attention_bias = self.encode(
          inputs, target_space, hparams, features=features)
    else:
      encoder_output, encoder_decoder_attention_bias = (None, None)

    targets = features["targets"]
    targets_shape = common_layers.shape_list(targets)
    targets = common_layers.flatten4d3d(targets)
    decoder_input, decoder_self_attention_bias = transformer_prepare_decoder(
        targets, hparams, features=features)
    decoder_output = self.decode(
        decoder_input,
        encoder_output,
        encoder_decoder_attention_bias,
        decoder_self_attention_bias,
        hparams,
        nonpadding=features_to_nonpadding(features, "targets"))

    return tf.reshape(decoder_output, targets_shape)

  # TODO(b/124008972): Refactor the code and change tests accordingly to test
  # public API rather than internal methods.
  def _beam_decode(self,
                   features,
                   decode_length,
                   beam_size,
                   top_beams,
                   alpha,
                   use_tpu=False):
    """Beam search decoding.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for longer translations.
      use_tpu: A bool, whether to do beam decode on TPU.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search.
      }
    """
    with tf.variable_scope(self.name):
      return (
          self._fast_decode_tpu(
              features, decode_length, beam_size, top_beams, alpha))

  def _fast_decode_tpu(self,
                       features,
                       decode_length,
                       beam_size,
                       top_beams=1,
                       alpha=1.0):
    """Fast decoding.

    Implements beam search decoding on TPU.

    Args:
      features: A map of string to model features.
      decode_length: An integer, how many additional timesteps to decode.
      beam_size: An integer, number of beams.
      top_beams: An integer, how many of the beams to return.
      alpha: A float that controls the length penalty. Larger the alpha,
        stronger the preference for longer translations.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search.
      }.

    Raises:
      NotImplementedError: If there are multiple data shards or
        beam_size is one.
    """
    if beam_size == 1:
      raise NotImplementedError(
            "Greedy Decoding is not supported in this MLPerf version.")
    if "targets_segmentation" in features:
      raise NotImplementedError(
          "Decoding not supported on packed datasets "
          " If you want to decode from a dataset, use the non-packed version"
          " of the dataset when decoding.")
    hparams = self._hparams
    target_modality = self._problem_hparams.modality["targets"]

    if self.hparams.activation_dtype == "bfloat16":
      for k, v in sorted(six.iteritems(features)):
        if v.dtype == tf.float32:
          features[k] = tf.cast(v, tf.bfloat16)

    if self.has_input:
      inputs = features["inputs"]
      if target_modality.is_class_modality:
        decode_length = 1
      else:
        decode_length = (
            common_layers.shape_list(inputs)[1] + features.get(
                "decode_length", decode_length))

      # TODO(llion): Clean up this reshaping logic.
      inputs = tf.expand_dims(inputs, axis=1)
      if len(inputs.shape) < 5:
        inputs = tf.expand_dims(inputs, axis=4)
      s = common_layers.shape_list(inputs)
      batch_size = s[0]
      inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
      input_modality = self._problem_hparams.modality["inputs"]
      with tf.variable_scope(input_modality.name):
        inputs = input_modality.bottom(inputs)
      if self.hparams.activation_dtype == "bfloat16":
        inputs = tf.cast(inputs, tf.bfloat16)
      with tf.variable_scope("body"):
        encoder_output, encoder_decoder_attention_bias = self.encode(
            inputs,
            features["target_space_id"],
            hparams,
            features=features)
      partial_targets = None
    else:
      # The problem has no inputs.
      encoder_output = None
      encoder_decoder_attention_bias = None

      # Prepare partial targets.
      # In either features["inputs"] or features["targets"].
      # We force the outputs to begin with these sequences.
      partial_targets = features.get("inputs")
      if partial_targets is None:
        partial_targets = features["targets"]
      assert partial_targets is not None
      partial_targets = common_layers.expand_squeeze_to_nd(partial_targets, 2)
      partial_targets = tf.to_int64(partial_targets)
      partial_targets_shape = common_layers.shape_list(partial_targets)
      partial_targets_length = partial_targets_shape[1]
      decode_length = (
          partial_targets_length + features.get("decode_length", decode_length))
      batch_size = partial_targets_shape[0]

    def preprocess_targets(targets, i):
      """Performs preprocessing steps on the targets to prepare for the decoder.

      This includes:
        - Embedding the ids.
        - Flattening to 3D tensor.
        - Optionally adding timing signals.

      Args:
        targets: A tensor, inputs ids to the decoder. [batch_size, 1].
        i: An integer, Step number of the decoding loop.

      Returns:
        A tensor, processed targets [batch_size, 1, hidden_dim].
      """
      with tf.variable_scope(target_modality.name):
        targets = target_modality.targets_bottom(targets)
      if self.hparams.activation_dtype == "bfloat16":
        targets = tf.cast(targets, tf.bfloat16)
      targets = common_layers.flatten4d3d(targets)

      # TODO(llion): Explain! Is this even needed?
      targets = tf.cond(
          tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)

      positional_encoding = common_attention.get_timing_signal_1d(
          decode_length + 1, hparams.hidden_size)
      positional_encoding_shape = positional_encoding.shape.as_list()
      positional_encoding = common_layers.cast_like(
          positional_encoding, targets)
      targets += tf.slice(
          positional_encoding, [0, i, 0],
          [positional_encoding_shape[0], 1, positional_encoding_shape[2]])
      return targets

    decoder_self_attention_bias = (
        common_attention.attention_bias_lower_triangle(decode_length))

    def symbols_to_logits_tpu_fn(ids, i, cache, kv_encdecs):
      """Go from ids to logits for next symbol on TPU.

      Args:
        ids: A tensor, symbol IDs.
        i: An integer, step number of the decoding loop. Only used for inference
          on TPU.
        cache: A dict, containing tensors which are the results of previous
          attentions, used for fast decoding.
        kv_encdecs: A dict, representing the keys and values for encoder-decoder
          attention used by decoding (inference).

      Returns:
        ret: A tensor, computed logits.
        cache: A dict, containing tensors which are the results of previous
            attentions, used for fast decoding.
      """
      ids = ids[:, -1:]
      targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
      targets = preprocess_targets(targets, i)

      bias_shape = decoder_self_attention_bias.shape.as_list()
      bias = tf.slice(decoder_self_attention_bias, [0, 0, i, 0],
                      [bias_shape[0], bias_shape[1], 1, bias_shape[3]])

      # All other states in the cache are batch major to accomendate gather
      # op for permutation.
      tiled_encoder_output = beam_search.merge_beam_dim(
          beam_search.expand_to_beam_size(encoder_output, beam_size))
      tiled_encoder_decoder_attention_bias = beam_search.merge_beam_dim(
          beam_search.expand_to_beam_size(
              encoder_decoder_attention_bias, beam_size))

      with tf.variable_scope("body"):
        body_outputs = self.decode(
            targets,
            tiled_encoder_output,
            tiled_encoder_decoder_attention_bias,
            bias,
            hparams,
            cache,
            kv_encdecs,
            i,
            nonpadding=features_to_nonpadding(features, "targets"))

      with tf.variable_scope(target_modality.name):
        logits = target_modality.top(body_outputs, None)

      ret = tf.squeeze(logits, axis=[1, 2, 3])
      if partial_targets is not None:
        # If the position is within the given partial targets, we alter the
        # logits to always return those values.
        # A faster approach would be to process the partial targets in one
        # iteration in order to fill the corresponding parts of the cache.
        # This would require broader changes, though.
        vocab_size = tf.shape(ret)[1]

        def forced_logits():
          return tf.one_hot(
              tf.tile(
                  tf.slice(partial_targets, [0, i],
                           [partial_targets.shape.as_list()[0], 1]),
                  [beam_size]), vocab_size, 0.0, -1e9)

        ret = tf.cond(
            tf.less(i, partial_targets_length), forced_logits, lambda: ret)
      return ret, cache

    ret = fast_decode_tpu(
        encoder_output=encoder_output,
        symbols_to_logits_fn=symbols_to_logits_tpu_fn,
        hparams=hparams,
        decode_length=decode_length,
        vocab_size=target_modality.top_dimensionality,
        beam_size=beam_size,
        top_beams=top_beams,
        alpha=alpha,
        batch_size=batch_size)
    if partial_targets is not None:
      ret["outputs"] = ret["outputs"][:, :, partial_targets_length:]
    return ret


def fast_decode_tpu(encoder_output,
                    symbols_to_logits_fn,
                    hparams,
                    decode_length,
                    vocab_size,
                    beam_size,
                    top_beams=1,
                    alpha=1.0,
                    sos_id=0,
                    eos_id=beam_search.EOS_ID,
                    batch_size=None,
                    scope_prefix="body/"):
  """Given encoder output and a symbols to logits function, does fast decoding.

  Implements beam search decoding for TPU.

  Args:
    encoder_output: A tensor, output from encoder.
    symbols_to_logits_fn: Incremental decoding, function mapping triple
      `(ids, step, cache)` to symbol logits.
    hparams: Run hyperparameters.
    decode_length: An integer, how many additional timesteps to decode.
    vocab_size: Output vocabulary size.
    beam_size: An integer, number of beams.
    top_beams: An integer, how many of the beams to return.
    alpha: A float that controls the length penalty. Larger the alpha, stronger
      the preference for longer translations.
    sos_id: Start-of-sequence symbol.
    eos_id: End-of-sequence symbol.
    batch_size: An integer, must be passed if there is no input.
    scope_prefix: str, prefix for decoder layer variable scopes.

  Returns:
    A dict of decoding results {
        "outputs": integer `Tensor` of decoded ids of shape
            [batch_size, top_beams, <= decode_length]
        "scores": decoding log probs from the beam search.
    }.

  Raises:
    NotImplementedError: If beam size > 1 with partial targets.
  """
  if encoder_output is not None:
    batch_size = common_layers.shape_list(encoder_output)[0]

  key_channels = hparams.attention_key_channels or hparams.hidden_size
  value_channels = hparams.attention_value_channels or hparams.hidden_size
  num_layers = hparams.num_decoder_layers or hparams.num_hidden_layers

  cache = {
      "layer_%d" % layer: {
          "k":
              tf.zeros([
                  batch_size,
                  hparams.num_heads,
                  key_channels // hparams.num_heads,
                  decode_length
              ], dtype=encoder_output.dtype),
          "v":
              tf.zeros([
                  batch_size,
                  hparams.num_heads,
                  value_channels // hparams.num_heads,
                  decode_length
              ], dtype=encoder_output.dtype),
      } for layer in range(num_layers)
  }

  kv_encdecs = {
      "layer_%d" % layer: {} for layer in range(num_layers)
  }
  if encoder_output is not None:
    for layer in range(num_layers):
      layer_name = "layer_%d" % layer
      with tf.variable_scope(
          "%sdecoder/%s/encdec_attention/multihead_attention" % (scope_prefix,
                                                                 layer_name)):
        k_encdec = common_attention.compute_attention_component(
            encoder_output, key_channels, hparams.num_heads, name="k")
        k_encdec = beam_search.merge_beam_dim(
            beam_search.expand_to_beam_size(k_encdec, beam_size))
        v_encdec = common_attention.compute_attention_component(
            encoder_output, value_channels, hparams.num_heads, name="v")
        v_encdec = beam_search.merge_beam_dim(
            beam_search.expand_to_beam_size(v_encdec, beam_size))
      kv_encdecs[layer_name]["k_encdec"] = k_encdec
      kv_encdecs[layer_name]["v_encdec"] = v_encdec

  initial_ids = sos_id * tf.ones([batch_size], dtype=tf.int32)
  decoded_ids, scores = beam_search.beam_search(
      symbols_to_logits_fn,
      initial_ids,
      beam_size,
      decode_length,
      vocab_size,
      alpha,
      states=cache,
      kv_encdecs=kv_encdecs,
      eos_id=eos_id,
      stop_early=(top_beams == 1))

  if top_beams == 1:
    decoded_ids = decoded_ids[:, 0, 1:]
    scores = scores[:, 0]
  else:
    decoded_ids = decoded_ids[:, :top_beams, 1:]
    scores = scores[:, :top_beams]

  return {"outputs": decoded_ids, "scores": scores}


def features_to_nonpadding(features, inputs_or_targets="inputs"):
  key = inputs_or_targets + "_segmentation"
  if features and key in features:
    return tf.minimum(tf.to_float(features[key]), 1.0)
  return None


def transformer_prepare_decoder(targets, hparams, features=None):
  """Prepare one shard of the model for the decoder.

  Args:
    targets: a Tensor.
    hparams: run hyperparameters
    features: optionally pass the entire features dictionary as well.
      This is needed now for "packed" datasets.

  Returns:
    decoder_input: a Tensor, bottom of decoder stack
    decoder_self_attention_bias: a bias tensor for use in decoder self-attention
  """
  decoder_self_attention_bias = (
      common_attention.attention_bias_lower_triangle(
          common_layers.shape_list(targets)[1]))

  if features and "targets_segmentation" in features:
    # "Packed" dataset - keep the examples from seeing each other.
    targets_segmentation = features["targets_segmentation"]
    targets_position = features["targets_position"]
    decoder_self_attention_bias += common_attention.attention_bias_same_segment(
        targets_segmentation, targets_segmentation)
  else:
    targets_position = None
  decoder_input = common_layers.shift_right_3d(targets)
  if targets_position is not None:
    decoder_input = common_attention.add_timing_signal_1d_given_position(
        decoder_input, targets_position)
  else:
    decoder_input = common_attention.add_timing_signal_1d(decoder_input)

  if hparams.activation_dtype == "bfloat16":
    decoder_self_attention_bias = tf.cast(decoder_self_attention_bias,
                                          tf.bfloat16)
  return (decoder_input, decoder_self_attention_bias)


def transformer_decoder(decoder_input,
                        encoder_output,
                        decoder_self_attention_bias,
                        encoder_decoder_attention_bias,
                        hparams,
                        cache=None,
                        kv_encdecs=None,
                        decode_loop_step=None,
                        name="decoder",
                        nonpadding=None):
  """A stack of transformer layers.

  Args:
    decoder_input: a Tensor
    encoder_output: a Tensor
    decoder_self_attention_bias: bias Tensor for self-attention
      (see common_attention.attention_bias())
    encoder_decoder_attention_bias: bias Tensor for encoder-decoder attention
      (see common_attention.attention_bias())
    hparams: hyperparameters for model
    cache: A dict, containing tensors which are the results of previous
      attentions, used for fast decoding.
    kv_encdecs: A dict, representing the keys and values for encoder-decoder
      attention used by decoding (inference).
    decode_loop_step: An integer, step number of the decoding loop.
      Only used for inference on TPU.
    name: a string
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This is used
      to mask out padding in convolutional layers.  We generally only
      need this mask for "packed" datasets, because for ordinary datasets,
      no padding is ever followed by nonpadding.

  Returns:
    y: a Tensors
  """
  x = decoder_input
  attention_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "attention_dropout_broadcast_dims", "")))

  with tf.variable_scope(name):
    for layer in range(hparams.num_decoder_layers or hparams.num_hidden_layers):
      layer_name = "layer_%d" % layer
      layer_cache = cache[layer_name] if cache is not None else None
      layer_kv_encdecs = (kv_encdecs[layer_name] if kv_encdecs is not None
                          else None)
      with tf.variable_scope(layer_name):
        with tf.variable_scope("self_attention"):
          y = common_attention.multihead_attention(
              common_layers.layer_preprocess(x, hparams),
              None,
              decoder_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,
              cache=layer_cache,
              dropout_broadcast_dims=attention_dropout_broadcast_dims,
              max_length=hparams.get("max_length"),
              decode_loop_step=decode_loop_step)
          x = common_layers.layer_postprocess(x, y, hparams)
        if encoder_output is not None:
          with tf.variable_scope("encdec_attention"):
            y = common_attention.multihead_attention(
                common_layers.layer_preprocess(x, hparams),
                encoder_output,
                encoder_decoder_attention_bias,
                hparams.attention_key_channels or hparams.hidden_size,
                hparams.attention_value_channels or hparams.hidden_size,
                hparams.hidden_size,
                hparams.num_heads,
                hparams.attention_dropout,
                cache=layer_cache,
                kv_encdecs=layer_kv_encdecs,
                dropout_broadcast_dims=attention_dropout_broadcast_dims,
                max_length=hparams.get("max_length"))
            x = common_layers.layer_postprocess(x, y, hparams)
        with tf.variable_scope("ffn"):
          y = transformer_ffn_layer(
              common_layers.layer_preprocess(x, hparams),
              hparams,
              cache=layer_cache)
          x = common_layers.layer_postprocess(x, y, hparams)
    # if normalization is done in layer_preprocess, then it should also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    return common_layers.layer_preprocess(x, hparams)


@registry.register_hparams
def transformer_base_v1():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.norm_type = "layer"
  hparams.hidden_size = 512
  hparams.batch_size = 4096
  hparams.max_length = 256
  hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.learning_rate_schedule = "legacy"
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 4000
  hparams.initializer_gain = 1.0
  hparams.num_hidden_layers = 6
  hparams.initializer = "uniform_unit_scaling"
  hparams.weight_decay = 0.0
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.98
  hparams.num_sampled_classes = 0
  hparams.label_smoothing = 0.1
  hparams.shared_embedding_and_softmax_weights = True

  # Add new ones like this.
  hparams.add_hparam("filter_size", 2048)
  # Layer-related flags. If zero, these fall back on hparams.num_hidden_layers.
  hparams.add_hparam("num_encoder_layers", 0)
  hparams.add_hparam("num_decoder_layers", 0)
  # Attention-related flags.
  hparams.add_hparam("num_heads", 8)
  hparams.add_hparam("attention_key_channels", 0)
  hparams.add_hparam("attention_value_channels", 0)
  hparams.add_hparam("parameter_attention_key_channels", 0)
  hparams.add_hparam("parameter_attention_value_channels", 0)
  # All hyperparameters ending in "dropout" are automatically set to 0.0
  # when not in training mode.
  hparams.add_hparam("attention_dropout", 0.0)
  hparams.add_hparam("attention_dropout_broadcast_dims", "")
  hparams.add_hparam("relu_dropout", 0.0)
  hparams.add_hparam("relu_dropout_broadcast_dims", "")
  hparams.add_hparam("pos", "timing")  # timing, none
  hparams.add_hparam("nbr_decoder_problems", 1)
  hparams.add_hparam("conv_first_kernel", 3)
  hparams.add_hparam("attention_variables_3d", False)
  hparams.add_hparam("use_target_space_embedding", True)
  # If specified, use this value instead of problem name in metrics.py.
  # This is useful for programs that can automatically compare experiments side
  #   by side based on the same metric names.
  hparams.add_hparam("overload_eval_metric_name", "")
  return hparams


@registry.register_hparams
def transformer_base_v2():
  """Set of hyperparameters."""
  hparams = transformer_base_v1()
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.attention_dropout = 0.1
  hparams.relu_dropout = 0.1
  hparams.learning_rate_warmup_steps = 8000
  hparams.learning_rate = 0.2
  return hparams


@registry.register_hparams
def transformer_base_v3():
  """Base parameters for Transformer model."""
  # Update parameters here, then occasionally cut a versioned set, e.g.
  # transformer_base_v2.
  hparams = transformer_base_v2()
  hparams.optimizer_adam_beta2 = 0.997
  # New way of specifying learning rate schedule.
  # Equivalent to previous version.
  hparams.learning_rate_schedule = (
      "constant*linear_warmup*rsqrt_decay*rsqrt_hidden_size")
  hparams.learning_rate_constant = 2.0
  return hparams


@registry.register_hparams
def transformer_base():
  """Base parameters for Transformer model."""
  hparams = transformer_base_v3()
  return hparams


@registry.register_hparams
def transformer_test():
  hparams = transformer_base()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 16
  hparams.filter_size = 8
  hparams.num_heads = 2
  return hparams


@registry.register_hparams
def transformer_small():
  hparams = transformer_base()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 256
  hparams.filter_size = 1024
  hparams.num_heads = 4
  return hparams


@registry.register_hparams
def transformer_mlperf_tpu():
  """HParams for Transformer model on TPU for MLPerf on TPU 2x2."""
  hparams = transformer_base_v3()
  hparams.mlperf_mode = True
  hparams.symbol_modality_num_shards = 1
  hparams.max_length = 256  # ignored when using "_packed" problems
  hparams.batch_size = 2048  # per-chip batch size matches the reference model
  hparams.hidden_size = 1024
  hparams.filter_size = 4096
  hparams.num_heads = 16
  hparams.learning_rate_warmup_steps = 2000
  hparams.pad_batch = True
  hparams.activation_dtype = "bfloat16"
  return hparams


@registry.register_hparams
def transformer_mlperf_tpu_nan_test():
  hparams = transformer_mlperf_tpu()
  hparams.num_hidden_layers = 1
  hparams.hidden_size = 128
  hparams.filter_size = 512
  hparams.num_heads = 4
  return hparams


@registry.register_hparams
def transformer_mlperf_8_heads_tpu():
  hparams = transformer_mlperf_tpu()
  hparams.num_heads = 8
  return hparams


@registry.register_hparams
def transformer_mlperf_tpu_sm3():
  """HParams for Transformer model on TPU for MLPerf with SM3 Optimizer."""
  hparams = transformer_mlperf_tpu()
  hparams.optimizer = "SM3"
  hparams.optimizer_momentum_momentum=0.9
  hparams.learning_rate_schedule = ("constant*linear_warmup")
  hparams.learning_rate_constant = 0.25
  hparams.learning_rate_warmup_steps = 40000
  return hparams


@registry.register_hparams
def transformer_small_tpu():
  hparams = transformer_mlperf_tpu()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 128
  hparams.filter_size = 512
  hparams.num_heads = 4
  return hparams


@registry.register_ranged_hparams
def transformer_mlperf_tpu_base_range(rhp):
  """Small range of hyperparameters."""
  rhp.set_float("learning_rate_constant", 1.0, 8.0)
  rhp.set_discrete("learning_rate_warmup_steps",
                   [400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900,
                    950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400,
                    1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900,
                    1950, 2000])
  rhp.set_float("optimizer_adam_beta1", 0.76, 0.95)
  rhp.set_float("optimizer_adam_beta2", 0.75, 0.998)
