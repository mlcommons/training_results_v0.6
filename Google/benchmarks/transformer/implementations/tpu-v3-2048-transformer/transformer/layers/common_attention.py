"""Utilities for attention."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import math
import operator

import numpy as np

from six.moves import range  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin

import tensorflow as tf

from tensorflow.python.framework import function
from layers import common_layers

# Struct containing the sequences ids and order on a batch (are send to the
# expert to allow them to compute the bias mask)
BatchInfo = collections.namedtuple("BatchInfo", "coordinates, order")

_expert_count = 0


def get_timing_signal_1d(length,
                         channels,
                         min_timescale=1.0,
                         max_timescale=1.0e4,
                         start_index=0):
  """Gets a bunch of sinusoids of different frequencies.

  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase.

  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.

  The use of relative position is possible because sin(x+y) and cos(x+y) can be
  expressed in terms of y, sin(x) and cos(x).

  In particular, we use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels / 2. For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.

  Args:
    length: scalar, length of timing signal sequence.
    channels: scalar, size of timing embeddings to create. The number of
      different timescales is equal to channels / 2.
    min_timescale: a float
    max_timescale: a float
    start_index: index of first position

  Returns:
    a Tensor of timing signals [1, length, channels]
  """
  position = tf.to_float(tf.range(length) + start_index)
  num_timescales = channels // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      tf.maximum(tf.to_float(num_timescales) - 1, 1))
  inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
  signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
  signal = tf.reshape(signal, [1, length, channels])
  return signal


def add_timing_signal_1d(x,
                         min_timescale=1.0,
                         max_timescale=1.0e4,
                         start_index=0):
  """Adds a bunch of sinusoids of different frequencies to a Tensor.

  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase.

  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.

  The use of relative position is possible because sin(x+y) and cos(x+y) can be
  experessed in terms of y, sin(x) and cos(x).

  In particular, we use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels / 2. For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.

  Args:
    x: a Tensor with shape [batch, length, channels]
    min_timescale: a float
    max_timescale: a float
    start_index: index of first position

  Returns:
    a Tensor the same shape as x.
  """
  length = common_layers.shape_list(x)[1]
  channels = common_layers.shape_list(x)[2]
  signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale,
                                start_index)
  signal = common_layers.cast_like(signal, x)
  return x + signal


def add_timing_signal_1d_given_position(x,
                                        position,
                                        min_timescale=1.0,
                                        max_timescale=1.0e4):
  """Adds sinusoids of diff frequencies to a Tensor, with timing position given.

  Args:
    x: a Tensor with shape [batch, length, channels]
    position: a Tensor with shape [batch, length]
    min_timescale: a float
    max_timescale: a float

  Returns:
    a Tensor the same shape as x.
  """
  channels = common_layers.shape_list(x)[2]
  num_timescales = channels // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (tf.to_float(num_timescales) - 1))
  inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  scaled_time = (
      tf.expand_dims(tf.to_float(position), 2) *
      tf.expand_dims(tf.expand_dims(inv_timescales, 0), 0))
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=2)
  signal = tf.pad(signal, [[0, 0], [0, 0], [0, tf.mod(channels, 2)]])
  signal = common_layers.cast_like(signal, x)
  return x + signal


def embedding_to_padding(emb):
  """Calculates the padding mask based on which embeddings are all zero.

  We have hacked symbol_modality to return all-zero embeddings for padding.

  Args:
    emb: a Tensor with shape [..., depth].

  Returns:
    a float Tensor with shape [...]. Each element is 1 if its corresponding
    embedding vector is all zero, and is 0 otherwise.
  """
  emb_sum = tf.reduce_sum(tf.abs(emb), axis=-1)
  return tf.to_float(tf.equal(emb_sum, 0.0))


def padding_to_length(padding):
  """Calculate the length of mask based on padding.

  Args:
    padding: a Tensor with shape [..., length].

  Returns:
    a Tensor with shape [...].
  """
  non_padding = 1.0 - padding
  return tf.to_int32(tf.reduce_sum(non_padding, axis=-1))


def attention_bias_local(length, max_backward, max_forward):
  """Create an bias tensor to be added to attention logits.

  A position may attend to positions at most max_distance from it,
  forward and backwards.

  This does not actually save any computation.

  Args:
    length: int
    max_backward: int, maximum distance backward to attend. Negative values
      indicate unlimited.
    max_forward: int, maximum distance forward to attend. Negative values
      indicate unlimited.

  Returns:
    a `Tensor` with shape [1, 1, length, length].
  """
  band = common_layers.ones_matrix_band_part(
      length,
      length,
      max_backward,
      max_forward,
      out_shape=[1, 1, length, length])
  return -1e9 * (1.0 - band)


def attention_bias_lower_triangle(length):
  """Create an bias tensor to be added to attention logits.

  Allows a query to attend to all positions up to and including its own.

  Args:
   length: a Scalar.

  Returns:
    a `Tensor` with shape [1, 1, length, length].
  """
  return attention_bias_local(length, -1, 0)


def attention_bias_same_segment(query_segment_id, memory_segment_id):
  """Create an bias tensor to be added to attention logits.

  Positions with the same segment_ids can see each other.

  Args:
    query_segment_id: a float `Tensor` with shape [batch, query_length].
    memory_segment_id: a float `Tensor` with shape [batch, memory_length].

  Returns:
    a `Tensor` with shape [batch, 1, query_length, memory_length].
  """
  ret = tf.to_float(
      tf.not_equal(
          tf.expand_dims(query_segment_id, 2),
          tf.expand_dims(memory_segment_id, 1))) * -1e9
  return tf.expand_dims(ret, axis=1)


def attention_bias_ignore_padding(memory_padding):
  """Create an bias tensor to be added to attention logits.

  Args:
    memory_padding: a float `Tensor` with shape [batch, memory_length].

  Returns:
    a `Tensor` with shape [batch, 1, 1, memory_length].
  """
  ret = memory_padding * -1e9
  return tf.expand_dims(tf.expand_dims(ret, axis=1), axis=1)


def attention_bias_to_padding(attention_bias):
  """Inverse of attention_bias_ignore_padding().

  Args:
    attention_bias: a `Tensor` with shape [batch, 1, 1, memory_length], as
      returned by attention_bias_ignore_padding().

  Returns:
    a Tensor with shape [batch, memory_length] with 1.0 in padding positions
    and 0.0 in non-padding positions.
  """
  # `attention_bias` is a large negative number in padding positions and 0.0
  # elsewhere.
  return tf.squeeze(tf.to_float(tf.less(attention_bias, -1)), axis=[1, 2])


def split_last_dimension(x, n):
  """Reshape x so that the last dimension becomes two dimensions.

  The first of these two dimensions is n.

  Args:
    x: a Tensor with shape [..., m]
    n: an integer.

  Returns:
    a Tensor with shape [..., n, m/n]
  """
  x_shape = common_layers.shape_list(x)
  m = x_shape[-1]
  if isinstance(m, int) and isinstance(n, int):
    assert m % n == 0
  return tf.reshape(x, x_shape[:-1] + [n, m // n])


def combine_last_two_dimensions(x):
  """Reshape x so that the last two dimension become one.

  Args:
    x: a Tensor with shape [..., a, b]

  Returns:
    a Tensor with shape [..., ab]
  """
  x_shape = common_layers.shape_list(x)
  a, b = x_shape[-2:]
  return tf.reshape(x, x_shape[:-2] + [a * b])


def split_heads(x, num_heads):
  """Split channels (dimension 2) into multiple heads (becomes dimension 1).

  Args:
    x: a Tensor with shape [batch, length, channels]
    num_heads: an integer

  Returns:
    a Tensor with shape [batch, num_heads, length, channels / num_heads]
  """
  return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])


def combine_heads(x):
  """Inverse of split_heads.

  Args:
    x: a Tensor with shape [batch, num_heads, length, channels / num_heads]

  Returns:
    a Tensor with shape [batch, length, channels]
  """
  return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))


def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0,
                          name=None,
                          dropout_broadcast_dims=None):
  """Dot-product attention.

  Args:
    q: Tensor with shape [..., length_q, depth_k].
    k: Tensor with shape [..., length_kv, depth_k]. Leading dimensions must
      match with q.
    v: Tensor with shape [..., length_kv, depth_v] Leading dimensions must match
      with q.
    bias: bias Tensor (see attention_bias())
    dropout_rate: a float.
    name: an optional string
    dropout_broadcast_dims: an optional list of integers less than rank of q.
      Specifies in which dimensions to broadcast the dropout decisions.

  Returns:
    Tensor with shape [..., length_q, depth_v].
  """

  with tf.variable_scope(
      name, default_name="dot_product_attention", values=[q, k, v]) as scope:
    logits = tf.einsum("bqhf,bkhf->bhqk", q, k)
    if bias is not None:
      bias = common_layers.cast_like(bias, logits)
      logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    # Drop out attention links for each head.
    weights = common_layers.dropout_with_broadcast_dims(
        weights, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)
    return tf.einsum("bhqk,bkhf->bqhf", weights, v)


def compute_attention_component(antecedent, total_depth, heads, name="c"):
  """Computes attention compoenent (query, key or value).

  Args:
    antecedent: a Tensor with shape [batch, length, channels]
    total_depth: an integer
    heads: number of attention heads
    name: a string specifying scope name.

  Returns:
    c : [batch, length, num_heads, depth / num_heads]
  """
  return common_layers.dense(
      antecedent,
      total_depth,
      heads,
      use_bias=False,
      name=name,
      reuse=tf.AUTO_REUSE)


def compute_qkv(query_antecedent, memory_antecedent, total_key_depth,
                total_value_depth, heads):
  """Computes query, key and value.

  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels]
    total_key_depth: an integer
    total_value_depth: an integer
    heads: number of attention head

  Returns:
    q, k, v: [batch, length, num_heads, depth / num_heads] tensors
  """
  if memory_antecedent is None:
    memory_antecedent = query_antecedent
  q = compute_attention_component(query_antecedent, total_key_depth, heads, "q")
  k = compute_attention_component(memory_antecedent, total_key_depth, heads,
                                  "k")
  v = compute_attention_component(memory_antecedent, total_value_depth, heads,
                                  "v")
  return q, k, v


def multihead_attention(query_antecedent,
                        memory_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        block_length=128,
                        block_width=128,
                        cache=None,
                        kv_encdecs=None,
                        name="multihead_attention",
                        dropout_broadcast_dims=None,
                        **kwargs):
  """Multihead scaled-dot-product attention with input/output transformations.

  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels] or None
    bias: bias Tensor (see attention_bias())
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    dropout_rate: a floating point number
    block_length: an integer - relevant for "local_mask_right"
    block_width: an integer - relevant for "local_unmasked"
    cache: A dict, containing Tensors which are the results of previous
      attentions, used for fast decoding. Expects the dict to contrain two keys
      ('k' and 'v'), for the initial call the values for these keys should be
      empty Tensors of the appropriate shape.
      'k': [batch_size, 0, key_channels];
      'v': [batch_size, 0, value_channels].
    kv_encdecs: A dict, representing the key and value for encoder-decoder
      attention used by decoding (inference).
    name: an optional string.
    dropout_broadcast_dims:  an optional list of integers less than 4 specifying
      in which dimensions to broadcast the dropout decisions. saves memory.
    **kwargs (dict): Parameters for the attention function
  Caching:
    WARNING: For decoder self-attention, i.e. when memory_antecedent == None,
      the caching assumes that the bias contains future masking.  The caching
      works by saving all the previous key and value values so that you are able
      to send just the last query location to this attention function. I.e. if
      the cache dict is provided it assumes the query is of the shape
      [batch_size, 1, hidden_dim] rather than the full memory.

  Returns:
    The result of the attention transformation. The output shape is
        [batch_size, length_q, hidden_dim]
    unless the cache dict is provided in which case only the last memory
    position is calculated and the output shape is [batch_size, 1, hidden_dim]

  Raises:
    ValueError: if the key depth or value depth are not divisible by the
      number of attention heads.
  """
  if total_key_depth % num_heads != 0:
    raise ValueError("Key depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_key_depth, num_heads))
  if total_value_depth % num_heads != 0:
    raise ValueError("Value depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_value_depth, num_heads))
  with tf.variable_scope(
      name,
      default_name="multihead_attention",
      values=[query_antecedent, memory_antecedent]):

    if cache is None or memory_antecedent is None:
      q, k, v = compute_qkv(query_antecedent, memory_antecedent,
                            total_key_depth, total_value_depth, num_heads)
    if cache is not None:
      if bias is None:
        raise ValueError("Bias required for caching. See function docstring "
                         "for details.")

      if memory_antecedent is not None:
        # Encoder-Decoder Attention Cache
        q = compute_attention_component(query_antecedent, total_key_depth,
                                        num_heads, "q")
        k = kv_encdecs["k_encdec"]
        v = kv_encdecs["v_encdec"]
      else:
        decode_loop_step = kwargs.get("decode_loop_step")
        # Updating the tensor by adding the result of matmul(one_hot,
        # update_in_current_step). As inplace_ops only supports inplace_update
        # on the first dimension. This implementation is faster than the
        # previous version due to the elimination of expensive transpose ops.
        s = common_layers.shape_list(cache["k"])
        indices = tf.reshape(
            tf.one_hot(decode_loop_step, s[3], dtype=k.dtype), [1, 1, 1, s[3]])
        k = tf.transpose(k, [0, 2, 3, 1])
        cache["k"] = cache["k"] + k * indices
        k = tf.transpose(cache["k"], [0, 3, 1, 2])
        s = common_layers.shape_list(cache["v"])
        indices = tf.reshape(
            tf.one_hot(decode_loop_step, s[3], dtype=k.dtype), [1, 1, 1, s[3]])
        v = tf.transpose(v, [0, 2, 3, 1])
        cache["v"] = cache["v"] + v * indices
        v = tf.transpose(cache["v"], [0, 3, 1, 2])

    key_depth_per_head = total_key_depth // num_heads
    q *= key_depth_per_head**-0.5

    x = dot_product_attention(
        q,
        k,
        v,
        bias,
        dropout_rate,
        dropout_broadcast_dims=dropout_broadcast_dims)

    x = common_layers.dense(
        x,
        output_depth,
        num_heads,
        use_bias=False,
        name="output_transform",
        reuse=tf.AUTO_REUSE)
    return x
