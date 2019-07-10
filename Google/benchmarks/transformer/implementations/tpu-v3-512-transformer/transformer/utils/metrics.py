"""Utils for metrics used in eval."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from layers import common_layers


class Metrics(object):
  """Available evaluation metrics."""
  # Entries here should match the keys in METRICS_FNS below
  ACC = "accuracy"
  ACC_TOP5 = "accuracy_top5"
  ACC_PER_SEQ = "accuracy_per_sequence"
  NEG_LOG_PERPLEXITY = "neg_log_perplexity"


def padded_accuracy_topk(predictions,
                         labels,
                         k,
                         weights_fn=common_layers.weights_nonzero):
  """Percentage of times that top-k predictions matches labels on non-0s."""
  with tf.variable_scope("padded_accuracy_topk", values=[predictions, labels]):
    padded_predictions, padded_labels = common_layers.pad_with_zeros(
        predictions, labels)
    weights = weights_fn(padded_labels)
    effective_k = tf.minimum(k,
                             common_layers.shape_list(padded_predictions)[-1])
    _, outputs = tf.nn.top_k(padded_predictions, k=effective_k)
    outputs = tf.to_int32(outputs)
    padded_labels = tf.to_int32(padded_labels)
    padded_labels = tf.expand_dims(padded_labels, axis=-1)
    padded_labels += tf.zeros_like(outputs)  # Pad to same shape.
    same = tf.to_float(tf.equal(outputs, padded_labels))
    same_topk = tf.reduce_sum(same, axis=-1)
    return same_topk, weights


def padded_accuracy_top5(predictions,
                         labels,
                         weights_fn=common_layers.weights_nonzero):
  return padded_accuracy_topk(predictions, labels, 5, weights_fn)


def rounding_sequence_accuracy(predictions,
                               labels,
                               weights_fn=common_layers.weights_nonzero):
  """Sequence accuracy for L1/L2 losses: round down the predictions to ints."""
  outputs = tf.squeeze(tf.to_int32(predictions), axis=-1)
  weights = weights_fn(labels)
  labels = tf.to_int32(labels)
  not_correct = tf.to_float(tf.not_equal(outputs, labels)) * weights
  axis = list(range(1, len(outputs.get_shape())))
  correct_seq = 1.0 - tf.minimum(1.0, tf.reduce_sum(not_correct, axis=axis))
  return correct_seq, tf.constant(1.0)


def padded_sequence_accuracy(predictions,
                             labels,
                             weights_fn=common_layers.weights_nonzero):
  """Percentage of times that predictions matches labels everywhere (non-0)."""
  # If the last dimension is 1 then we're using L1/L2 loss.
  if common_layers.shape_list(predictions)[-1] == 1:
    return rounding_sequence_accuracy(
        predictions, labels, weights_fn=weights_fn)
  with tf.variable_scope(
      "padded_sequence_accuracy", values=[predictions, labels]):
    padded_predictions, padded_labels = common_layers.pad_with_zeros(
        predictions, labels)
    weights = weights_fn(padded_labels)

    # Flatten, keeping batch dim (and num_classes dim for predictions)
    # TPU argmax can only deal with a limited number of dimensions
    predictions_shape = common_layers.shape_list(padded_predictions)
    batch_size = predictions_shape[0]
    num_classes = predictions_shape[-1]
    flat_size = common_layers.list_product(
        common_layers.shape_list(padded_labels)[1:])
    padded_predictions = tf.reshape(
        padded_predictions,
        [batch_size, common_layers.list_product(predictions_shape[1:-1]),
         num_classes])
    padded_labels = tf.reshape(padded_labels, [batch_size, flat_size])
    weights = tf.reshape(weights, [batch_size, flat_size])

    outputs = tf.to_int32(tf.argmax(padded_predictions, axis=-1))
    padded_labels = tf.to_int32(padded_labels)
    not_correct = tf.to_float(tf.not_equal(outputs, padded_labels)) * weights
    axis = list(range(1, len(outputs.get_shape())))
    correct_seq = 1.0 - tf.minimum(1.0, tf.reduce_sum(not_correct, axis=axis))
    return correct_seq, tf.constant(1.0)


def padded_neg_log_perplexity(predictions,
                              labels,
                              weights_fn=common_layers.weights_nonzero):
  """Average log-perplexity exluding padding 0s. No smoothing."""
  num, den = common_layers.padded_cross_entropy(
      predictions, labels, 0.0, weights_fn=weights_fn, reduce_sum=False)
  return (-num, den)


def rounding_accuracy(predictions,
                      labels,
                      weights_fn=common_layers.weights_nonzero):
  """Rounding accuracy for L1/L2 losses: round down the predictions to ints."""
  outputs = tf.squeeze(tf.to_int32(predictions))
  labels = tf.squeeze(labels)
  weights = weights_fn(labels)
  labels = tf.to_int32(labels)
  return tf.to_float(tf.equal(outputs, labels)), weights


def padded_accuracy(predictions,
                    labels,
                    weights_fn=common_layers.weights_nonzero):
  """Percentage of times that predictions matches labels on non-0s."""
  # If the last dimension is 1 then we're using L1/L2 loss.
  if common_layers.shape_list(predictions)[-1] == 1:
    return rounding_accuracy(predictions, labels, weights_fn=weights_fn)
  with tf.variable_scope("padded_accuracy", values=[predictions, labels]):
    padded_predictions, padded_labels = common_layers.pad_with_zeros(
        predictions, labels)
    weights = weights_fn(padded_labels)
    outputs = tf.to_int32(tf.argmax(padded_predictions, axis=-1))
    padded_labels = tf.to_int32(padded_labels)
    return tf.to_float(tf.equal(outputs, padded_labels)), weights


# Metrics are functions that take predictions and labels and return
# a tensor of metrics and a tensor of weights.
# If the function has "features" as an argument, it will receive the whole
# features dict as well.
# The results are passed to tf.metrics.mean to accumulate properly.
METRICS_FNS = {
    Metrics.ACC: padded_accuracy,
    Metrics.ACC_TOP5: padded_accuracy_top5,
    Metrics.ACC_PER_SEQ: padded_sequence_accuracy,
    Metrics.NEG_LOG_PERPLEXITY: padded_neg_log_perplexity,
}
