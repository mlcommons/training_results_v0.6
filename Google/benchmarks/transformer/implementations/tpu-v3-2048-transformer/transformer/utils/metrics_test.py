"""Tests for t2t_transformer.tensor2tensor.utils.metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from utils import metrics


class MetricsTest(tf.test.TestCase):

  def testAccuracyMetric(self):
    predictions = np.random.randint(1, 5, size=(12, 12, 12, 1))
    targets = np.random.randint(1, 5, size=(12, 12, 12, 1))
    expected = np.mean((predictions == targets).astype(float))
    with self.test_session() as session:
      scores, _ = metrics.padded_accuracy(
          tf.one_hot(predictions, depth=5, dtype=tf.float32),
          tf.constant(targets, dtype=tf.int32))
      a = tf.reduce_mean(scores)
      session.run(tf.global_variables_initializer())
      actual = session.run(a)
    self.assertAlmostEqual(actual, expected)

  def testAccuracyTopKMetric(self):
    predictions = np.random.randint(1, 5, size=(12, 12, 12, 1))
    targets = np.random.randint(1, 5, size=(12, 12, 12, 1))
    expected = np.mean((predictions == targets).astype(float))
    with self.test_session() as session:
      predicted = tf.one_hot(predictions, depth=5, dtype=tf.float32)
      scores1, _ = metrics.padded_accuracy_topk(
          predicted, tf.constant(targets, dtype=tf.int32), k=1)
      scores2, _ = metrics.padded_accuracy_topk(
          predicted, tf.constant(targets, dtype=tf.int32), k=7)
      a1 = tf.reduce_mean(scores1)
      a2 = tf.reduce_mean(scores2)
      session.run(tf.global_variables_initializer())
      actual1, actual2 = session.run([a1, a2])
    self.assertAlmostEqual(actual1, expected)
    self.assertAlmostEqual(actual2, 1.0)

  def testSequenceAccuracyMetric(self):
    predictions = np.random.randint(4, size=(12, 12, 12, 1))
    targets = np.random.randint(4, size=(12, 12, 12, 1))
    expected = np.mean(
        np.prod((predictions == targets).astype(float), axis=(1, 2)))
    with self.test_session() as session:
      scores, _ = metrics.padded_sequence_accuracy(
          tf.one_hot(predictions, depth=4, dtype=tf.float32),
          tf.constant(targets, dtype=tf.int32))
      a = tf.reduce_mean(scores)
      session.run(tf.global_variables_initializer())
      actual = session.run(a)
    self.assertEqual(actual, expected)

  def testNegativeLogPerplexity(self):
    predictions = np.random.randint(4, size=(12, 12, 12, 1))
    targets = np.random.randint(4, size=(12, 12, 12, 1))
    with self.test_session() as session:
      scores, _ = metrics.padded_neg_log_perplexity(
          tf.one_hot(predictions, depth=4, dtype=tf.float32),
          tf.constant(targets, dtype=tf.int32))
      a = tf.reduce_mean(scores)
      session.run(tf.global_variables_initializer())
      actual = session.run(a)
    self.assertEqual(actual.shape, ())


if __name__ == '__main__':
  tf.test.main()
