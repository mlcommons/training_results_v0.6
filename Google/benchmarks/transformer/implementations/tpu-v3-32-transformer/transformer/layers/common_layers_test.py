"""Tests for common layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

import tensorflow as tf

from layers import common_layers


class CommonLayersTest(parameterized.TestCase, tf.test.TestCase):

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testFlatten4D3D(self):
    x = np.random.random_integers(1, high=8, size=(3, 5, 2))
    y = common_layers.flatten4d3d(common_layers.embedding(x, 10, 7))
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(y)
    self.assertEqual(res.shape, (3, 5 * 2, 7))

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testEmbedding(self):
    x = np.random.random_integers(1, high=8, size=(3, 5))
    y = common_layers.embedding(x, 10, 16)
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(y)
    self.assertEqual(res.shape, (3, 5, 16))

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testLayerNorm(self):
    x = np.random.rand(5, 7, 11)
    y = common_layers.layer_norm(tf.constant(x, dtype=tf.float32), (11))
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(y)
    self.assertEqual(res.shape, (5, 7, 11))

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testGroupNorm(self):
    x = np.random.rand(5, 7, 3, 16)
    y = common_layers.group_norm(tf.constant(x, dtype=tf.float32))
    self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(y)
    self.assertEqual(res.shape, (5, 7, 3, 16))

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testPadToSameLength(self):
    x1 = np.random.rand(5, 7, 11)
    x2 = np.random.rand(5, 9, 11)
    a, b = common_layers.pad_to_same_length(
        tf.constant(x1, dtype=tf.float32), tf.constant(x2, dtype=tf.float32))
    c, d = common_layers.pad_to_same_length(
        tf.constant(x1, dtype=tf.float32),
        tf.constant(x2, dtype=tf.float32),
        final_length_divisible_by=4)
    res1, res2 = self.evaluate([a, b])
    res1a, res2a = self.evaluate([c, d])
    self.assertEqual(res1.shape, (5, 9, 11))
    self.assertEqual(res2.shape, (5, 9, 11))
    self.assertEqual(res1a.shape, (5, 12, 11))
    self.assertEqual(res2a.shape, (5, 12, 11))

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testApplyNormLayer(self):
    x1 = np.random.rand(5, 2, 1, 11)
    x2 = common_layers.apply_norm(
        tf.constant(x1, dtype=tf.float32), "layer", depth=11, epsilon=1e-6)
    self.evaluate(tf.global_variables_initializer())
    actual = self.evaluate(x2)
    self.assertEqual(actual.shape, (5, 2, 1, 11))

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testApplyNormNoam(self):
    x1 = np.random.rand(5, 2, 1, 11)
    x2 = common_layers.apply_norm(
        tf.constant(x1, dtype=tf.float32), "noam", depth=11, epsilon=1e-6)
    self.evaluate(tf.global_variables_initializer())
    actual = self.evaluate(x2)
    self.assertEqual(actual.shape, (5, 2, 1, 11))

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testApplyNormBatch(self):
    x1 = np.random.rand(5, 2, 1, 11)
    x2 = common_layers.apply_norm(
        tf.constant(x1, dtype=tf.float32), "batch", depth=11, epsilon=1e-6)
    self.evaluate(tf.global_variables_initializer())
    actual = self.evaluate(x2)
    self.assertEqual(actual.shape, (5, 2, 1, 11))

  @tf.contrib.eager.run_test_in_graph_and_eager_modes()
  def testApplyNormNone(self):
    x1 = np.random.rand(5, 2, 1, 11)
    x2 = common_layers.apply_norm(
        tf.constant(x1, dtype=tf.float32), "none", depth=11, epsilon=1e-6)
    self.evaluate(tf.global_variables_initializer())
    actual = self.evaluate(x2)
    self.assertEqual(actual.shape, (5, 2, 1, 11))
    self.assertAllClose(actual, x1, atol=1e-03)


if __name__ == "__main__":
  tf.test.main()
