"""Tests for Modalities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf

from layers import common_hparams
from layers import modalities


class ModalityTest(tf.test.TestCase):

  def testSymbolModalityInputs(self):
    batch_size = 10
    length = 5
    vocab_size = 5000
    hidden_size = 9
    model_hparams = common_hparams.basic_params1()
    model_hparams.hidden_size = hidden_size
    model_hparams.mode = tf.estimator.ModeKeys.TRAIN
    x = -1 + np.random.random_integers(
        vocab_size, size=(batch_size, length, 1, 1))
    m = modalities.SymbolModality(model_hparams, vocab_size)
    output = m.bottom(tf.convert_to_tensor(x))
    init = self.evaluate(tf.global_variables_initializer())
    res = self.evaluate(output)
    tf.logging.info(init)
    self.assertEqual(res.shape, (batch_size, length, 1, hidden_size))

  def testSymbolModalityTargets(self):
    batch_size = 10
    length = 6
    height = 7
    hidden_size = 9
    vocab_size = 11
    model_hparams = common_hparams.basic_params1()
    model_hparams.hidden_size = hidden_size
    model_hparams.mode = tf.estimator.ModeKeys.TRAIN
    body_output = -1 + np.random.random_integers(
        100, size=(batch_size, length, height, hidden_size))
    targets = -1 + np.random.random_integers(
        vocab_size, size=(batch_size, length, height, 1))
    m = modalities.SymbolModality(model_hparams, vocab_size)
    logits = m.top(tf.to_float(body_output), targets)
    train_loss_num, train_loss_den = m.loss(logits, targets)
    train_loss = train_loss_num / tf.maximum(1.0, train_loss_den)
    self.evaluate(tf.global_variables_initializer())
    res1, res2 = self.evaluate((logits, train_loss))
    self.assertEqual(res1.shape, (batch_size, length, height, 1, vocab_size))
    self.assertEqual(res2.shape, ())

if __name__ == "__main__":
  tf.test.main()
