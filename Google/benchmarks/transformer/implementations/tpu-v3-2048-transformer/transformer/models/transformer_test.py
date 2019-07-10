"""Tests for Transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np

import tensorflow as tf

from data_generators import problem_hparams
from models import transformer


BATCH_SIZE = 3
INPUT_LENGTH = 5
TARGET_LENGTH = 7
VOCAB_SIZE = 10


def get_model(hparams=None, mode=tf.estimator.ModeKeys.TRAIN,
              model_cls=transformer.Transformer):
  if hparams is None:
    hparams = transformer.transformer_small()
  hparams.hidden_size = 8
  hparams.filter_size = 32
  hparams.num_heads = 1
  hparams.layer_prepostprocess_dropout = 0.0

  p_hparams = problem_hparams.test_problem_hparams(VOCAB_SIZE,
                                                   VOCAB_SIZE,
                                                   hparams)
  hparams.problem_hparams = p_hparams

  inputs = -1 + np.random.random_integers(
      VOCAB_SIZE, size=(BATCH_SIZE, INPUT_LENGTH, 1, 1))
  targets = -1 + np.random.random_integers(
      VOCAB_SIZE, size=(BATCH_SIZE, TARGET_LENGTH, 1, 1))
  features = {
      "targets": tf.constant(targets, dtype=tf.int32, name="targets"),
      "target_space_id": tf.constant(1, dtype=tf.int32)
  }
  features["inputs"] = tf.constant(inputs, dtype=tf.int32, name="inputs")

  return model_cls(hparams, mode, p_hparams), features


class TransformerTest(tf.test.TestCase):

  def testTransformer(self):
    model, features = get_model(transformer.transformer_small())
    logits, _ = model(features)
    with self.test_session() as session:
      session.run(tf.global_variables_initializer())
      res = session.run(logits)
    self.assertEqual(res.shape, (BATCH_SIZE, TARGET_LENGTH, 1, 1, VOCAB_SIZE))

  def testGreedy(self):
    model, features = get_model(transformer.transformer_small())

    decode_length = 3

    out_logits, _ = model(features)
    out_logits = tf.squeeze(out_logits, axis=[2, 3])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=tf.reshape(out_logits, [-1, VOCAB_SIZE]),
        labels=tf.reshape(features["targets"], [-1]))
    loss = tf.reduce_mean(loss)
    apply_grad = tf.train.AdamOptimizer(0.001).minimize(loss)

    with self.test_session():
      tf.global_variables_initializer().run()
      for _ in range(100):
        apply_grad.run()

    model.set_mode(tf.estimator.ModeKeys.PREDICT)

    with self.assertRaises(NotImplementedError):
      with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        greedy_result = model.infer(features, decode_length)["outputs"]

  def testTransformerWithoutProblem(self):
    hparams = transformer.transformer_test()

    embedded_inputs = np.random.random_sample(
        (BATCH_SIZE, INPUT_LENGTH, 1, hparams.hidden_size))
    embedded_targets = np.random.random_sample(
        (BATCH_SIZE, TARGET_LENGTH, 1, hparams.hidden_size))

    transformed_features = {
        "inputs": tf.constant(embedded_inputs, dtype=tf.float32),
        "targets": tf.constant(embedded_targets, dtype=tf.float32)
    }

    model = transformer.Transformer(hparams)
    body_out, _ = model(transformed_features)

    self.assertAllEqual(
        body_out.get_shape().as_list(),
        [BATCH_SIZE, TARGET_LENGTH, 1, hparams.hidden_size])


if __name__ == "__main__":
  tf.test.main()
