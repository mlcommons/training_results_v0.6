"""Tests for t2t_transformer.tensor2tensor.beam_search."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from utils import beam_search


class BeamSearchTest(tf.test.TestCase):

  def testComputeTopkScoresAndSeq(self):
    beam_size = 3

    sequences = tf.constant([[[2, 3], [4, 5], [6, 7], [19, 20]],
                             [[8, 9], [10, 11], [12, 13], [80, 17]]])

    scores = tf.constant([[-0.1, -2.5, 0., -1.5],
                          [-100., -5., -0.00789, -1.34]])
    flags = tf.constant([[True, False, False, True],
                         [False, False, False, True]])

    topk_seq, topk_scores, topk_flags, _ = (
        beam_search.compute_topk_scores_and_seq(
            sequences, scores, scores, flags, beam_size))

    with self.test_session():
      topk_seq = topk_seq.eval()
      topk_scores = topk_scores.eval()
      topk_flags = topk_flags.eval()

    exp_seq = [[[6, 7], [2, 3], [19, 20]], [[12, 13], [80, 17], [10, 11]]]
    exp_scores = [[0., -0.1, -1.5], [-0.00789, -1.34, -5.]]

    exp_flags = [[False, True, True], [False, True, False]]
    self.assertAllEqual(exp_seq, topk_seq)
    self.assertAllClose(exp_scores, topk_scores)
    self.assertAllEqual(exp_flags, topk_flags)

  def testStateBeamTwo(self):
    batch_size = 1
    beam_size = 2
    vocab_size = 3
    decode_length = 3

    initial_ids = tf.constant([0] * batch_size)  # GO
    probabilities = tf.constant([[[0.1, 0.1, 0.8], [0.1, 0.1, 0.8]],
                                 [[0.4, 0.5, 0.1], [0.2, 0.4, 0.4]],
                                 [[0.05, 0.9, 0.05], [0.4, 0.4, 0.2]]])

    # The top beam is always selected so we should see the top beam's state
    # at each position, which is the one thats getting 3 added to it each step.
    expected_states = tf.constant([[[0.], [0.]], [[3.], [3.]], [[6.], [6.]]])

    def symbols_to_logits(_, i, states, kv_encdecs):  # pylint: disable=unused-argument
      # We have to assert the values of state inline here since we can't fetch
      # them out of the loop!
      with tf.control_dependencies(
          [tf.assert_equal(states["state"], expected_states[i])]):
        logits = tf.to_float(tf.log(probabilities[i, :]))

      states["state"] += tf.constant([[3.], [7.]])
      return logits, states

    states = {
        "state": tf.zeros((batch_size, 1)),
    }
    states["state"] = tf.placeholder_with_default(
        states["state"], shape=(None, 1))

    final_ids, _ = beam_search.beam_search(
        symbols_to_logits,
        initial_ids,
        beam_size,
        decode_length,
        vocab_size,
        0.0,
        eos_id=1,
        states=states)

    with self.test_session() as sess:
      # Catch and fail so that the testing framework doesn't think it's an error
      try:
        sess.run(final_ids)
      except tf.errors.InvalidArgumentError as e:
        raise AssertionError(e.message)

  def testTPUBeam(self):
    batch_size = 1
    beam_size = 2
    vocab_size = 3
    decode_length = 3

    initial_ids = tf.constant([0] * batch_size)  # GO
    probabilities = tf.constant([[[0.1, 0.1, 0.8], [0.1, 0.1, 0.8]],
                                 [[0.4, 0.5, 0.1], [0.2, 0.4, 0.4]],
                                 [[0.05, 0.9, 0.05], [0.4, 0.4, 0.2]]])

    # The top beam is always selected so we should see the top beam's state
    # at each position, which is the one thats getting 3 added to it each step.
    expected_states = tf.constant([[[0.], [0.]], [[3.], [3.]], [[6.], [6.]]])

    def symbols_to_logits(_, i, states, kv_encdecs):  # pylint: disable=unused-argument
      # We have to assert the values of state inline here since we can't fetch
      # them out of the loop!
      with tf.control_dependencies(
          [tf.assert_equal(states["state"], expected_states[i])]):
        logits = tf.to_float(tf.log(probabilities[i, :]))

      states["state"] += tf.constant([[3.], [7.]])
      return logits, states

    states = {
        "state": tf.zeros((batch_size, 1)),
    }
    states["state"] = tf.placeholder_with_default(
        states["state"], shape=(None, 1))

    final_ids, _ = beam_search.beam_search(
        symbols_to_logits,
        initial_ids,
        beam_size,
        decode_length,
        vocab_size,
        3.5,
        eos_id=1,
        states=states)

    with self.test_session() as sess:
      # Catch and fail so that the testing framework doesn't think it's an error
      try:
        sess.run(final_ids)
      except tf.errors.InvalidArgumentError as e:
        raise AssertionError(e.message)
    self.assertAllEqual([[[0, 2, 0, 1], [0, 2, 1, 0]]], final_ids)

if __name__ == "__main__":
  tf.test.main()
