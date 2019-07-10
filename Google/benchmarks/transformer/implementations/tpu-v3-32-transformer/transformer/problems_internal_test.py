"""t2t_transformer.tensor2tensor.problems test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import problems

MODES = tf.estimator.ModeKeys


class ProblemsTest(tf.test.TestCase):

  def testBuildDataset(self):
    # See all the available problems
    self.assertTrue(len(problems.available()) > 10)

    # Retrieve a problem by name
    problem = problems.problem("translate_ende_wmt32k_packed")

    # Access train and dev datasets through Problem
    train_dataset = problem.dataset(MODES.TRAIN)
    dev_dataset = problem.dataset(MODES.EVAL)

    # Access vocab size and other info (e.g. the data encoders used to
    # encode/decode data for the feature, used below) through feature_info.
    feature_info = problem.feature_info
    self.assertTrue(feature_info["inputs"].vocab_size > 0)
    self.assertTrue(feature_info["targets"].vocab_size > 0)

    train_example = train_dataset.make_one_shot_iterator().get_next()
    dev_example = dev_dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
      train_ex_val, _ = sess.run([train_example, dev_example])
      _ = feature_info["inputs"].encoder.decode(train_ex_val["inputs"])
      _ = feature_info["targets"].encoder.decode(train_ex_val["targets"])


if __name__ == "__main__":
  tf.test.main()
