"""t2t_transformer.tensor2tensor.problems test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import problems


class ProblemsTest(tf.test.TestCase):

  def testImport(self):
    self.assertIsNotNone(problems)

if __name__ == "__main__":
  tf.test.main()
