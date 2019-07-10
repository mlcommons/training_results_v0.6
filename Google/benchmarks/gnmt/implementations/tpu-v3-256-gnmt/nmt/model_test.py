from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import tensorflow as tf

from tensorflow.python.data.util import nest
import model


class CellTest(tf.test.TestCase):

  def random(self, shape):
    elements = reduce(lambda a, b: a * b, shape)
    return tf.reshape(
        tf.constant([random.random() for _ in range(elements)],
                    dtype=tf.float32), shape)

  def testLstmCellGradient(self):
    dim = 3
    batch = 4
    with self.test_session() as sess:
      theta = []
      state = []
      dstate1 = []
      theta = {
          'kernel': self.random([dim, 4 * dim]),
          'bias': self.random([4 * dim])
      }
      state = {'c': self.random([batch, dim]), 'h': self.random([batch, dim])}
      dstate1 = {'c': self.random([batch, dim]), 'h': self.random([batch, dim])}
      inputs = {
          'rnn': self.random([batch, dim * 4]),
      }
      new_states, gates = model.lstm_cell(theta, state, inputs)

      flat_new_states = nest.flatten(new_states)
      flat_dstate1 = nest.flatten(dstate1)

      dtheta, dstates, dinputs = model.lstm_cell_grad(
          theta, state, inputs, gates, dstate1)

      for j in ['bias']:
        self.assertAllEqual(
            sess.run(tf.gradients(flat_new_states, theta[j], flat_dstate1)[0]),
            sess.run(dtheta[j]))
      for j in ['c', 'h']:
        self.assertAllEqual(
            sess.run(tf.gradients(flat_new_states, state[j], flat_dstate1)[0]),
            sess.run(dstates[j]))
      self.assertAllEqual(
          sess.run(
              tf.gradients(flat_new_states, inputs['rnn'], flat_dstate1)[0]),
          sess.run(dinputs['rnn']))

  def testAttentionCellGradient(self):
    dim = 2
    batch = 3
    time = 5
    with self.test_session() as sess:
      theta = []
      state = []
      dstate1 = []
      theta = {
          'kernel': self.random([dim, 4 * dim]),
          'attention_kernel': self.random([dim, 4 * dim]),
          'bias': self.random([4 * dim]),
          'memory_kernel': self.random([dim, dim]),
          'query_kernel': self.random([dim, dim]),
          'atten_v': self.random([dim]),
          'atten_g': self.random([1]),
          'atten_b': self.random([dim]),
          'keys': self.random([time, batch, dim]),
          'values': self.random([time, batch, dim]),
          'seq_mask': self.random([batch, time])
      }
      state = {
          'c': self.random([batch, dim]),
          'h': self.random([batch, dim]),
          'attention': self.random([batch, dim]),
      }
      dstate1 = {
          'c': self.random([batch, dim]),
          'h': self.random([batch, dim]),
          'attention': self.random([batch, dim]),
      }
      inputs = {
          'rnn': self.random([batch, 4 * dim]),
      }
      new_states, gates = model.attention_cell(theta, state, inputs)
      del new_states['alignments']

      flat_new_states = nest.flatten(new_states)
      flat_dstate1 = nest.flatten(dstate1)

      dtheta, dstates, dinputs = model.attention_cell_grad(
          theta, state, inputs, gates, dstate1)

      for j in ['bias']:
        self.assertAllEqual(
            sess.run(tf.gradients(flat_new_states, theta[j], flat_dstate1)[0]),
            sess.run(dtheta[j]))
      for j in ['c', 'h']:
        x = state[j]
        dx = dstates[j]
        self.assertAllEqual(
            sess.run(tf.gradients(flat_new_states, x, flat_dstate1)[0]),
            sess.run(dx))

      self.assertAllEqual(
          sess.run(
              tf.gradients(flat_new_states, inputs['rnn'], flat_dstate1)[0]),
          sess.run(dinputs['rnn']))


if __name__ == '__main__':
  tf.test.main()
