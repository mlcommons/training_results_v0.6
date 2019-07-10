"""Utilties for genearting data from tf.Examples."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from contextlib import contextmanager
import six
import tensorflow as tf


@contextmanager
def managed_session():
  """Yield a session that handles initialization and teardown."""
  with tf.Session() as sess:
    sess.run(
        [tf.global_variables_initializer(),
         tf.local_variables_initializer()])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
      yield sess
    except tf.errors.OutOfRangeError:
      tf.logging.info("OutOfRangeError")

    coord.request_stop()
    coord.join(threads)


def example_generator(filenames, features):
  """Yield example dictionaries from TFRecord files."""
  tf.logging.info("Reading examples from %s", filenames)
  fname_q = tf.train.string_input_producer(
      filenames, num_epochs=1, shuffle=False)
  reader = tf.TFRecordReader()
  # TODO(peterjliu): Refactor to use tf.data.Dataset.
  _, example_serialized = reader.read(fname_q)
  feature_values = tf.parse_single_example(example_serialized, features)
  for k, v in six.iteritems(feature_values):
    feature_values[k] = v.values

  with managed_session() as sess:
    i = 0
    while True:
      if i % 50 == 0:
        tf.logging.info("Reading example %d..." % i)
      fvals = sess.run(feature_values)
      fvals = dict([(k, list(v)) for k, v in six.iteritems(fvals)])
      yield fvals
      i += 1
