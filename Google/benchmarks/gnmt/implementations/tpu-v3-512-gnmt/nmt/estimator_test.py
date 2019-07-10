# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for nmt.py, train.py and inference.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf

import estimator
import nmt
from estimator import make_input_fn


def update_flags(flags, test_name):
  """Update flags for basic training."""
  flags.steps_per_stats = 5
  flags.src = "en"
  flags.tgt = "vi"
  flags.learning_rate = 5e-4
  flags.num_units = 32

  flags.train_prefix = ("third_party/tensorflow_models/mlperf/models/rough/nmt/"
                        "testdata/iwslt15.tst2013.100")
  flags.vocab_prefix = ("third_party/tensorflow_models/mlperf/models/rough/nmt/"
                        "testdata/iwslt15.vocab.100")
  flags.test_prefix = ("third_party/tensorflow_models/mlperf/models/rough/nmt/"
                       "testdata/iwslt15.tst2013.100")
  flags.out_dir = os.path.join(tf.test.get_temp_dir(), test_name)
  flags.max_train_epochs = 2
  flags.num_examples_per_epoch = 16
  flags.src_max_len = 128
  flags.tgt_max_len = 128
  flags.batch_size = 8
  flags.infer_batch_size = 4
  flags.num_buckets = 1
  flags.use_borg = True
  flags.examples_to_infer = 4
  flags.tgt_max_len_infer = 10
  flags.activation_dtype = "bfloat16"
  flags.num_shards = 2
  flags.num_shards_per_host = 2
  flags.master = "local"
  flags.use_preprocessed_data = False


class EstimatorTest(tf.test.TestCase):

  def testTrainInputFn(self):
    nmt_parser = argparse.ArgumentParser()
    nmt.add_arguments(nmt_parser)
    flags, _ = nmt_parser.parse_known_args()
    update_flags(flags, "input_fn_test")
    default_hparams = nmt.create_hparams(flags)
    hparams = nmt.extend_hparams(default_hparams)

    with self.test_session() as sess:
      input_fn = make_input_fn(hparams, tf.contrib.learn.ModeKeys.TRAIN)
      outputs = input_fn({})
      sess.run(tf.tables_initializer())
      iterator = outputs.make_initializable_iterator()
      sess.run(iterator.initializer)
      features = sess.run(iterator.get_next())
      tf.logging.info("source: %s", features["source"])
      tf.logging.info("target_input: %s", features["target_input"])
      tf.logging.info("target_output: %s", features["target_output"])
      tf.logging.info("source_sequence_length: %s",
                      features["source_sequence_length"])
      tf.logging.info("target_sequence_length: %s",
                      features["target_sequence_length"])

  def testTrain(self):
    """Test the training loop is functional with basic hparams."""
    nmt_parser = argparse.ArgumentParser()
    nmt.add_arguments(nmt_parser)
    flags, _ = nmt_parser.parse_known_args()
    update_flags(flags, "nmt_train_test")
    default_hparams = nmt.create_hparams(flags)

    def estimator_fn(hparams):
      return estimator.train_and_eval_with_low_level_api(hparams)

    nmt.run_main(flags, default_hparams, estimator_fn)


if __name__ == "__main__":
  tf.test.main()
