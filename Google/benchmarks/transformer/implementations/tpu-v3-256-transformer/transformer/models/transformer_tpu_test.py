"""Tests for t2t_transformer.tensor2tensor.models.transformer_tpu."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from data_generators import translate_ende
from models import transformer
from utils import decoding
from utils import trainer_lib

BATCH_SIZE = 3
INPUT_LENGTH = 5
DECODE_LENGTH = 5
TARGET_LENGTH = 7
VOCAB_SIZE = 10


class TransformerTpuTest(tf.test.TestCase):

  def setUp(self):
    self._model_dir = os.path.join(tf.test.get_temp_dir(), "model_dir")
    tf.gfile.MakeDirs(self._model_dir)

  def tearDown(self):
    tf.gfile.DeleteRecursively(self._model_dir)

  def testFastBeamDecodeTPU(self):
    hparams = transformer.transformer_small_tpu()
    hparams.data_dir = os.path.expanduser(
        "/placer/prod/home/tensor2tensor/datasets/rs=6.3/v1/")
    hparams.problem = translate_ende.TranslateEndeWmt32k()

    p_hparams = hparams.problem.get_hparams(hparams)
    hparams.problem_hparams = p_hparams
    hparams.pad_batch = False

    decode_hp = decoding.decode_hparams(
        "batch_size=64,beam_size=4,num_samples=128")

    if decode_hp.batch_size:
      hparams.batch_size = decode_hp.batch_size
      hparams.use_fixed_batch_size = True

    dataset_kwargs = {
        "shard": None,
        "dataset_split": None,
        "max_records": decode_hp.num_samples
    }

    run_config = trainer_lib.create_run_config(
        model_name="transformer",
        master="",
        model_dir=os.path.expanduser(self._model_dir),
        num_shards=2,
        use_tpu=True)

    estimator = trainer_lib.create_estimator(
        "transformer",
        hparams,
        run_config,
        decode_hparams=decode_hp,
        use_tpu=True)

    infer_input_fn = hparams.problem.make_estimator_input_fn(
        tf.estimator.ModeKeys.PREDICT, hparams, dataset_kwargs=dataset_kwargs)

    predictions = estimator.predict(infer_input_fn)
    num_predictions = 0
    for _, predictions in enumerate(predictions):
      num_predictions += 1
      self.assertEqual(predictions["outputs"].shape[0],
                       predictions["inputs"].shape[0] + decode_hp.extra_length)

    self.assertEqual(num_predictions, 128)


if __name__ == "__main__":
  tf.test.main()
