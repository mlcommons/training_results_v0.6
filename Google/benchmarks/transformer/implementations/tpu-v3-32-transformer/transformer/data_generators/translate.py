"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

import tensorflow as tf
from mlp_log import mlp_log
from data_generators import generator_utils
from data_generators import problem
from data_generators import text_problems
from utils import bleu_hook

FLAGS = tf.flags.FLAGS


class TranslateProblem(text_problems.Text2TextProblem):
  """Base class for translation problems."""

  def is_generate_per_split(self):
    return True

  @property
  def approx_vocab_size(self):
    return 2**15

  def source_data_files(self, dataset_split):
    """Files to be passed to compile_data."""
    raise NotImplementedError()

  def vocab_data_files(self):
    """Files to be passed to get_or_generate_vocab."""
    return self.source_data_files(problem.DatasetSplit.TRAIN)

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    datasets = self.source_data_files(dataset_split)
    tag = "train" if dataset_split == problem.DatasetSplit.TRAIN else "dev"
    data_path = compile_data(tmp_dir, datasets, "%s-compiled-%s" % (self.name,
                                                                    tag))
    return text_problems.text2text_txt_iterator(data_path + ".lang1",
                                                data_path + ".lang2")

  def generate_text_for_vocab(self, data_dir, tmp_dir):
    return generator_utils.generate_lines_for_vocab(tmp_dir,
                                                    self.vocab_data_files())

  @property
  def decode_hooks(self):
    return [compute_bleu_summaries]


def compute_bleu_summaries(hook_args):
  """Compute BLEU core summaries using the decoder output.

  Args:
    hook_args: DecodeHookArgs namedtuple
  Returns:
    A list of tf.Summary values if hook_args.hparams contains the
    reference file and the translated file.
  """
  outputs, references = [], []
  for output, reference in hook_args.predictions:
    outputs.append(output)
    references.append(reference)

  decode_hparams = hook_args.decode_hparams

  values = []
  bleu = 100 * bleu_hook.bleu_wrapper(references, outputs)
  values.append(tf.Summary.Value(tag="BLEU", simple_value=bleu))
  tf.logging.info("BLEU = %6.2f" % (bleu))
  if hook_args.hparams.mlperf_mode:
    current_step = decode_hparams.mlperf_decode_step
    mlp_log.mlperf_print(
        "eval_stop",
        None,
        metadata={
            "epoch_num":
                max(current_step // decode_hparams.iterations_per_loop, 1)
        })
    mlp_log.mlperf_print(
        "eval_accuracy",
        bleu,
        metadata={
            "epoch_num":
                max(current_step // decode_hparams.iterations_per_loop, 1)
        })

  if bleu >= decode_hparams.mlperf_threshold:
    mlp_log.mlperf_print("run_stop", None, metadata={"status": "success"})
    decode_hparams.set_hparam("mlperf_success", True)

  return values


def _preprocess_sgm(line, is_sgm):
  """Preprocessing to strip tags in SGM files."""
  if not is_sgm:
    return line
  # In SGM files, remove <srcset ...>, <p>, <doc ...> lines.
  if line.startswith("<srcset") or line.startswith("</srcset"):
    return ""
  if line.startswith("<doc") or line.startswith("</doc"):
    return ""
  if line.startswith("<p>") or line.startswith("</p>"):
    return ""
  # Strip <seg> tags.
  line = line.strip()
  if line.startswith("<seg") and line.endswith("</seg>"):
    i = line.index(">")
    return line[i + 1:-6]  # Strip first <seg ...> and last </seg>.


def compile_data(tmp_dir, datasets, filename):
  """Concatenate all `datasets` and save to `filename`."""
  filename = os.path.join(tmp_dir, filename)
  lang1_fname = filename + ".lang1"
  lang2_fname = filename + ".lang2"
  if tf.gfile.Exists(lang1_fname) and tf.gfile.Exists(lang2_fname):
    tf.logging.info("Skipping compile data, found files:\n%s\n%s", lang1_fname,
                    lang2_fname)
    return filename
  with tf.gfile.GFile(lang1_fname, mode="w") as lang1_resfile:
    with tf.gfile.GFile(lang2_fname, mode="w") as lang2_resfile:
      for dataset in datasets:
        url = dataset[0]
        compressed_filename = os.path.basename(url)
        compressed_filepath = os.path.join(tmp_dir, compressed_filename)
        if url.startswith("http"):
          generator_utils.maybe_download(tmp_dir, compressed_filename, url)

        if dataset[1][0] == "tsv":
          _, src_column, trg_column, glob_pattern = dataset[1]
          filenames = tf.gfile.Glob(os.path.join(tmp_dir, glob_pattern))
          if not filenames:
            # Capture *.tgz and *.tar.gz too.
            mode = "r:gz" if compressed_filepath.endswith("gz") else "r"
            with tarfile.open(compressed_filepath, mode) as corpus_tar:
              corpus_tar.extractall(tmp_dir)
            filenames = tf.gfile.Glob(os.path.join(tmp_dir, glob_pattern))
          for tsv_filename in filenames:
            if tsv_filename.endswith(".gz"):
              new_filename = tsv_filename.strip(".gz")
              generator_utils.gunzip_file(tsv_filename, new_filename)
              tsv_filename = new_filename
            with tf.gfile.Open(tsv_filename) as tsv_file:
              for line in tsv_file:
                if line and "\t" in line:
                  parts = line.split("\t")
                  source, target = parts[src_column], parts[trg_column]
                  source, target = source.strip(), target.strip()
                  if source and target:
                    lang1_resfile.write(source)
                    lang1_resfile.write("\n")
                    lang2_resfile.write(target)
                    lang2_resfile.write("\n")
        else:
          lang1_filename, lang2_filename = dataset[1]
          lang1_filepath = os.path.join(tmp_dir, lang1_filename)
          lang2_filepath = os.path.join(tmp_dir, lang2_filename)
          is_sgm = (
              lang1_filename.endswith("sgm") and lang2_filename.endswith("sgm"))

          if not (tf.gfile.Exists(lang1_filepath) and
                  tf.gfile.Exists(lang2_filepath)):
            # For .tar.gz and .tgz files, we read compressed.
            mode = "r:gz" if compressed_filepath.endswith("gz") else "r"
            with tarfile.open(compressed_filepath, mode) as corpus_tar:
              corpus_tar.extractall(tmp_dir)
          if lang1_filepath.endswith(".gz"):
            new_filepath = lang1_filepath.strip(".gz")
            generator_utils.gunzip_file(lang1_filepath, new_filepath)
            lang1_filepath = new_filepath
          if lang2_filepath.endswith(".gz"):
            new_filepath = lang2_filepath.strip(".gz")
            generator_utils.gunzip_file(lang2_filepath, new_filepath)
            lang2_filepath = new_filepath

          for example in text_problems.text2text_txt_iterator(
              lang1_filepath, lang2_filepath):
            line1res = _preprocess_sgm(example["inputs"], is_sgm)
            line2res = _preprocess_sgm(example["targets"], is_sgm)
            if line1res and line2res:
              lang1_resfile.write(line1res)
              lang1_resfile.write("\n")
              lang2_resfile.write(line2res)
              lang2_resfile.write("\n")

  return filename
