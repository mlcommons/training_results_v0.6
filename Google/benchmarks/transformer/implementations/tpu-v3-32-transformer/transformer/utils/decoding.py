"""Decoding utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import operator
import os
import re
import time

import numpy as np
import six

import tensorflow as tf
from data_generators import text_encoder

FLAGS = tf.flags.FLAGS


def decode_hparams(overrides=""):
  """Hyperparameters for decoding."""
  hp = tf.contrib.training.HParams(
      log_results=True,
      extra_length=50,
      batch_size=64,
      beam_size=4,
      alpha=0.6,
      eos_penalty=0.0,
      block_size=0,
      guess_and_check_top_k=0,
      guess_and_check_epsilon=-1,
      write_beam_scores=False,
      max_input_size=-1,
      identity_output=False,
      num_samples=3003,
      delimiter="\n",
      decode_to_file=None,
      decode_in_memory=True,
      summaries_log_dir="decode",  # Directory to write hook summaries.
      shards=1,  # How many shards of data to decode (treating 1 as None).
      shard_id=0,  # Which shard are we decoding if more than 1 above.
      shards_start_offset=0,  # Number of the first shard to decode.
      num_decodes=1,
      force_decode_length=False,
      display_decoded_images=False,
      # Used for video decoding.
      frames_per_second=10,
      skip_eos_postprocess=False,
      # Creates a blue/red border covering border_percent of the frame.
      border_percent=2,
      # Maximum number of videos displayed.
      # Total number of videos are max_display_outputs * num_decodes
      max_display_outputs=10,
      # Used for MLPerf compliance logging.
      mlperf_decode_step=0.0,
      mlperf_threshold=25.0,
      mlperf_success=False)
  hp.parse(overrides)
  return hp


def log_decode_results(inputs,
                       outputs,
                       inputs_vocab,
                       targets_vocab,
                       targets=None,
                       identity_output=False,
                       log_results=True):
  """Log inference results."""
  decoded_inputs = None
  if inputs is not None and inputs_vocab:
    if identity_output:
      decoded_inputs = " ".join(map(str, inputs.flatten()))
    else:
      decoded_inputs = inputs_vocab.decode(save_until_eos(inputs))

    if log_results:
      tf.logging.info("Inference results INPUT: %s" % decoded_inputs)

  decoded_targets = None
  decoded_outputs = None
  if identity_output:
    decoded_outputs = " ".join(map(str, outputs.flatten()))
    if targets is not None:
      decoded_targets = " ".join(map(str, targets.flatten()))
  else:
    decoded_outputs = targets_vocab.decode(save_until_eos(outputs))
    if targets is not None and log_results:
      decoded_targets = targets_vocab.decode(save_until_eos(targets))

  tf.logging.info("Inference results OUTPUT: %s" % decoded_outputs)
  if targets is not None and log_results:
    tf.logging.info("Inference results TARGET: %s" % decoded_targets)
  return decoded_inputs, decoded_outputs, decoded_targets


def decode_from_dataset(estimator,
                        problem_name,
                        hparams,
                        decode_hp,
                        decode_to_file=None,
                        dataset_split=None,
                        checkpoint_path=None,
                        erunner=None,
                        runner=None):  # pylint: disable=unused-argument
  """Perform decoding from dataset."""
  tf.logging.info("Performing local inference from dataset for %s.",
                  str(problem_name))
  # We assume that worker_id corresponds to shard number.
  shard = decode_hp.shard_id if decode_hp.shards > 1 else None

  # Setup decode output directory for any artifacts that may be written out
  output_dir = os.path.join(estimator.model_dir, "decode")
  tf.gfile.MakeDirs(output_dir)

  # If decode_hp.batch_size is specified, use a fixed batch size
  if decode_hp.batch_size:
    hparams.batch_size = decode_hp.batch_size
    hparams.use_fixed_batch_size = True

  dataset_kwargs = {
      "shard": shard,
      "dataset_split": dataset_split,
      "max_records": decode_hp.num_samples
  }

  # Build the inference input function
  problem = hparams.problem
  if not erunner:
    infer_input_fn = problem.make_estimator_input_fn(
        tf.estimator.ModeKeys.PREDICT, hparams, dataset_kwargs=dataset_kwargs)

  predictions, output_dirs = [], []
  for decode_id in range(decode_hp.num_decodes):
    tf.logging.info("Decoding {}".format(decode_id))

    # Create decode directory if not in-memory decoding.
    if not decode_hp.decode_in_memory:
      output_dir = os.path.join(estimator.model_dir, "decode_%05d" % decode_id)
      tf.gfile.MakeDirs(output_dir)
      output_dirs.append(output_dir)

    result = decode_once(
        estimator,
        problem_name,
        hparams,
        infer_input_fn if not erunner else None,
        decode_hp,
        decode_to_file,
        log_results=not decode_hp.decode_in_memory,
        checkpoint_path=checkpoint_path,
        erunner=erunner)

    if decode_hp.decode_in_memory:
      output_dirs = [output_dir]
      inputs_vocab = hparams.problem_hparams.vocabulary["inputs"]
      targets_vocab = hparams.problem_hparams.vocabulary["targets"]
      for prediction in result:
        inputs = prediction.get("inputs")
        targets = prediction.get("targets")
        outputs = prediction.get("outputs")
        if not re.match("^({})+$".format(text_encoder.PAD),
                        inputs_vocab.decode(save_until_eos(inputs))):
          predictions.append((targets_vocab.decode(save_until_eos(outputs)),
                              targets_vocab.decode(save_until_eos(targets))))

  if decode_hp.decode_to_file:
    decode_hp.decode_to_file = _decode_filename(
        decode_hp.decode_to_file, problem_name, decode_hp)

  run_postdecode_hooks(DecodeHookArgs(
      estimator=estimator,
      problem=problem,
      output_dirs=output_dirs,
      hparams=hparams,
      decode_hparams=decode_hp,
      predictions=predictions
  ), dataset_split)
  return predictions


def decode_once(estimator,
                problem_name,
                hparams,
                infer_input_fn,
                decode_hp,
                decode_to_file,
                log_results=True,
                checkpoint_path=None,
                erunner=None):
  """Decodes once."""

  # Get the predictions as an iterable
  if hparams.decode_with_low_level_api:
    predictions = erunner.predict(decode_hp, checkpoint_path=checkpoint_path)
  else:
    predictions = estimator.predict(
        infer_input_fn, checkpoint_path=checkpoint_path)

  if not log_results:
    return list(predictions)

  # Prepare output file writers if decode_to_file passed
  decode_to_file = decode_to_file or decode_hp.decode_to_file
  if decode_to_file:
    output_filepath = _decode_filename(decode_to_file, problem_name, decode_hp)
    parts = output_filepath.split(".")
    parts[-1] = "targets"
    target_filepath = ".".join(parts)
    parts[-1] = "inputs"
    input_filepath = ".".join(parts)

    output_file = tf.gfile.Open(output_filepath, "w")
    target_file = tf.gfile.Open(target_filepath, "w")
    input_file = tf.gfile.Open(input_filepath, "w")

  problem_hparams = hparams.problem_hparams
  # Inputs vocabulary is set to targets if there are no inputs in the problem,
  # e.g., for language models where the inputs are just a prefix of targets.
  has_input = "inputs" in problem_hparams.vocabulary
  inputs_vocab_key = "inputs" if has_input else "targets"
  inputs_vocab = problem_hparams.vocabulary[inputs_vocab_key]
  targets_vocab = problem_hparams.vocabulary["targets"]

  num_eval_samples = 0
  for num_predictions, prediction in enumerate(predictions):
    num_eval_samples += 1
    num_predictions += 1
    inputs = prediction.get("inputs")
    targets = prediction.get("targets")
    outputs = prediction.get("outputs")

    # Log predictions
    decoded_outputs = []
    decoded_scores = []
    decoded = log_decode_results(
        inputs,
        outputs,
        inputs_vocab,
        targets_vocab,
        identity_output=decode_hp.identity_output,
        targets=targets,
        log_results=decode_hp.log_results)
    decoded_outputs.append(decoded)

    # Write out predictions if decode_to_file passed
    if decode_to_file:
      for i, (d_input, d_output, d_target) in enumerate(decoded_outputs):
        # Skip if all padding
        if d_input and re.match("^({})+$".format(text_encoder.PAD), d_input):
          continue
        beam_score_str = ""
        if decode_hp.write_beam_scores:
          beam_score_str = "\t%.2f" % decoded_scores[i]
        output_file.write(str(d_output) + beam_score_str + decode_hp.delimiter)
        target_file.write(str(d_target) + decode_hp.delimiter)
        input_file.write(str(d_input) + decode_hp.delimiter)

    if (decode_hp.num_samples >= 0 and
        num_predictions >= decode_hp.num_samples):
      break

  if decode_to_file:
    output_file.close()
    target_file.close()
    input_file.close()


def decode_from_file(estimator,
                     filename,
                     hparams,
                     decode_hp,
                     decode_to_file=None,
                     checkpoint_path=None):
  """Compute predictions on entries in filename and write them out."""
  if not decode_hp.batch_size:
    decode_hp.batch_size = 32
    tf.logging.info(
        "decode_hp.batch_size not specified; default=%d" % decode_hp.batch_size)

  # Inputs vocabulary is set to targets if there are no inputs in the problem,
  # e.g., for language models where the inputs are just a prefix of targets.
  p_hp = hparams.problem_hparams
  has_input = "inputs" in p_hp.vocabulary
  inputs_vocab_key = "inputs" if has_input else "targets"
  inputs_vocab = p_hp.vocabulary[inputs_vocab_key]
  targets_vocab = p_hp.vocabulary["targets"]
  problem_name = FLAGS.problem
  filename = _add_shard_to_filename(filename, decode_hp)
  tf.logging.info("Performing decoding from file (%s)." % filename)
  sorted_inputs, sorted_keys = _get_sorted_inputs(filename, decode_hp.delimiter)
  num_decode_batches = (len(sorted_inputs) - 1) // decode_hp.batch_size + 1

  def input_fn():
    input_gen = _decode_batch_input_fn(
        num_decode_batches, sorted_inputs,
        inputs_vocab, decode_hp.batch_size,
        decode_hp.max_input_size)
    gen_fn = make_input_fn_from_generator(input_gen)
    example = gen_fn()
    return _decode_input_tensor_to_features_dict(example, hparams)

  decodes = []
  result_iter = estimator.predict(input_fn, checkpoint_path=checkpoint_path)

  start_time = time.time()
  total_time_per_step = 0
  total_cnt = 0

  def timer(gen):
    while True:
      try:
        start_time = time.time()
        item = next(gen)
        elapsed_time = time.time() - start_time
        yield elapsed_time, item
      except StopIteration:
        break

  for elapsed_time, result in timer(result_iter):
    _, decoded_outputs, _ = log_decode_results(
        result["inputs"],
        result["outputs"],
        inputs_vocab,
        targets_vocab,
        log_results=decode_hp.log_results)
    decodes.append(decoded_outputs)
    total_time_per_step += elapsed_time
    total_cnt += result["outputs"].shape[-1]
  tf.logging.info("Elapsed Time: %5.5f" % (time.time() - start_time))
  tf.logging.info("Averaged Single Token Generation Time: %5.7f "
                  "(time %5.7f count %d)" %
                  (total_time_per_step / total_cnt,
                   total_time_per_step, total_cnt))

  # Reversing the decoded inputs and outputs because they were reversed in
  # _decode_batch_input_fn
  sorted_inputs.reverse()
  decodes.reverse()
  # If decode_to_file was provided use it as the output filename without change
  # (except for adding shard_id if using more shards for decoding).
  # Otherwise, use the input filename plus model, hp, problem, beam, alpha.
  decode_filename = decode_to_file if decode_to_file else filename
  if not decode_to_file:
    decode_filename = _decode_filename(decode_filename, problem_name, decode_hp)
  else:
    decode_filename = _add_shard_to_filename(decode_filename, decode_hp)
  tf.logging.info("Writing decodes into %s" % decode_filename)
  outfile = tf.gfile.Open(decode_filename, "w")
  for index in range(len(sorted_inputs)):
    outfile.write("%s%s" % (decodes[sorted_keys[index]], decode_hp.delimiter))
  outfile.flush()
  outfile.close()

  output_dir = os.path.join(estimator.model_dir, "decode")
  tf.gfile.MakeDirs(output_dir)

  run_postdecode_hooks(DecodeHookArgs(
      estimator=estimator,
      problem=hparams.problem,
      output_dirs=[output_dir],
      hparams=hparams,
      decode_hparams=decode_hp,
      predictions=list(result_iter)
  ), None)


def _add_shard_to_filename(filename, decode_hp):
  if decode_hp.shards > 1:
    shard_id = decode_hp.shard_id + decode_hp.shards_start_offset
    filename = filename + ("%.3d" % shard_id)
  return filename


def _decode_filename(base_filename, problem_name, decode_hp):
  """Generates decode filename.

  Args:
    base_filename: A string, base of the decode filename.
    problem_name: A string, name of the problem.
    decode_hp: HParams for decoding.

  Returns:
    A string, produced decode filename.
  """
  if decode_hp.shards > 1:
    base_filename = _add_shard_to_filename(base_filename, decode_hp)
  if ("beam{beam}.alpha{alpha}.decodes".format(
      beam=str(decode_hp.beam_size), alpha=str(decode_hp.alpha))
      in base_filename):
    return base_filename
  else:
    return (
        "{base}.{model}.{hp}.{problem}.beam{beam}.alpha{alpha}.decodes".format(
            base=base_filename,
            model=FLAGS.model,
            hp=FLAGS.hparams_set,
            problem=problem_name,
            beam=str(decode_hp.beam_size),
            alpha=str(decode_hp.alpha)))


def make_input_fn_from_generator(gen):
  """Use py_func to yield elements from the given generator."""
  first_ex = six.next(gen)
  flattened = tf.contrib.framework.nest.flatten(first_ex)
  types = [t.dtype for t in flattened]
  shapes = [[None] * len(t.shape) for t in flattened]
  first_ex_list = [first_ex]

  def py_func():
    if first_ex_list:
      example = first_ex_list.pop()
    else:
      example = six.next(gen)
    return tf.contrib.framework.nest.flatten(example)

  def input_fn():
    flat_example = tf.py_func(py_func, [], types)
    _ = [t.set_shape(shape) for t, shape in zip(flat_example, shapes)]
    example = tf.contrib.framework.nest.pack_sequence_as(first_ex, flat_example)
    return example

  return input_fn


def _decode_batch_input_fn(num_decode_batches, sorted_inputs, vocabulary,
                           batch_size, max_input_size):
  """Generator to produce batches of inputs."""
  tf.logging.info(" batch %d" % num_decode_batches)
  # First reverse all the input sentences so that if you're going to get OOMs,
  # you'll see it in the first batch
  sorted_inputs.reverse()
  for b in range(num_decode_batches):
    tf.logging.info("Decoding batch %d" % b)
    batch_length = 0
    batch_inputs = []
    for inputs in sorted_inputs[b * batch_size:(b + 1) * batch_size]:
      input_ids = vocabulary.encode(inputs)
      if max_input_size > 0:
        # Subtract 1 for the EOS_ID.
        input_ids = input_ids[:max_input_size - 1]
      final_id = text_encoder.EOS_ID
      input_ids.append(final_id)
      batch_inputs.append(input_ids)
      if len(input_ids) > batch_length:
        batch_length = len(input_ids)
    final_batch_inputs = []
    for input_ids in batch_inputs:
      assert len(input_ids) <= batch_length
      x = input_ids + [0] * (batch_length - len(input_ids))
      final_batch_inputs.append(x)

    yield {
        "inputs": np.array(final_batch_inputs).astype(np.int32),
    }


def _get_sorted_inputs(filename, delimiter="\n"):
  """Returning inputs sorted according to length.

  Args:
    filename: path to file with inputs, 1 per line.
    delimiter: str, delimits records in the file.

  Returns:
    a sorted list of inputs

  """
  tf.logging.info("Getting sorted inputs")
  with tf.gfile.Open(filename) as f:
    text = f.read()
    records = text.split(delimiter)
    inputs = [record.strip() for record in records]
    # Strip the last empty line.
    if not inputs[-1]:
      inputs.pop()
  input_lens = [(i, len(line.split())) for i, line in enumerate(inputs)]
  sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1))
  # We'll need the keys to rearrange the inputs back into their original order
  sorted_keys = {}
  sorted_inputs = []
  for i, (index, _) in enumerate(sorted_input_lens):
    sorted_inputs.append(inputs[index])
    sorted_keys[index] = i
  return sorted_inputs, sorted_keys


def save_until_eos(ids, skip=False):
  """Strips everything after the first <EOS> token, which is normally 1."""
  ids = ids.flatten()
  if skip:
    return ids
  try:
    index = list(ids).index(text_encoder.EOS_ID)
    return ids[0:index]
  except ValueError:
    # No EOS_ID: return the array as-is.
    return ids


def _decode_input_tensor_to_features_dict(feature_map, hparams):
  """Convert the interactive input format (see above) to a dictionary.

  Args:
    feature_map: dict with inputs.
    hparams: model hyperparameters

  Returns:
    a features dictionary, as expected by the decoder.
  """
  inputs = tf.convert_to_tensor(feature_map["inputs"])

  x = inputs
  p_hparams = hparams.problem_hparams
  # Add a third empty dimension
  x = tf.expand_dims(x, axis=[2])
  x = tf.to_int32(x)
  input_space_id = tf.constant(p_hparams.input_space_id)
  target_space_id = tf.constant(p_hparams.target_space_id)

  features = {}
  features["input_space_id"] = input_space_id
  features["target_space_id"] = target_space_id
  features["decode_length"] = tf.shape(x)[1] + 50
  features["inputs"] = x
  return features


def latest_checkpoint_step(ckpt_dir):
  ckpt = tf.train.get_checkpoint_state(ckpt_dir)
  if not ckpt:
    return None
  path = ckpt.model_checkpoint_path
  step = int(path.split("-")[-1])
  return step


class DecodeHookArgs(collections.namedtuple(
    "DecodeHookArgs",
    ["estimator", "problem", "output_dirs", "hparams",
     "decode_hparams", "predictions"])):
  pass


def run_postdecode_hooks(decode_hook_args, dataset_split):
  """Run hooks after decodes have run."""
  hooks = decode_hook_args.problem.decode_hooks
  if not hooks:
    return
  if decode_hook_args.hparams.mlperf_mode:
    # Use the step when the checkpoint was generated for clearer eval metric.
    global_step = decode_hook_args.decode_hparams.mlperf_decode_step
  else:
    global_step = latest_checkpoint_step(decode_hook_args.estimator.model_dir)
  if global_step is None:
    tf.logging.info(
        "Skipping decode hooks because no checkpoint yet available.")
    return
  tf.logging.info("Running decode hooks.")
  if decode_hook_args.hparams.write_summary:
    parent_dir = os.path.join(decode_hook_args.output_dirs[0], os.pardir)
    child_dir = decode_hook_args.decode_hparams.summaries_log_dir
    if dataset_split is not None:
      child_dir += "_{}".format(dataset_split)
    final_dir = os.path.join(parent_dir, child_dir)
    summary_writer = tf.summary.FileWriter(final_dir)

  for hook in hooks:
    # Isolate each hook in case it creates TF ops
    with tf.Graph().as_default():
      summaries = hook(decode_hook_args)
    if summaries and decode_hook_args.hparams.write_summary:
      summary = tf.Summary(value=list(summaries))
      summary_writer.add_summary(summary, global_step)
  if decode_hook_args.hparams.write_summary:
    summary_writer.close()
  tf.logging.info("Decode hooks done.")
