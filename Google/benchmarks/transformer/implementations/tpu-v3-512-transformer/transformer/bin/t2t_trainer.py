"""Train and evaluate."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from mlp_log import mlp_log
import models  # pylint: disable=unused-import
import problems as problems_lib  # pylint: disable=unused-import
from data_generators import problem  # pylint: disable=unused-import
from utils import decoding
from utils import flags as t2t_flags  # pylint: disable=unused-import
from utils import trainer_lib

flags = tf.flags
FLAGS = flags.FLAGS

# See flags.py for additional command-line flags.
flags.DEFINE_string("t2t_usr_dir", None,
                    "Path to a Python module that will be imported. The "
                    "__init__.py file should include the necessary imports. "
                    "The imported files should contain registrations, "
                    "e.g. @registry.register_model calls, that will then be "
                    "available to the t2t-trainer.")
flags.DEFINE_integer("random_seed", None, "Random seed.")
flags.DEFINE_bool("deterministic_input", False,
                  "Whether to use deterministic input data.")
flags.DEFINE_bool("broadcast_input_all_replicas", False,
                  "Whether to feed the same input to all replicas each step.")
flags.DEFINE_integer("tpu_num_shards", 8, "Number of tpu shards.")
flags.DEFINE_integer("tpu_num_shards_per_host", 8, "Number of tpu shards per "
                     "host.")
flags.DEFINE_integer("iterations_per_loop", 100,
                     "Number of iterations in a TPU training loop.")
flags.DEFINE_bool("use_tpu", True, "Whether to use TPU.")
flags.DEFINE_bool("train_with_low_level_api", False, "Whether to use low level "
                  "API for training.")
flags.DEFINE_bool("decode_with_low_level_api", False, "Whether to use low "
                  "level API for decoding evaluation.")
flags.DEFINE_bool("train_and_decode_with_low_level_api", False, "Whether to "
                  "use low level API for training and decoding evaluation.")
flags.DEFINE_integer("xla_jit_level", -1,
                     "GlobalJitLevel to use while compiling the full graph.")
# To maintain compatibility with some internal libs, we guard against these flag
# definitions possibly erroring. Apologies for the ugliness.
try:
  flags.DEFINE_string("master", "", "Address of TensorFlow master.")
  flags.DEFINE_string("output_dir", "", "Base output directory for run.")
  flags.DEFINE_string("schedule", "continuous_train_and_eval",
                      "Method of Experiment to run.")
  flags.DEFINE_integer("eval_steps", 100,
                       "Number of steps in evaluation. By default, eval will "
                       "stop after eval_steps or when it runs through the eval "
                       "dataset once in full, whichever comes first, so this "
                       "can be a very large number.")
except:  # pylint: disable=bare-except
  pass

# Google Cloud TPUs
flags.DEFINE_string("tpu_job_name", None, "TPU job name.")
flags.DEFINE_string("cloud_tpu_name", "%s-tpu" % os.getenv("USER"),
                    "Name of Cloud TPU instance to use or create.")

# Hyperparameter tuning on Cloud ML Engine
# Pass an --hparams_range to enable
flags.DEFINE_bool("autotune_maximize", True,
                  "Whether to maximize (vs. minimize) autotune_objective.")
flags.DEFINE_integer("autotune_max_trials", 10,
                     "Maximum number of tuning experiments to run.")
flags.DEFINE_integer("autotune_parallel_trials", 1,
                     "How many trials to run in parallel (will spin up this "
                     "many jobs.")
# Note than in open-source TensorFlow, the dash gets converted to an underscore,
# so access is FLAGS.job_dir.
flags.DEFINE_string("job-dir", None,
                    "DO NOT USE. Exists only for Cloud ML Engine to pass in "
                    "during hyperparameter tuning. Overrides --output_dir.")
flags.DEFINE_integer("log_step_count_steps", 100,
                     "Number of local steps after which progress is printed "
                     "out")


def set_hparams_from_args(args):
  """Set hparams overrides from unparsed args list."""
  if not args:
    return

  hp_prefix = "--hp_"
  tf.logging.info("Found unparsed command-line arguments. Checking if any "
                  "start with %s and interpreting those as hparams "
                  "settings.", hp_prefix)

  pairs = []
  i = 0
  while i < len(args):
    arg = args[i]
    if arg.startswith(hp_prefix):
      pairs.append((arg[len(hp_prefix):], args[i+1]))
      i += 2
    else:
      tf.logging.warn("Found unknown flag: %s", arg)
      i += 1

  as_hparams = ",".join(["%s=%s" % (key, val) for key, val in pairs])
  if FLAGS.hparams:
    as_hparams = "," + as_hparams
  FLAGS.hparams += as_hparams


def create_hparams():
  """Create hparams."""
  if FLAGS.use_tpu and "tpu" not in FLAGS.hparams_set:
    tf.logging.warn("Not all hyperparameter sets work on TPU. "
                    "Prefer hparams_sets with a '_tpu' suffix, "
                    "e.g. transformer_tpu, if available for your model.")
  return trainer_lib.create_hparams(FLAGS.hparams_set, FLAGS.hparams)


def create_experiment_fn():
  return trainer_lib.create_experiment_fn(
      model_name=FLAGS.model,
      problem_name=FLAGS.problem,
      data_dir=os.path.expanduser(FLAGS.data_dir),
      train_steps=FLAGS.train_steps,
      eval_steps=FLAGS.eval_steps,
      min_eval_frequency=FLAGS.local_eval_frequency,
      schedule=FLAGS.schedule,
      decode_hparams=decoding.decode_hparams(FLAGS.decode_hparams),
      eval_timeout_mins=FLAGS.eval_timeout_mins,
      use_tpu=FLAGS.use_tpu,
      train_with_low_level_api=FLAGS.train_with_low_level_api,
      decode_with_low_level_api=FLAGS.decode_with_low_level_api,
      train_and_decode_with_low_level_api=(
          FLAGS.train_and_decode_with_low_level_api),
      tpu_num_hosts=FLAGS.tpu_num_shards / FLAGS.tpu_num_shards_per_host,
      iterations_per_loop=FLAGS.iterations_per_loop,
      decode_from_file=FLAGS.decode_from_file,
      decode_to_file=FLAGS.decode_to_file,
      decode_reference=FLAGS.decode_reference)


def create_run_config(hp, output_dir=None):   # pylint: disable=unused-argument
  """Create a run config.

  Args:
    hp: model hyperparameters
    output_dir: model's output directory, defaults to output_dir flag.

  Returns:
    a run config
  """
  save_ckpt_steps = max(FLAGS.iterations_per_loop, FLAGS.local_eval_frequency)
  assert FLAGS.output_dir or FLAGS.checkpoint_path
  tpu_config_extra_kwargs = {}
  if FLAGS.tpu_job_name:
    tpu_config_extra_kwargs["tpu_job_name"] = FLAGS.tpu_job_name
  return trainer_lib.create_run_config(
      model_name=FLAGS.model,
      model_dir=output_dir or os.path.expanduser(FLAGS.output_dir),
      master=FLAGS.master,
      iterations_per_loop=FLAGS.iterations_per_loop,
      num_shards=FLAGS.tpu_num_shards,
      save_checkpoints_steps=save_ckpt_steps,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours,
      use_tpu=FLAGS.use_tpu,
      schedule=FLAGS.schedule,
      random_seed=FLAGS.random_seed,
      log_step_count_steps=FLAGS.log_step_count_steps,
      cloud_tpu_name=FLAGS.cloud_tpu_name,
      tpu_config_extra_kwargs=tpu_config_extra_kwargs,
  )


def execute_schedule(exp):
  if not hasattr(exp, FLAGS.schedule):
    raise ValueError(
        "Experiment has no method %s, from --schedule" % FLAGS.schedule)
  getattr(exp, FLAGS.schedule)()


def main(argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  if argv:
    set_hparams_from_args(argv[1:])
  hparams = create_hparams()

  mlp_log.mlperf_print(key="benchmark", value="transformer")
  trainer_lib.set_random_seed(FLAGS.random_seed)
  mlp_log.mlperf_print(key="init_start", value=None)

  exp_fn = create_experiment_fn()
  exp = exp_fn(create_run_config(hparams), hparams)
  execute_schedule(exp)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
