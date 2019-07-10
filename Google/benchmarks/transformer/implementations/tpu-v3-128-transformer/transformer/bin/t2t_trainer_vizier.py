"""Auto tune Transformer via Vizier."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import tensorflow as tf

from bin import t2t_trainer
from bin import vizier_flags  # pylint: disable=unused-import
from bin import vizier_utils
from utils import bleu_hook

flags = tf.flags
FLAGS = flags.FLAGS


def run_continuous_train_and_eval_with_trial_params(trial_params):
  """Runs the experiment_fn with the specified trial parameters."""

  tf.Session.reset(FLAGS.master)
  # Reset graph
  tf.reset_default_graph()

  # Create hparams and override them with the trial specific ones.
  hp = t2t_trainer.create_hparams()
  hp.override_from_dict(trial_params.hparams.values())

  # Create a run_config.
  run_config = t2t_trainer.create_run_config(hp, trial_params.output_dir)

  # The actual experiment.
  exp_fn = t2t_trainer.create_experiment_fn()
  experiment = exp_fn(run_config, hp)

  # Run the experiment.
  predictions, max_steps = experiment.train_and_decode()

  # Compute the objective result (BLEU)
  outputs, references = [], []
  for output, reference in predictions:
    outputs.append(output)
    references.append(reference)
  objective_result = 100 * bleu_hook.bleu_wrapper(references, outputs)

  # Report the objective.
  tf.logging.info('BLEU = %f, step = %d', objective_result, max_steps)
  trial_params.report_fn(objective_result, max_steps)


def main(argv):
  if not FLAGS.autotune:
    return t2t_trainer.main(argv)

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.set_random_seed(123)

  vizier_utils.run_vizier(
      FLAGS.hparams_range,
      FLAGS.output_dir,
      client_handle=FLAGS.client_handle,
      experiment_fn=run_continuous_train_and_eval_with_trial_params)


if __name__ == '__main__':
  app.run(main)
