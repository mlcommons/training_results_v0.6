"""Hyperparameters defining different problems.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from data_generators import problem
from layers import modalities


class TestProblem(problem.Problem):
  """Test problem."""

  def __init__(self, input_vocab_size, target_vocab_size):
    super(TestProblem, self).__init__(False, False)
    self.input_vocab_size = input_vocab_size
    self.target_vocab_size = target_vocab_size

  def hparams(self, defaults, model_hparams):
    hp = defaults
    hp.modality = {"inputs": modalities.SymbolModality,
                   "targets": modalities.SymbolModality}
    hp.vocab_size = {"inputs": self.input_vocab_size,
                     "targets": self.target_vocab_size}


def test_problem_hparams(input_vocab_size=None,
                         target_vocab_size=None,
                         model_hparams=None):
  """Problem hparams for testing model bodies."""
  p = TestProblem(input_vocab_size, target_vocab_size)
  return p.get_hparams(model_hparams)
