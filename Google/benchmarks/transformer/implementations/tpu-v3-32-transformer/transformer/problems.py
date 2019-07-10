"""Access T2T Problems."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from data_generators import all_problems
from utils import registry


def problem(name):
  return registry.problem(name)


def available():
  return sorted(registry.list_problems())


all_problems.import_modules(all_problems.ALL_MODULES)
