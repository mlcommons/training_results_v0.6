"""Models defined in T2T. Imports here force registration."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# pylint: disable=unused-import

from layers import modalities  # pylint: disable=g-import-not-at-top
from models import transformer

from utils import registry

# pylint: enable=unused-import


def model(name):
  return registry.model(name)
