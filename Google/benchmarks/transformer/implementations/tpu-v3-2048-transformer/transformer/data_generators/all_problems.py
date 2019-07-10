"""Imports for problem modules."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import re

MODULES = [
    "problem_hparams",  # pylint: disable=line-too-long
    "translate_ende",  # pylint: disable=line-too-long
    "translate_enfr",  # pylint: disable=line-too-long
]
ALL_MODULES = list(MODULES)


def _is_import_err_msg(err_str, module):
  module_pattern = "(.)?".join(["(%s)?" % m for m in module.split(".")])
  return re.match("^No module named (')?%s(')?$" % module_pattern, err_str)


def _handle_errors(errors):
  """Log out and possibly reraise errors during import."""
  if not errors:
    return
  log_all = True  # pylint: disable=unused-variable
  err_msg = "Skipped importing {num_missing} data_generators modules."
  # BEGIN GOOGLE-INTERNAL
  err_msg += (" OK if no other errors. Depend on _heavy or problem-specific "
              "py_binary targets if trying to use a module that was skipped.")
  log_all = False
  # END GOOGLE-INTERNAL
  print(err_msg.format(num_missing=len(errors)))
  for module, err in errors:
    err_str = str(err)
    if not _is_import_err_msg(err_str, module):
      print("From module %s" % module)
      raise err
    if log_all:
      print("Did not import module: %s; Cause: %s" % (module, err_str))


def import_modules(modules):
  errors = []
  for module in modules:
    try:
      importlib.import_module(module)
    except ImportError as error:
      errors.append((module, error))
  _handle_errors(errors)
