# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluates a directory of models against the target model."""

import sys
sys.path.insert(0, '.')  # nopep8

from tensorflow import gfile
import os
import logging

from absl import app
from reference_implementation import evaluate_model, wait
from rl_loop import fsdb

from ml_perf.utils import *
from mlperf_compliance import mlperf_log, constants

def load_train_times():
  models = []
  path = os.path.join(fsdb.models_dir(), 'train_times.txt')
  with gfile.Open(path, 'r') as f:
    for line in f.readlines():
      line = line.strip()
      if line:
        timestamp, name = line.split(' ')
        path = 'tf,' + os.path.join(fsdb.models_dir(), name + '.pb')
        models.append((float(timestamp), name, path))
  return models


def main(unused_argv):
  logging.getLogger('mlperf_compliance').propagate = False

  sgf_dir = os.path.join(fsdb.eval_dir(), 'target')
  target = 'tf,' + os.path.join(fsdb.models_dir(), 'target.pb.og')
  models = load_train_times()

  timestamp_to_log = 0
  iter_evaluated = 0

  for i, (timestamp, name, path) in enumerate(models):
    minigo_print(key=constants.EVAL_START, metadata={'epoch_num': i + 1})

    iter_evaluated += 1
    winrate = wait(evaluate_model(path+'.og', target, sgf_dir, i + 1))

    minigo_print(key=constants.EVAL_ACCURACY, value=winrate, metadata={'epoch_num': i + 1})
    minigo_print(key=constants.EVAL_STOP, metadata={'epoch_num': i + 1})

    if winrate >= 0.50:
      timestamp_to_log = timestamp
      print('Model {} beat target after {}s'.format(name, timestamp))
      break

  minigo_print(key='eval_result', metadata={'iteration': iter_evaluated, 'timestamp' : timestamp_to_log})

if __name__ == '__main__':
  app.run(main)
