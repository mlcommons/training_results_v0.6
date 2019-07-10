# Copyright 2019 MLBenchmark Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for compliance logging."""

import logging
import time
import inspect
import sys

def init_start():
  log('init_start', caller_depth=3)

def init_stop():
  log('init_stop', caller_depth=3)

def run_start():
  log('run_start', caller_depth=3)

def run_stop(status):
  assert status == 'success' or status == 'aborted'
  log('run_stop',
      meta_data = {'status':status},
      caller_depth=3)

def block_start(epoch, count):
  log('block_start',
      meta_data = {'first_epoch_num':epoch,
                   'epoch_count':count},
      caller_depth=3)

def block_stop(epoch):
  log('block_stop',
      meta_data = {'first_epoch_num':epoch},
      caller_depth=3)

def epoch_start(epoch):
  log('epoch_start',
      meta_data = {'epoch_num':epoch},
      caller_depth=3)

def epoch_stop(epoch):
  log('epoch_stop',
      meta_data = {'epoch_num':epoch},
      caller_depth=3)

def eval_start(epoch):
  log('eval_start',
      meta_data = {'epoch_num':epoch},
      caller_depth=3)

def eval_stop(epoch):
  log('eval_stop',
      meta_data = {'epoch_num':epoch},
      caller_depth=3)

def eval_accuracy(epoch, accuracy):
  log('eval_accuracy',
      val = '{}'.format(accuracy),
      meta_data = {'epoch_num':epoch},
      caller_depth=3)

def global_batch_size(batch_size):
  log('global_batch_size',
      val = '{}'.format(batch_size),
      caller_depth=3)

def lr_rates(rates):
  log('opt_base_learning_rate',
      val = '{}'.format(rates),
      caller_depth=3)

def lr_boundaries(boundaries):
  log('opt_learning_rate_decay_boundary_steps',
      val = '{}'.format(boundaries),
      caller_depth=3)

def save_model(iteration):
  log('save_model',
      meta_data = {'iteration':iteration},
      caller_depth=3)

def eval_result(iteration, timestamp):
  log('eval_result',
      meta_data = {'iteration':iteration, 'timestamp':timestamp},
      caller_depth=3)

def log(key, val='null', meta_data = None, caller_depth=2):
  filename, lineno = get_caller(caller_depth)
  meta_dict = {'lineno': lineno, 'file': filename}
  if meta_data != None:
    meta_dict.update(meta_data)
  meta_string = '{}'.format(meta_dict)
  print(':::MLL %f %s: {"value": %s, "metadata": %s}'%(time.time(), key, val, meta_string), file=sys.stderr)

def get_caller(stack_index=2, root_dir=None):
  ''' Returns file.py:lineno of your caller. A stack_index of 2 will provide
      the caller of the function calling this function. Notice that stack_index
      of 2 or more will fail if called from global scope. '''
  caller = inspect.getframeinfo(inspect.stack()[stack_index][0])

  # Trim the filenames for readability.
  filename = caller.filename
  if root_dir is not None:
    filename = re.sub("^" + root_dir + "/", "", filename)
  return (filename, caller.lineno)
