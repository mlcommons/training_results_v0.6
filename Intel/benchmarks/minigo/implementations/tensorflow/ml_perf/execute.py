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

"""Run the command in multi-instance mode

If there is a --seed parameter from input, change seed to generate randomness among instances

Args:
  num_instance: the number of instance needed to start
"""

import sys
sys.path.insert(0, '.')  # nopep8

import asyncio
from ml_perf.utils import *

from absl import app, flags

flags.DEFINE_integer('num_instance', 1, 'Number of instances for selfplay')

FLAGS = flags.FLAGS

# Self-play a number of games.
async def do_execute_mi():

  num_instance = FLAGS.num_instance

  start_copy = False
  arg_list = []
  for arg in sys.argv:
    if start_copy:
      arg_list.append(arg)
    if arg == '--':
      start_copy = True

  if num_instance > 1:
    result_list = checked_run_mi(
      num_instance,
      *arg_list
    )
    for result in result_list:
      # TODO needs to be more generic
      print ('\n'.join(result.split('\n')[-7:]))
  else:
    result = checked_run(
      *arg_list
    )
    print (result)

def main(unused_argv):
  try:
    wait(do_execute_mi())
  finally:
    asyncio.get_event_loop().close()

if __name__ == '__main__':
  app.run(main)
