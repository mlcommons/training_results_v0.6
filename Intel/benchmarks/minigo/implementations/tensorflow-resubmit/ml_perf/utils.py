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

"""Utilities for the reinforcement trainer."""

import sys
sys.path.insert(0, '.')  # nopep8

import asyncio
import logging
import os
import multiprocessing
import subprocess
import fcntl

from absl import flags
from utils import *


def expand_cmd_str(cmd):
  result = ' '.join(flags.FlagValues().read_flags_from_files(cmd))
  if cmd[0] == 'mpiexec' or cmd[0] == 'mpirun':
    result = ' \\\n-host '.join(result.split(' -host '))
  # avoid buffer too big to block I/O
  return result[:8192]


def get_cmd_name(cmd):
  if cmd[0] == 'python' or cmd[0] == 'python3':
    path = cmd[1]
    for index in range(len(cmd)):
      if cmd[index] == 'bazel-bin/cc/selfplay':
        path = cmd[index]
        break
      if cmd[index] == 'bazel-bin/cc/eval':
        path = cmd[index]
        break
  elif cmd[0] == 'mpirun' or cmd[0] == 'mpiexec':
    for index in range(len(cmd)):
      if cmd[index] == 'train.py':
        path = cmd[index]
        break
      if cmd[index] == 'bazel-bin/cc/selfplay':
        path = cmd[index]
        break
      if cmd[index] == 'bazel-bin/cc/eval':
        path = cmd[index]
        break
      if cmd[index] == 'python' or cmd[index] == 'python3':
        path = cmd[index+1]
  else:
    path = cmd[0]
  return os.path.splitext(os.path.basename(path))[0]


async def checked_run(*cmd):
  """Run the given subprocess command in a coroutine.

  Args:
    *cmd: the command to run and its arguments.

  Returns:
    The output that the command wrote to stdout.

  Raises:
    RuntimeError: if the command returns a non-zero result.
  """

  # Start the subprocess.
  logging.info('Running: %s', expand_cmd_str(cmd))
  with logged_timer('{} finished'.format(get_cmd_name(cmd))):
    p = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)

    # Stream output from the process stdout.
    chunks = []
    while True:
      chunk = await p.stdout.read(16 * 1024)
      if not chunk:
        break
      chunks.append(chunk)

    # Wait for the process to finish, check it was successful & build stdout.
    await p.wait()
    stdout = b''.join(chunks).decode()[:-1]
    if p.returncode:
      raise RuntimeError('Return code {} from process: {}\n{}'.format(
          p.returncode, expand_cmd_str(cmd), stdout))

    return stdout

async def checked_run_distributed(genvs, num_instance, hosts, proclists, numa_nodes, seed, log_path, *cmd):
  mpi_cmd = ['mpiexec',
             '-outfile-pattern',
             '{}/out-{}-{}-%r.txt'.format(log_path, get_cmd_name(cmd), seed)]
  for genv in genvs:
    mpi_cmd = mpi_cmd + ['-genv', genv]
  num_nodes = len(hosts)
  instance_per_node = num_instance // num_nodes
  instance_remaining = num_instance - num_nodes * instance_per_node
  for index in range(num_nodes):
    if index < instance_remaining:
      instance_to_launch = instance_per_node + 1
    else:
      instance_to_launch = instance_per_node

    if index > 0:
      mpi_cmd = mpi_cmd + [':']
    mpi_cmd = mpi_cmd + ['-host', hosts[index]]

    if proclists != None:
      mpi_cmd = mpi_cmd + ['-env', 'KMP_AFFINITY=granularity=fine,compact,1,{}'.format(proclists[index])]

    if numa_nodes != None:
      mpi_cmd = mpi_cmd + ['numactl', '-l', '-N', numa_nodes[index]]

    if num_instance > 1:
      mpi_cmd = mpi_cmd + ['python3', 'ml_perf/execute.py',
                           '--num_instance={}'.format(instance_to_launch),
                           '--']
    mpi_cmd = mpi_cmd + [*cmd]

    if seed != None:
      # ensure different seed for different node
      mpi_cmd = mpi_cmd + ['--seed={}'.format(seed + index*1023779831)]

  result = await checked_run(*mpi_cmd)
  for index in range(num_nodes):
    filename = '{}/out-{}-{}-{}.txt'.format(log_path, get_cmd_name(cmd), seed,
                                            index)
    outfile = open(filename, 'r')
    result += outfile.read()
    outfile.close()
  return result

def checked_run_mi(num_instance, *cmd):
  name = get_cmd_name(cmd)
  logging.debug('Running %s*%d: %s', name, num_instance, expand_cmd_str(cmd))
  num_parallel_instance = int(multiprocessing.cpu_count())
  procs=[None]*num_parallel_instance
  results = [""]*num_parallel_instance
  result_list = []

  cur_instance = 0
  # add new proc into procs
  while cur_instance < num_instance or not all (
      proc is None for proc in procs):
    if None in procs and cur_instance < num_instance:
      index = procs.index(None)
      subproc_cmd = [
              'OMP_NUM_THREADS=1',
              'KMP_AFFINITY=granularity=fine,proclist=[{}],explicit'.format(
                  ','.join(str(i) for i in list(range(
                      index, index+1)))),
              *cmd,
              '--instance_id={}'.format(cur_instance),
      ]
      subproc_cmd = ' '.join(subproc_cmd)
      if (cur_instance == 0):
        logging.debug("subproc_cmd = {}".format(subproc_cmd))
      procs[index] = subprocess.Popen(subproc_cmd, shell=True,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT)

      proc_count = 0
      for i in range(num_parallel_instance):
        if procs[i] != None:
          proc_count += 1
      logging.debug('started instance {} in proc {}. proc count = {}'.format(
          cur_instance, index, proc_count))

      # change stdout of the process to non-blocking
      # this is for collect output in a single thread
      flags = fcntl.fcntl(procs[index].stdout, fcntl.F_GETFL)
      fcntl.fcntl(procs[index].stdout, fcntl.F_SETFL, flags | os.O_NONBLOCK)

      cur_instance += 1
    for index in range(num_parallel_instance):
      if procs[index] != None:
        # collect proc output
        while True:
          try:
            line = procs[index].stdout.readline()
            if line == b'':
              break
            results[index] = results[index] + line.decode()
          except IOError:
            break

        ret_val = procs[index].poll()
        if ret_val == None:
          continue
        elif ret_val != 0:
          logging.info(results[index])
          raise RuntimeError(
            'Non-zero return code (%d) executing %s' % (
                ret_val, subproc_cmd))

        if index == 0:
          logging.debug(results[index])
        result_list.append(results[index])
        results[index] = ""
        procs[index] = None

        proc_count = 0
        for i in range(num_parallel_instance):
          if procs[i] != None:
            proc_count += 1
        logging.debug('proc {} finished. proc count = {}'.format(
            index, proc_count))
    time.sleep(0.001)  # avoid busy loop
  return result_list

def wait(aws):
  """Waits for all of the awaitable objects (e.g. coroutines) in aws to finish.

  All the awaitable objects are waited for, even if one of them raises an
  exception. When one or more awaitable raises an exception, the exception from
  the awaitable with the lowest index in the aws list will be reraised.

  Args:
    aws: a single awaitable, or list awaitables.

  Returns:
    If aws is a single awaitable, its result.
    If aws is a list of awaitables, a list containing the of each awaitable in
    the list.

  Raises:
    Exception: if any of the awaitables raises.
  """

  aws_list = aws if isinstance(aws, list) else [aws]
  results = asyncio.get_event_loop().run_until_complete(asyncio.gather(
      *aws_list, return_exceptions=True))
  # If any of the cmds failed, re-raise the error.
  for result in results:
    if isinstance(result, Exception):
      raise result
  return results if isinstance(aws, list) else results[0]
