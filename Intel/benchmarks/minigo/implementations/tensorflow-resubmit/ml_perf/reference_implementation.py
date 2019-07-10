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

"""Runs a reinforcement learning loop to train a Go playing model."""

import sys
sys.path.insert(0, '.')  # nopep8

import asyncio
import glob
import logging
import numpy as np
import os
import random
import re
import shutil
import subprocess
import functools
import tensorflow as tf
import time
import copy
import multiprocessing as mp
from ml_perf.utils import *
import ml_perf.mlp_log as mll

from fractions import gcd

from absl import app, flags
from rl_loop import example_buffer, fsdb
import dual_net

from tensorflow.python.platform import gfile

N = int(os.environ.get('BOARD_SIZE', 19))

flags.DEFINE_string('checkpoint_dir', None,
                    'The checkpoint directory specify a start model and a set '
                    'of golden chunks used to start training.  If not '
                    'specified, will start from scratch.')

flags.DEFINE_string('target_path', 'ml_perf/target/{}/target.pb'.format(N),
                    'Path to the target model to beat.')

flags.DEFINE_integer('iterations', 100, 'Number of iterations of the RL loop.')

flags.DEFINE_float('gating_win_rate', 0.55,
                   'Win-rate against the current best required to promote a '
                   'model to new best.')

flags.DEFINE_string('flags_dir', None,
                    'Directory in which to find the flag files for each stage '
                    'of the RL loop. The directory must contain the following '
                    'files: bootstrap.flags, selfplay.flags, eval.flags, '
                    'train.flags.')

flags.DEFINE_integer('window_size', 10,
                     'Maximum number of recent selfplay rounds to train on.')
flags.DEFINE_integer('golden_chunk_split', 2,
                     'Golden chunk of each selfplay is splited to accelerate write golden chunk')

flags.DEFINE_integer('parallel_post_train', 0,
                     '0: run the post-training stages in serial mode'
                     '1: run the post-training stages (eval, validation '
                     '& selfplay) in parallel.'
                     '2: run the post-train stage in pipeline mode.')

flags.DEFINE_string('engine', 'tf', 'The engine to use for selfplay.')

flags.DEFINE_integer('physical_cores', None, 'The number of cores for each node.')
flags.DEFINE_integer('virtual_cores', None, 'The number of SMT for each node.')
flags.DEFINE_integer('numa_cores', None, 'The number of core for each numa node.')
flags.DEFINE_integer('train_instance_per_numa', 2, 'The number of instance for each numa node.')

flags.DEFINE_multi_string('train_node', [], 'The node:core list for training')
flags.DEFINE_multi_string('eval_node', [], 'The node list for evaluation')
flags.DEFINE_multi_string('selfplay_node', [], 'The node list for selfplay.')

flags.DEFINE_bool('quantization', True, 'Using Int8 if true.')
flags.DEFINE_bool('eval_min_max_every_epoch', True, 'Genereting min max log every epoch if true.')
flags.DEFINE_boolean('random_rotation', True, 'Do random rotation when running for min&max log.')
flags.DEFINE_integer('quantize_test_steps', 5, 'The steps to run for min&max log.')
flags.DEFINE_integer('quantize_test_batch_size', 16, 'The batch size for running inference for min&max log.')

FLAGS = flags.FLAGS

class State:
  """State data used in each iteration of the RL loop.

  Models are named with the current reinforcement learning loop iteration number
  and the model generation (how many models have passed gating). For example, a
  model named "000015-000007" was trained on the 15th iteration of the loop and
  is the 7th models that passed gating.
  Note that we rely on the iteration number being the first part of the model
  name so that the training chunks sort correctly.
  """

  def __init__(self):
    self.start_time = time.time()

    self.iter_num = 0
    self.gen_num = 0

    self.best_model_name = None

  @property
  def output_model_name(self):
    return '%06d-%06d' % (self.iter_num, self.gen_num)

  @property
  def train_model_name(self):
    return '%06d-%06d' % (self.iter_num, self.gen_num + 1)

  @property
  def best_model_path(self):
    if self.best_model_name is None:
      # We don't have a good model yet, use a random fake model implementation.
      return 'random:0,0.4:0.4'
    else:
      return '{},{}.pb'.format(
         FLAGS.engine, os.path.join(fsdb.models_dir(), self.best_model_name))

  @property
  def train_model_path(self):
    return '{},{}.pb'.format(
         FLAGS.engine, os.path.join(fsdb.models_dir(), self.train_model_name))

  @property
  def seed(self):
    return self.iter_num + 1


class ColorWinStats:
  """Win-rate stats for a single model & color."""

  def __init__(self, total, both_passed, opponent_resigned, move_limit_reached):
    self.total = total
    self.both_passed = both_passed
    self.opponent_resigned = opponent_resigned
    self.move_limit_reached = move_limit_reached
    # Verify that the total is correct
    assert total == both_passed + opponent_resigned + move_limit_reached


class WinStats:
  """Win-rate stats for a single model."""

  def __init__(self, line):
    pattern = '\s*(\S+)' + '\s+(\d+)' * 8
    match = re.search(pattern, line)
    if match is None:
      raise ValueError('Can\'t parse line "{}"'.format(line))
    self.model_name = match.group(1)
    raw_stats = [float(x) for x in match.groups()[1:]]
    self.black_wins = ColorWinStats(*raw_stats[:4])
    self.white_wins = ColorWinStats(*raw_stats[4:])
    self.total_wins = self.black_wins.total + self.white_wins.total

def initialize_from_checkpoint(state, out_files_number):
  """Initialize the reinforcement learning loop from a checkpoint."""
  # The checkpoint's work_dir should contain the most recently trained model.
  model_paths = glob.glob(os.path.join(FLAGS.checkpoint_dir,
                                       'work_dir/model.ckpt-*.pb'))
  if len(model_paths) != 1:
    raise RuntimeError('Expected exactly one model in the checkpoint work_dir, '
                       'got [{}]'.format(', '.join(model_paths)))
  start_model_path = model_paths[0]

  golden_chunks_dir = os.path.join(FLAGS.checkpoint_dir, 'golden_chunks')
  for basename in os.listdir(golden_chunks_dir):
    path = os.path.join(golden_chunks_dir, basename)
    out_path = os.path.join(fsdb.golden_chunk_dir(), basename)
    buffer = example_buffer.ExampleBuffer(sampling_frac=1.0)
    example_num = buffer.parallel_fill(tf.gfile.Glob(path),FLAGS.physical_cores)
    buffer.flush_new(out_path, example_num, out_files_number, 1)

  # Copy the latest trained model into the models directory and use it on the
  # first round of selfplay.
  state.best_model_name = 'checkpoint'
  best_model_path = os.path.join(fsdb.models_dir(), state.best_model_name)

  dual_net.optimize_graph(start_model_path, best_model_path, FLAGS.quantization, fsdb.golden_chunk_dir()+'/*.zz*', FLAGS.eval_min_max_every_epoch)

  # Copy the training files.
  work_dir = os.path.join(FLAGS.checkpoint_dir, 'work_dir')
  for basename in os.listdir(work_dir):
    path = os.path.join(work_dir, basename)
    shutil.copy(path, fsdb.working_dir())


def parse_win_stats_table(stats_str, num_lines):
  result = []
  lines = stats_str.split('\n')

  while True:
    while True:
      # Find the start of the win stats table.
      if len(lines) == 0:
        return result
      if 'Black' in lines[0] and 'White' in lines[0] and 'm.lmt.' in lines[1]:
          break
      lines = lines[1:]

    # Parse the expected number of lines from the table.
    for line in lines[2:2 + num_lines]:
      stat = WinStats(line)
      for s in result:
        if s.model_name == stat.model_name:
          s.black_wins.total += stat.black_wins.total
          s.white_wins.total += stat.white_wins.total
          s.total_wins += stat.total_wins
          stat = None
          break
      if stat != None:
        result.append(stat)
    lines = lines[2 + num_lines:]

def extract_multi_instance(cmd):
  cmd_list = flags.FlagValues().read_flags_from_files(cmd)
  new_cmd_list = []
  multi_instance = False
  num_instance = 0
  num_games = 0
  parallel_games = 0

  for arg in cmd_list:
    argsplit = arg.split('=', 1)
    flag = argsplit[0]
    if flag == '--multi_instance':
      if argsplit[1] == 'True':
        multi_instance = True
      else:
        multi_instance = False
    elif flag == '--num_games':
      num_games = int(argsplit[1])
    elif flag == '--parallel_games':
      parallel_games = int(argsplit[1])

  if multi_instance:
    if num_games % parallel_games != 0:
      logging.error('Error num_games must be multiply of %d', parallel_games)
      raise RuntimeError('incompatible num_games/parallel_games combination')
    num_instance = num_games//parallel_games

  for arg in cmd_list:
    argsplit = arg.split('=', 1)
    flag = argsplit[0]
    if flag == '--multi_instance':
      pass
    elif multi_instance and flag == '--num_games':
      pass
    else:
      new_cmd_list.append(arg)

  return multi_instance, num_instance, new_cmd_list

async def run(*cmd):
  """Run the given subprocess command in a coroutine.

  Args:
    *cmd: the command to run and its arguments.

  Returns:
    The output that the command wrote to stdout as a list of strings, one line
    per element (stderr output is piped to stdout).

  Raises:
    RuntimeError: if the command returns a non-zero result.
  """

  stdout = await checked_run(*cmd)

  log_path = os.path.join(FLAGS.base_dir, get_cmd_name(cmd) + '.log')
  with gfile.Open(log_path, 'a') as f:
    f.write(expand_cmd_str(cmd))
    f.write('\n')
    f.write(stdout)
    f.write('\n')

  # Split stdout into lines.
  return stdout.split('\n')

async def run_distributed(genvs, num_instance, hosts, proclists, numa_nodes,
                          seed, *cmd):
  """Run the given subprocess command in a coroutine.

  Args:
    *cmd: the command to run and its arguments.

  Returns:
    The output that the command wrote to stdout as a list of strings, one line
    per element (stderr output is piped to stdout).

  Raises:
    RuntimeError: if the command returns a non-zero result.
  """

  stdout = await checked_run_distributed(genvs, num_instance, hosts, proclists,
                                         numa_nodes, seed, fsdb.mpi_log_dir(), *cmd)

  log_path = os.path.join(FLAGS.base_dir, get_cmd_name(cmd) + '.log')
  with gfile.Open(log_path, 'a') as f:
    f.write(expand_cmd_str(cmd))
    f.write('\n')
    f.write(stdout)
    f.write('\n')

  # Split stdout into lines.
  return stdout.split('\n')

def get_golden_chunk_records(window_size):
  """Return up to num_records of golden chunks to train on.

  Returns:
    A list of golden chunks up to num_records in length, sorted by path.
  """

  pattern = os.path.join(fsdb.golden_chunk_dir(), '*.zz*')
  return sorted(tf.gfile.Glob(pattern), reverse=True)[:window_size]


def gen_golden_chunk(files, state):
  buffer = example_buffer.ExampleBuffer(sampling_frac=1.0)
  buffer.parallel_fill(files[1], threads=1)
  buffer.flush(os.path.join(fsdb.golden_chunk_dir(),
                            state.output_model_name + '-{}.tfrecord.zz'.format(files[0])))

# Self-play a number of games.
async def selfplay(state, flagfile='selfplay'):
  """Run selfplay and write a training chunk to the fsdb golden_chunk_dir.

  Args:
    state: the RL loop State instance.
    flagfile: the name of the flagfile to use for selfplay, either 'selfplay'
        (the default) or 'boostrap'.
  """
  output_dir = os.path.join(fsdb.selfplay_dir(), state.output_model_name)
  holdout_dir = os.path.join(fsdb.holdout_dir(), state.output_model_name)
  output_dir = '/tmp/minigo' + output_dir

  multi_instance, num_instance, flag_list = extract_multi_instance(
      ['--flagfile={}_mi.flags'.format(os.path.join(FLAGS.flags_dir, flagfile))])
  sp_cmd = ['bazel-bin/cc/selfplay',
            '--flagfile={}.flags'.format(os.path.join(FLAGS.flags_dir, flagfile)),
            '--model={}'.format(state.best_model_path),
            '--output_dir={}'.format(output_dir),
            '--holdout_dir={}'.format(holdout_dir)]
  if not multi_instance:
    lines = await run(
        *sp_cmd,
        '--seed={}'.format(state.seed))
  else:
    if FLAGS.selfplay_node == []:
      # run selfplay locally
      lines = await run(
          'python3', 'ml_perf/execute.py',
          '--num_instance={}'.format(num_instance),
          '--',
          *sp_cmd,
          '--seed={}'.format(state.seed))
    else:
      with logged_timer('selfplay mn'):
        # run one selfplay instance per host
        lines = await run_distributed(
            ['LD_LIBRARY_PATH=$LD_LIBRARY_PATH:cc/tensorflow'],
            num_instance, FLAGS.selfplay_node, None, None, state.seed,
            *sp_cmd)

  #result = '\n'.join(lines)
  #with logged_timer('parse win stats'):
  #  stats = parse_win_stats_table(result, 1)[0]
  #  num_games = stats.total_wins
  #  black_total = stats.black_wins.total
  #  white_total = stats.white_wins.total

  #  logging.info('Black won %0.3f, white won %0.3f',
  #               black_total / num_games,
  #               white_total / num_games)
  #  bias = abs(white_total - black_total)/num_games
  #  logging.info('Black total %d, white total %d, total games %d, bias %0.3f.',
  #               black_total, white_total, num_games, bias)

  with logged_timer('generate golden chunk'):
    # Write examples to a single record.
    hosts = FLAGS.selfplay_node
    if hosts == []:
      hosts = ['localhost']
    num_instance = len(hosts)
    numa_per_node = FLAGS.physical_cores // FLAGS.numa_cores
    train_instance_num = FLAGS.train_instance_per_numa * len(FLAGS.train_node) * numa_per_node
    selfplay_node_num = len(hosts)
    selfplay_num = selfplay_node_num
    out_files_number = int(train_instance_num/gcd(train_instance_num, selfplay_num))

    cmd = ['python3', 'ml_perf/divide_golden_chunk.py',
        '--read_path={}'.format(output_dir + "/*"),
        '--write_path={}'.format(os.path.join(fsdb.golden_chunk_dir(), state.output_model_name + '.tfrecord.zz')),
        '--out_files_number={}'.format(out_files_number),
        '--physical_cores={}'.format(FLAGS.physical_cores),
        '--base_dir={}'.format(FLAGS.base_dir)]
    lines = await run_distributed([], 1, hosts, None, None, state.seed, *cmd)

    #print(lines)

  # return bias

async def train(state, window_size):
  """Run training and write a new model to the fsdb models_dir.

  Args:
    state: the RL loop State instance.
  """
  train_node = FLAGS.train_node
  num_node = len(train_node)
  if num_node == 0:
    dist_train = False
  else:
    dist_train = True

  if dist_train:
    intra_threads = FLAGS.numa_cores // FLAGS.train_instance_per_numa - 1
    numa_per_node = FLAGS.physical_cores // FLAGS.numa_cores
    instance_per_node = numa_per_node * FLAGS.train_instance_per_numa

    mpi_async_progress = ''
    for i in range(numa_per_node):
      for j in range(FLAGS.train_instance_per_numa):
        if (not i==0) or (not j==0):
          mpi_async_progress += ','
        mpi_async_progress += '{}'.format(i * FLAGS.numa_cores + j)
  else:
    intra_threads = FLAGS.physical_cores

  model_path = os.path.join(fsdb.models_dir(), state.train_model_name)
  cmd = ['python3', 'train.py',
      '--flagfile={}'.format(os.path.join(FLAGS.flags_dir, 'train.flags')),
      '--work_dir={}'.format(fsdb.working_dir()),
      '--export_path={}'.format(model_path),
      '--window_size={}'.format(window_size),
      '--data_path={}'.format(fsdb.golden_chunk_dir()),
      '--training_seed={}'.format(state.seed),
      '--freeze=True',
      '--num_inter_threads=1',
      '--num_intra_threads={}'.format(intra_threads)]

  if(dist_train):
    genvs = ['HOROVOD_FUSION_THRESHOLD=134217728',
             'KMP_BLOCKTIME=0',
             'KMP_HW_SUBSET=1T',
             'OMP_BIND_PROC=true',
             'I_MPI_ASYNC_PROGRESS_PIN=' + mpi_async_progress,
             'OMP_NUM_THREADS={}'.format(intra_threads)]
    hosts = []
    proclists = []
    numa_nodes = []
    for node in range(num_node):
      # add all instance to the list
      for numa in range(numa_per_node):
        for instance in range(FLAGS.train_instance_per_numa):
          hosts += [train_node[node]]
          proclist = numa * FLAGS.numa_cores + FLAGS.train_instance_per_numa + instance * intra_threads
          proclists += ['{}'.format(proclist)]
          numa_nodes += ['{}'.format(numa)]

    lines = await run_distributed(genvs, 1, hosts, proclists, numa_nodes, None, *cmd, '--dist_train=True')
  else:
    lines = await run(*cmd)
  print('\n'.join(lines), file=sys.stderr)

def post_train(state):
  model_path = os.path.join(fsdb.models_dir(), state.train_model_name)
  dual_net.optimize_graph(model_path + '.pb', model_path, FLAGS.quantization, fsdb.golden_chunk_dir()+'/*.zz*', FLAGS.eval_min_max_every_epoch)
  mll.save_model(state.iter_num-1)

  # Append the time elapsed from when the RL was started to when this model
  # was trained.
  elapsed = time.time() - state.start_time
  timestamps_path = os.path.join(fsdb.models_dir(), 'train_times.txt')
  with gfile.Open(timestamps_path, 'a') as f:
    print('{:.3f} {}'.format(elapsed, state.train_model_name), file=f)


async def validate(state, holdout_glob):
  """Validate the trained model against holdout games.

  Args:
    state: the RL loop State instance.
    holdout_glob: a glob that matches holdout games.
  """

  if not glob.glob(holdout_glob):
    print('Glob "{}" didn\'t match any files, skipping validation'.format(
          holdout_glob))
  else:
    await run(
        'python3', 'validate.py', holdout_glob,
        '--flagfile={}'.format(os.path.join(FLAGS.flags_dir, 'validate.flags')),
        '--work_dir={}'.format(fsdb.working_dir()))


async def evaluate_model(eval_model_path, target_model_path, sgf_dir, seed, flagfile='eval'):
  """Evaluate one model against a target.

  Args:
    eval_model_path: the path to the model to evaluate.
    target_model_path: the path to the model to compare to.
    sgf_dif: directory path to write SGF output to.
    seed: random seed to use when running eval.

  Returns:
    The win-rate of eval_model against target_model in the range [0, 1].
  """

  multi_instance, num_instance, flag_list = extract_multi_instance(
      ['--flagfile={}_mi.flags'.format(os.path.join(FLAGS.flags_dir, flagfile))])
  eval_cmd = ['bazel-bin/cc/eval',
              '--flagfile={}.flags'.format(os.path.join(FLAGS.flags_dir, flagfile)),
              '--model={}'.format(eval_model_path),
              '--model_two={}'.format(target_model_path),
              '--sgf_dir={}'.format(sgf_dir)]
  if not multi_instance:
    lines = await run(*eval_cmd, '--seed={}'.format(seed))
  else:
    if FLAGS.eval_node == []:
      # run eval locally
      lines = await run(
          'python3', 'ml_perf/execute.py',
          '--num_instance={}'.format(num_instance),
          '--',
          *eval_cmd,
          '--seed={}'.format(seed))
    else:
      # run one selfplay instance per host
      lines = await run_distributed(
          ['LD_LIBRARY_PATH=$LD_LIBRARY_PATH:cc/tensorflow'],
          num_instance, FLAGS.eval_node, None, None, seed,
          *eval_cmd)
  result = '\n'.join(lines)
  #logging.info(result)
  eval_stats, target_stats = parse_win_stats_table(result, 2)
  num_games = eval_stats.total_wins + target_stats.total_wins
  win_rate = eval_stats.total_wins / num_games
  #eval_total = eval_stats.total_wins
  #black_total = eval_stats.black_wins.total
  #white_total = eval_stats.white_wins.total

  #if eval_total != 0:
  #  bias = abs(white_total - black_total) / eval_total
  #else:
  #  # by definition bias = 0.0 if eval model win zero games
  #  bias = 0.0
  logging.info('Win rate %s vs %s: %.3f', eval_stats.model_name,
               target_stats.model_name, win_rate)
  #logging.info('Black total %d, white total %d, eval total %d, bias %0.3f.',
  #             black_total, white_total, eval_total, bias)

  return win_rate


async def evaluate_trained_model(state):
  """Evaluate the most recently trained model against the current best model.

  Args:
    state: the RL loop State instance.
  """

  return await evaluate_model(
      state.train_model_path, state.best_model_path,
      os.path.join(fsdb.eval_dir(), state.train_model_name), state.seed)


async def evaluate_target_model(state):
  sgf_dir = os.path.join(fsdb.eval_dir(), 'target')
  target = 'tf,' + os.path.join(fsdb.models_dir(), 'target.pb')
  return await evaluate_model(
      state.train_model_path, target, sgf_dir, state.iter_num)

def rl_loop():
  """The main reinforcement learning (RL) loop."""

  # The 'window_size' reflect the split of golden chunk after selfplay
  # basically each selfplay generate N golden chunks instead of one to
  # accelerate write golden chunks (N determined by FLAGS.golden_chunk_slit).
  # Yet this make effective_window_size dynamic.   It should increase by N-1
  # to keep the effective window size not change.  Then increase by N if no big
  # chunk left.  Until it reach FLAGS.window_size * FLAGS.golden_chunk_split

  window_size = 0

  state = State()
  numa_per_node = FLAGS.physical_cores // FLAGS.numa_cores
  train_instance_num = FLAGS.train_instance_per_numa * len(FLAGS.train_node) * numa_per_node
  selfplay_node_num = max(len(FLAGS.selfplay_node), 1)
  selfplay_num = selfplay_node_num
  out_files_number = int(train_instance_num/gcd(train_instance_num, selfplay_num)*selfplay_node_num)
  FLAGS.golden_chunk_split = out_files_number

  window_size = out_files_number * FLAGS.window_size

  if FLAGS.checkpoint_dir != None:
    # Start from a partially trained model.
    initialize_from_checkpoint(state, out_files_number)
    window_size = len(get_golden_chunk_records(window_size))
    mll.init_stop()
    mll.run_start()
    state.start_time = time.time()
  else:
    # Play the first round of selfplay games with a fake model that returns
    # random noise. We do this instead of playing multiple games using a single
    # model bootstrapped with random noise to avoid any initial bias.
    mll.init_stop()
    mll.run_start()
    state.start_time = time.time()
    mll.epoch_start(state.iter_num)
    wait(selfplay(state, 'bootstrap'))
    window_size += FLAGS.golden_chunk_split

    # Train a real model from the random selfplay games.
    state.iter_num += 1
    wait(train(state, window_size))
    post_train(state)

    # Select the newly trained model as the best.
    state.best_model_name = state.train_model_name
    state.gen_num += 1

    # Run selfplay using the new model.
    wait(selfplay(state))
    window_size += FLAGS.golden_chunk_split
    mll.epoch_stop(state.iter_num - 1)

  first_iter = True
  state_copy = None
  model_win_rate = -1.0
  # Now start the full training loop.
  while state.iter_num <= FLAGS.iterations:
    with logged_timer('iteration time {}'.format(state.iter_num)):
      mll.epoch_start(state.iter_num)
      # Build holdout glob before incrementing the iteration number because we
      # want to run validation on the previous generation.
      holdout_glob = os.path.join(fsdb.holdout_dir(), '%06d-*' % state.iter_num,
                                  '*')

      if FLAGS.parallel_post_train == 0:
        state.iter_num += 1
        wait(train(state, window_size))
        post_train(state)
        # Run eval, validation & selfplay sequentially.
        wait(selfplay(state))
        model_win_rate = wait(evaluate_trained_model(state))
        if model_win_rate >= FLAGS.gating_win_rate:
          # Promote the trained model to the best model and increment the generation
          # number.
          state.best_model_name = state.train_model_name
          state.gen_num += 1
        mll.epoch_stop(state.iter_num - 1)
        #                               ^ compensate iter_num += 1 above

      if FLAGS.parallel_post_train == 1:
        state.iter_num += 1
        wait([train(state, window_size),
            selfplay(state)])
        post_train(state)
        # Run eval, validation & selfplay in parallel.
        model_win_rate = wait(evaluate_trained_model(state))
        if model_win_rate >= FLAGS.gating_win_rate:
          # Promote the trained model to the best model and increment the generation
          # number.
          state.best_model_name = state.train_model_name
          state.gen_num += 1
        mll.epoch_stop(state.iter_num - 1)
        #                               ^ compensate iter_num += 1 above

      if FLAGS.parallel_post_train == 2:
        state_copy = copy.copy(state)
        state.iter_num += 1
        # run training and evaluation/validation/selfplay in parallel
        # this is software pipeline-ish parallelism
        # start train[iter]
        # |   start valiation[iter-1]
        # |   wait for validation
        # |   if not first time start evaluation[iter-1]
        # |   if not first time wait for evaluation
        # |   if not first time check for promotion
        # |   start selfplay[iter]
        # |   wait selfplay
        # wait train
        train_handle = asyncio.gather(train(state, window_size), return_exceptions=True)
        if not first_iter:
          post_train(state_copy)
          model_win_rate = wait(evaluate_trained_model(state_copy))
          if model_win_rate >= FLAGS.gating_win_rate:
            # Promote the trained model to the best model
            state.best_model_name = state_copy.train_model_name
          mll.epoch_stop(state.iter_num - 1 - 1)
          #                               ^---^-- compensate iter_num += 1 above
          #                                   +-- it is actually last iteration
        else:
          first_iter = False
        wait(selfplay(state))
        asyncio.get_event_loop().run_until_complete(train_handle)
        if not first_iter:
          if model_win_rate >= FLAGS.gating_win_rate:
            # Increment the generation number.
            train_model_name_before = state.train_model_name
            state.gen_num += 1

            # Output dependency:
            # In parallel post train mode 1, there is output dependence between
            # evaluation of iteration i (gen_num++)  and train of iteration i+1
            # (use gen_num for export model path).  In parallel post train mode
            # 2 (this mode), the evluation of iteration i is postponed to
            # iteration i+1 after the training started, thus train of iteration
            # i+1 won't generate correct model name when promotion needs to
            # happen.  This part fix up the model name when evaluation decides
            # there's a promotion
            train_model_name_after = state.train_model_name
            model_paths = glob.glob(os.path.join(fsdb.models_dir(), '{}.*'.format(train_model_name_before)))
            for model in model_paths:
              logging.info('moving {} --> {}'.format(model,
                train_model_name_after.join(model.rsplit(train_model_name_before, 1))))
              shutil.copy(model, train_model_name_after.join(model.rsplit(train_model_name_before, 1)))

  # after the main loop, if parallel_post_train = 2
  # needs to print epoch_stop for last epoch
  if FLAGS.parallel_post_train == 2:
    mll.epoch_stop(state.iter_num - 1)

def main(unused_argv):
  """Run the reinforcement learning loop."""

  mll.init_start()
  print('Wiping dir %s' % FLAGS.base_dir, flush=True)
  shutil.rmtree(FLAGS.base_dir, ignore_errors=True)
  dirs = [fsdb.models_dir(), fsdb.selfplay_dir(), fsdb.holdout_dir(),
          fsdb.eval_dir(), fsdb.golden_chunk_dir(), fsdb.working_dir(),
          fsdb.mpi_log_dir()]
  for d in dirs:
    ensure_dir_exists(d);

  # Copy the flag files so there's no chance of them getting accidentally
  # overwritten while the RL loop is running.
  flags_dir = os.path.join(FLAGS.base_dir, 'flags')
  shutil.copytree(FLAGS.flags_dir, flags_dir)
  FLAGS.flags_dir = flags_dir

  # Copy the target model to the models directory so we can find it easily.
  shutil.copy(FLAGS.target_path, os.path.join(fsdb.models_dir(), 'target.pb'))

  logging.getLogger().addHandler(
      logging.FileHandler(os.path.join(FLAGS.base_dir, 'rl_loop.log')))
  formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                '%Y-%m-%d %H:%M:%S')
  for handler in logging.getLogger().handlers:
    handler.setFormatter(formatter)

  logging.info('Selfplay nodes = {}'.format(FLAGS.selfplay_node))
  logging.info('Train nodes = {}'.format(FLAGS.train_node))
  logging.info('Eval nodes = {}'.format(FLAGS.eval_node))

  with logged_timer('Total time'):
    try:
      rl_loop()
    finally:
      asyncio.get_event_loop().close()


if __name__ == '__main__':
  app.run(main)
