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
import tensorflow as tf
import time

from ml_perf.utils import *
from mlperf_compliance import mlperf_log, constants

from absl import app, flags
from rl_loop import example_buffer, fsdb
from tensorflow import gfile

##-->MPI
from mpi4py import MPI
import socket

N = int(os.environ.get('BOARD_SIZE', 19))

flags.DEFINE_string('checkpoint_dir', 'ml_perf/checkpoint/{}'.format(N),
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

flags.DEFINE_boolean('use_mgpu_horovod', False, 'If true run horovod based multi-gpu training.')
flags.DEFINE_boolean('use_multinode', False, 'If true run minigo on multinode, 1-node train and others self-play.')
flags.DEFINE_integer('train_rank', 0, 'MPI rank to put train on in a multinode configuration.')
flags.DEFINE_string('shared_dir_exchange', '/opt/reinforcement/minigo/piazza/', 'Shared FS dir for multinode file exchange.')
flags.DEFINE_integer('num_parallel_eval', 1, 'Chop eval into these many sub-processes.')
flags.DEFINE_integer('num_gpus_train', 1, 'Number of gpus to use for multi-gpu training with horovod.')
flags.DEFINE_string('engine', 'trt:1024', 'The engine to use for selfplay.')
flags.DEFINE_string('eval_engine', 'tf', 'The engine to use for eval.')

flags.DEFINE_integer('num_gpus_selfplay', 1,
                     'Number of GPU to use for selfplay.')

flags.DEFINE_integer('num_socket', 1,
                     'Number of socket.')

flags.DEFINE_integer('cores_per_socket', 20,
                     'Physical cores per socket')

flags.DEFINE_integer('trt_batch', 0,
                     'Batch size to create TRT graph. 0 means disable')

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
  def best_model_path_eval(self):
    return '{},{}.pb.og'.format(
         FLAGS.eval_engine, os.path.join(fsdb.models_dir(), self.best_model_name))

  @property
  def train_model_path_eval(self):
    return '{},{}.pb.og'.format(
         FLAGS.eval_engine, os.path.join(fsdb.models_dir(), self.train_model_name))

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
      raise ValueError('Can\t parse line "{}"'.format(line))
    self.model_name = match.group(1)
    raw_stats = [float(x) for x in match.groups()[1:]]
    self.black_wins = ColorWinStats(*raw_stats[:4])
    self.white_wins = ColorWinStats(*raw_stats[4:])
    self.total_wins = self.black_wins.total + self.white_wins.total

def get_golden_chunk_records(state, mpi_size=1):
  """Return up to num_records of golden chunks to train on.

  Returns:
    A list of golden chunks up to num_records in length, sorted by path.
  """

  ##how many selfplay nodes, do we fetch data from?
  num_selfplays = 1 if mpi_size == 1 else (mpi_size - 1)
  if state.iter_num <= FLAGS.window_size:
    win_size=(state.iter_num)*num_selfplays + (FLAGS.window_size-state.iter_num)
  else:
    win_size=(FLAGS.window_size)*num_selfplays
  print('Train get_golden_chunks at iter = {} has win_size = {}'.format(state.iter_num, win_size))

  pattern = os.path.join(fsdb.golden_chunk_dir(), '*.zz*')
  return sorted(tf.gfile.Glob(pattern), reverse=True)[:win_size*FLAGS.num_gpus_train]

def initialize_from_checkpoint(state):
  """Initialize the reinforcement learning loop from a checkpoint."""

  # The checkpoint's work_dir should contain the most recently trained model.
  model_paths = glob.glob(os.path.join(FLAGS.checkpoint_dir,
                                       'work_dir/model.ckpt-*.pb'))
  if len(model_paths) != 1:
    raise RuntimeError('Expected exactly one model in the checkpoint work_dir, '
                       'got [{}]'.format(', '.join(model_paths)))
  start_model_path = model_paths[0]

  # Copy the latest trained model into the models directory and use it on the
  # first round of selfplay.
  state.best_model_name = 'checkpoint'
  shutil.copy(start_model_path,
              os.path.join(fsdb.models_dir(), state.best_model_name + '.pb'))
  shutil.copy(start_model_path+'.og',
              os.path.join(fsdb.models_dir(), state.best_model_name + '.pb.og'))


  # Copy the training chunks.
  golden_chunks_dir = os.path.join(FLAGS.checkpoint_dir, 'golden_chunks')
  for basename in os.listdir(golden_chunks_dir):
    path = os.path.join(golden_chunks_dir, basename)
    out_path = os.path.join(fsdb.golden_chunk_dir(), basename)
    buffer = example_buffer.ExampleBuffer(sampling_frac=1.0)
    buffer.parallel_fill(tf.gfile.Glob(path))
    buffer.flush(out_path, FLAGS.num_gpus_train)

  # Copy the training files.
  work_dir = os.path.join(FLAGS.checkpoint_dir, 'work_dir')
  for basename in os.listdir(work_dir):
    path = os.path.join(work_dir, basename)
    shutil.copy(path, fsdb.working_dir())

def parse_win_stats_table(stats_str, num_lines):
  result = []
  lines = stats_str.split('\n')
  while True:
    # Find the start of the win stats table.
    assert len(lines) > 1
    if 'Black' in lines[0] and 'White' in lines[0] and 'm.lmt.' in lines[1]:
        break
    lines = lines[1:]

  # Parse the expected number of lines from the table.
  for line in lines[2:2 + num_lines]:
    result.append(WinStats(line))

  return result


async def run(env, *cmd):
  """Run the given subprocess command in a coroutine.

  Args:
    *cmd: the command to run and its arguments.

  Returns:
    The output that the command wrote to stdout as a list of strings, one line
    per element (stderr output is piped to stdout).

  Raises:
    RuntimeError: if the command returns a non-zero result.
  """

  stdout = await checked_run(env, *cmd)

  log_path = os.path.join(FLAGS.base_dir, get_cmd_name(cmd) + '.log')
  with gfile.Open(log_path, 'a') as f:
    f.write(expand_cmd_str(cmd))
    f.write('\n')
    f.write(stdout)
    f.write('\n')

  # Split stdout into lines.
  return stdout.split('\n')

# Self-play sub-process.
async def selfplay_sub(state, output_dir, holdout_dir, flagfile, worker_id):
  """Run selfplay and write a training chunk to the fsdb golden_chunk_dir.

  Args:
    state: the RL loop State instance.
    flagfile: the name of the flagfile to use for selfplay, either 'selfplay'
        (the default) or 'boostrap'.
  """
  # set device
  new_env = os.environ.copy()
  new_env['CUDA_VISIBLE_DEVICES'] = str(worker_id % FLAGS.num_gpus_selfplay)

  # set cpu affinity
  # todo: switch to mpi?
  # hyper threading(*2), 2 worker per gpu(/2) removed from equation
  if FLAGS.use_multinode:
    ##convert is running in parallel with eval, and val
    ##HT is on here.
    cores_per_worker = (FLAGS.cores_per_socket * FLAGS.num_socket) // FLAGS.num_gpus_selfplay
    start = (cores_per_worker*worker_id)
    end = start + cores_per_worker - 1
  else:
    cores_per_worker = (FLAGS.cores_per_socket * FLAGS.num_socket - FLAGS.num_gpus_train * 2) // FLAGS.num_gpus_selfplay
    worker_per_bucket = FLAGS.num_gpus_selfplay // FLAGS.num_socket
    bucket_id = worker_id // worker_per_bucket
    local_id = worker_id % worker_per_bucket
    start = (FLAGS.cores_per_socket * bucket_id) + (FLAGS.num_gpus_train * 2 // FLAGS.num_socket) + (local_id * cores_per_worker)
    end = start + cores_per_worker - 1
  ##cpu-str
  cpus = str(start)+'-'+str(end)

  # set seed
  base_seed = state.seed * FLAGS.num_gpus_selfplay * 2
  seed = base_seed + worker_id

  membind = (worker_id % FLAGS.num_gpus_selfplay) // (FLAGS.num_gpus_selfplay // FLAGS.num_socket)

  lines = await run(
    new_env,
    'numactl',
    '--physcpubind={}'.format(cpus),
    '--membind={}'.format(membind),
    'bazel-bin/cc/selfplay',
    '--flagfile={}.flags'.format(os.path.join(FLAGS.flags_dir, flagfile)),
    '--model={}'.format(state.best_model_path),
    '--output_dir={}'.format(output_dir),
    '--holdout_dir={}'.format(holdout_dir),
    '--seed={}'.format(seed))
  return lines

def divide_record(state, pattern, num_out, rank):
  if rank < 0:
      rank_str = ''
  else:
      rank_str = '-mpirank-' + str(rank)
  buffer = example_buffer.ExampleBuffer(sampling_frac=1.0)
  buffer.parallel_fill(tf.gfile.Glob(pattern))
  output = os.path.join(fsdb.golden_chunk_dir(),
                        state.output_model_name + rank_str + '.tfrecord.zz')
  buffer.flush(output, num_out)

  if rank >= 0:
    ##put files to exchange
    output = output + '*'
    put_files_exchange(state, rank, fileout=output)
  return

# Self-play a number of games with multiple sub process.
async def selfplay(state, flagfile='selfplay'):
  """Run selfplay and write a training chunk to the fsdb golden_chunk_dir.

  Args:
    state: the RL loop State instance.
    flagfile: the name of the flagfile to use for selfplay, either 'selfplay'
        (the default) or 'boostrap'.
  """

  output_dir = os.path.join(fsdb.selfplay_dir(), state.output_model_name)
  holdout_dir = os.path.join(fsdb.holdout_dir(), state.output_model_name)

  # instead of 2 workers in 1 process per device, we do 2 processes with 1 worker
  all_tasks = []
  loop = asyncio.get_event_loop()
  for i in range(FLAGS.num_gpus_selfplay * 2): # 2 worker per device
    all_tasks.append(loop.create_task(selfplay_sub(state, output_dir, holdout_dir, flagfile, i)))
  all_lines = await asyncio.gather(*all_tasks, return_exceptions=True)

  black_wins_total = white_wins_total = num_games = 0
  for lines in all_lines:
    if type(lines) == RuntimeError or type(lines) == OSError:
      raise lines
      continue
    result = '\n'.join(lines[-6:])
    logging.info(result)
    stats = parse_win_stats_table(result, 1)[0]
    num_games += stats.total_wins
    black_wins_total += stats.black_wins.total
    white_wins_total += stats.white_wins.total

  logging.info('Black won %0.3f, white won %0.3f',
               black_wins_total / num_games,
               white_wins_total / num_games)

  # Write examples to a single record.
  pattern = os.path.join(output_dir, '*', '*.zz')
  random.seed(state.seed)
  tf.set_random_seed(state.seed)
  np.random.seed(state.seed)
  logging.info('Writing golden chunk from "{}"'.format(pattern))
  if FLAGS.use_multinode:
    mpi_rank = MPI.COMM_WORLD.Get_rank()
    divide_record(state, pattern, FLAGS.num_gpus_train, mpi_rank)
  else:
    divide_record(state, pattern, FLAGS.num_gpus_train, -1)

# Self-play a number of games with multiple sub process.
def selfplay_noasync(state, flagfile='selfplay'):
  """Run selfplay and write a training chunk to the fsdb golden_chunk_dir.

  Args:
    state: the RL loop State instance.
    flagfile: the name of the flagfile to use for selfplay, either 'selfplay'
        (the default) or 'boostrap'.
  """

  output_dir = os.path.join(fsdb.selfplay_dir(), state.output_model_name)
  holdout_dir = os.path.join(fsdb.holdout_dir(), state.output_model_name)
  base_seed = state.seed * FLAGS.num_gpus_selfplay * 2

  if FLAGS.use_multinode:
    mpi_rank = MPI.COMM_WORLD.Get_rank()
    base_seed = base_seed + (mpi_rank * 1433)

  mpi_info = MPI.Info.Create()
  num_workers = 2*FLAGS.num_gpus_selfplay
  cores_per_worker = (FLAGS.cores_per_socket * FLAGS.num_socket) // num_workers

  # TODO: set hosts to self play nodes here.
  mpi_info.Set("host", socket.gethostname())
  mpi_info.Set("bind_to", "none")
  icomm = MPI.COMM_SELF.Spawn("ompi_bind_DGX1.sh", maxprocs=num_workers,
                              args=['bazel-bin/cc/selfplay_mpi',
                                    '--flagfile={}.flags'.format(os.path.join(FLAGS.flags_dir, flagfile)),
                                    '--model={}'.format(state.best_model_path),
                                    '--output_dir={}'.format(output_dir),
                                    '--holdout_dir={}'.format(holdout_dir),
                                    '--seed={}'.format(base_seed)],
                              info=mpi_info)

  icomm.barrier()
  icomm.Disconnect()

  black_wins_total = white_wins_total = num_games = 0

  #for lines in all_lines:
  #  if type(lines) == RuntimeError or type(lines) == OSError:
  #    raise lines
  #    continue
  #  result = '\n'.join(lines[-6:])
  #  logging.info(result)
  #  stats = parse_win_stats_table(result, 1)[0]
  #  num_games += stats.total_wins
  #  black_wins_total += stats.black_wins.total
  #  white_wins_total += stats.white_wins.total

  #logging.info('Black won %0.3f, white won %0.3f',
  #             black_wins_total / num_games,
  #             white_wins_total / num_games)

  # Write examples to a single record.
  pattern = os.path.join(output_dir, '*', '*.zz')
  random.seed(state.seed)
  tf.set_random_seed(state.seed)
  np.random.seed(state.seed)

  logging.info('Writing golden chunk from "{}"'.format(pattern))
  if FLAGS.use_multinode:
    mpi_rank = MPI.COMM_WORLD.Get_rank()
    divide_record(state, pattern, FLAGS.num_gpus_train, mpi_rank)
  else:
    divide_record(state, pattern, FLAGS.num_gpus_train, -1)

##spawn new work
def spawn_train_workers(state):
  # need to be removed
  tf_records = get_golden_chunk_records(state)
  comm_world = MPI.COMM_WORLD

  # spawn one worker process
  print("Spawning worker processes on {}".format(socket.gethostname()))
  mpi_info = MPI.Info.Create()
  num_workers = FLAGS.num_gpus_train
  # subtract 1 core from this value, oversubscription might not work
  cores_per_worker = (FLAGS.cores_per_socket * FLAGS.num_socket) // num_workers - 1

  mpi_info.Set("host", socket.gethostname())
  mpi_info.Set("map_by", "ppr:{}:socket,PE={}".format(num_workers // FLAGS.num_socket, cores_per_worker))
  icomm = MPI.COMM_SELF.Spawn("python3", maxprocs=num_workers,
                              args=['train.py', *tf_records,
                                    '--flagfile={}'.format(os.path.join(FLAGS.flags_dir, 'train.flags')),
                                    '--work_dir={}'.format(fsdb.working_dir()),
                                    '--export_path={}'.format(os.path.join(fsdb.models_dir(), 'new_model')),
                                    '--training_seed=13337',
                                    '--num_selfplays={}'.format(comm_world.size - 1),
                                    '--window_iters={}'.format(FLAGS.window_size),
                                    '--total_iters={}'.format(FLAGS.iterations),
                                    '--golden_chunk_pattern={}'.format(os.path.join(fsdb.golden_chunk_dir(), '*.zz*')),
                                    '--freeze=true',
                                    '--use_multinode=true',
                                    '--use_mgpu_horovod=true'],
                              info=mpi_info)
  return icomm

async def train(state, tf_records):
  """Run training and write a new model to the fsdb models_dir.

  Args:
    state: the RL loop State instance.
    tf_records: a list of paths to TensorFlow records to train on.
  """
  new_env = os.environ.copy()
  model_path = os.path.join(fsdb.models_dir(), state.train_model_name)

  if FLAGS.use_mgpu_horovod:
    # assign leading cores of sockets to train
    await run(
      new_env,
      'mpiexec', '--allow-run-as-root',
      '--map-by', 'ppr:{}:socket,pe=2'.format(str(FLAGS.num_gpus_train//FLAGS.num_socket)),
      '-np', str(FLAGS.num_gpus_train),
      'python3', 'train.py', *tf_records,
      '--flagfile={}'.format(os.path.join(FLAGS.flags_dir, 'train.flags')),
      '--work_dir={}'.format(fsdb.working_dir()),
      '--export_path={}'.format(model_path),
      '--training_seed={}'.format(state.seed),
      '--use_mgpu_horovod=true',
      '--freeze=true')
  else:
    new_env['CUDA_VISIBLE_DEVICES'] = '0'
    await run(
      new_env,
      'python3', 'train.py', *tf_records,
      '--flagfile={}'.format(os.path.join(FLAGS.flags_dir, 'train.flags')),
      '--work_dir={}'.format(fsdb.working_dir()),
      '--export_path={}'.format(model_path),
      '--training_seed={}'.format(state.seed),
      '--freeze=true')

  minigo_print(key='save_model', value={'iteration': state.iter_num})

  # Append the time elapsed from when the RL was started to when this model
  # was trained.
  elapsed = time.time() - state.start_time
  timestamps_path = os.path.join(fsdb.models_dir(), 'train_times.txt')
  with gfile.Open(timestamps_path, 'a') as f:
    print('{:.3f} {}'.format(elapsed, state.train_model_name), file=f)

async def convert(state):
  """Freeze the trained model and convert to TRT.

  Args:
    state: the RL loop State instance.
  """
  # set to use only second from last GPU
  new_env = os.environ.copy()
  new_env['CUDA_VISIBLE_DEVICES'] = str(FLAGS.num_gpus_train - 2)

  model_path = os.path.join(fsdb.models_dir(), state.train_model_name)
  if FLAGS.use_multinode:
    ##convert is running in parallel with eval, and val
    cores_per_worker = (FLAGS.cores_per_socket * FLAGS.num_socket) // FLAGS.num_gpus_selfplay
    start = (FLAGS.num_gpus_train - 2) * cores_per_worker
    end = start + cores_per_worker - 1
    ##cpu-str
    cpus = str(start)+'-'+str(end)
    await run(
      new_env,
      'taskset',
      '-c',
      cpus,
      'python3', 'freeze_graph.py',
      '--model_path={}'.format(model_path),
      '--trt_batch={}'.format(FLAGS.trt_batch))

  else:
    await run(
      new_env,
      'python3', 'freeze_graph.py',
      '--model_path={}'.format(model_path),
      '--trt_batch={}'.format(FLAGS.trt_batch))

def convert_noasync(state):
  """Freeze the trained model and convert to TRT.

  Args:
    state: the RL loop State instance.
  """

  model_path = os.path.join(fsdb.models_dir(), state.train_model_name)

  mpi_info = MPI.Info.Create()
  mpi_info.Set("host", socket.gethostname())
  mpi_info.Set("bind_to", "none")
  icomm = MPI.COMM_SELF.Spawn("python3", maxprocs=1,
                              args=['freeze_graph_mpi.py',
                                    '--model_path={}'.format(model_path),
                                    '--trt_batch={}'.format(FLAGS.trt_batch)],
                              info=mpi_info)
  #icomm.barrier()
  #icomm.Disconnect()

  return icomm

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
  # set to use only fisrt half of machine
  num_gpus_eval = FLAGS.num_gpus_train // 2
  gpus = '0'
  for i in range(num_gpus_eval-1):
    gpus = gpus + ',' + str(i+1)

  new_env = os.environ.copy()
  new_env['CUDA_VISIBLE_DEVICES'] = gpus

  start = 0
  cpus = str(start)+'-'+str(start + FLAGS.cores_per_socket - 1)
  start = start + FLAGS.cores_per_socket * FLAGS.num_socket
  cpus = cpus+','+str(start)+'-'+str(start + FLAGS.cores_per_socket - 1)

  membind = 0

  lines = await run(
    new_env,
    './w.sh',
    'numactl',
    '--physcpubind={}'.format(cpus),
    '--membind={}'.format(membind),
    'bazel-bin/cc/eval',
    '--flagfile={}.flags'.format(os.path.join(FLAGS.flags_dir, flagfile)),
    '--model={}'.format(eval_model_path),
    '--model_two={}'.format(target_model_path),
    '--sgf_dir={}'.format(sgf_dir),
    '--seed={}'.format(seed))
  result = '\n'.join(lines[-7:])
  logging.info(result)
  eval_stats, target_stats = parse_win_stats_table(result, 2)
  num_games = eval_stats.total_wins + target_stats.total_wins
  win_rate = eval_stats.total_wins / num_games
  logging.info('Win rate %s vs %s: %.3f', eval_stats.model_name,
               target_stats.model_name, win_rate)
  return win_rate

def evaluate_model_noasync(eval_model_path, target_model_path, sgf_dir, seed, flagfile='eval'):
  """Evaluate one model against a target.

  Args:
    eval_model_path: the path to the model to evaluate.
    target_model_path: the path to the model to compare to.
    sgf_dif: directory path to write SGF output to.
    seed: random seed to use when running eval.

  Returns:
    The win-rate of eval_model against target_model in the range [0, 1].
  """

  # hard coded affinity for now
  mpi_info = MPI.Info.Create()
  mpi_info.Set("host", socket.gethostname())
  mpi_info.Set("map_by", "ppr:2:socket,PE=10")
  icomm = MPI.COMM_SELF.Spawn("bazel-bin/cc/eval_mpi", maxprocs=4,
                              args=['--flagfile={}.flags'.format(os.path.join(FLAGS.flags_dir, flagfile)),
                                    '--model={}'.format(eval_model_path),
                                    '--model_two={}'.format(target_model_path),
                                    '--sgf_dir={}'.format(sgf_dir),
                                    '--seed={}'.format(seed)],
                              info=mpi_info)

  winrates = np.zeros(4, dtype=np.float32)
  icomm.Gather(None, winrates, root=MPI.ROOT)
  icomm.barrier()
  icomm.Disconnect()

  #result = '\n'.join(lines[-7:])
  #logging.info(result)
  #eval_stats, target_stats = parse_win_stats_table(result, 2)
  #print("eval_stats.total_wins: {}".format(eval_stats.total_wins))
  #print("target_stats.total_wins: {}".format(target_stats.total_wins))
  #
  #num_games = eval_stats.total_wins + target_stats.total_wins
  #win_rate = eval_stats.total_wins / num_games
  #logging.info('Win rate %s vs %s: %.3f', eval_stats.model_name,
  #             target_stats.model_name, win_rate)
  win_rate = sum(winrates) / 4.
  print("in python, win_rate {}".format(win_rate))
  logging.info('Win rate %s vs %s: %.3f', eval_model_path, target_model_path, win_rate)
  sys.stdout.flush()
  return win_rate


async def evaluate_model_parallel(eval_model_path, target_model_path, sgf_dir, seed, worker_id=0, flagfile='eval'):
  """Evaluate one model against a target.

  Args:
    eval_model_path: the path to the model to evaluate.
    target_model_path: the path to the model to compare to.
    sgf_dif: directory path to write SGF output to.
    seed: random seed to use when running eval.

  Returns:
    The win-rate of eval_model against target_model in the range [0, 1].
  """

  new_env = os.environ.copy()
  # fix cpu/gpu setting now
  if worker_id == 0:
    new_env['CUDA_VISIBLE_DEVICES'] = '0'
    cpus = '0-19'
  elif worker_id == 1:
    new_env['CUDA_VISIBLE_DEVICES'] = '1'
    cpus = '40-59'
  elif worker_id == 2:
    new_env['CUDA_VISIBLE_DEVICES'] = '4'
    cpus = '20-39'
  elif worker_id == 3:
    new_env['CUDA_VISIBLE_DEVICES'] = '5'
    cpus = '60-79'

  # set seed
  seed = seed + worker_id

  lines = await run(new_env,
      'taskset',
      '-c',
      cpus,
      'bazel-bin/cc/eval',
      '--flagfile={}.flags'.format(os.path.join(FLAGS.flags_dir, flagfile)),
      '--model={}'.format(eval_model_path),
      '--model_two={}'.format(target_model_path),
      '--sgf_dir={}'.format(sgf_dir),
      '--seed={}'.format(seed))
  result = '\n'.join(lines[-7:])
  logging.info(result)
  eval_stats, target_stats = parse_win_stats_table(result, 2)
  num_games = eval_stats.total_wins + target_stats.total_wins
  return (num_games, eval_stats.total_wins)

async def evaluate_trained_model(state):
  """Evaluate the most recently trained model against the current best model.

  Args:
    state: the RL loop State instance.
  """

  return await evaluate_model(
      state.train_model_path_eval, state.best_model_path_eval,
      os.path.join(fsdb.eval_dir(), state.train_model_name), state.seed)

def evaluate_trained_model_noasync(state):
  """Evaluate the most recently trained model against the current best model.

  Args:
    state: the RL loop State instance.
  """

  return evaluate_model_noasync(
      state.train_model_path_eval, state.best_model_path_eval,
      os.path.join(fsdb.eval_dir(), state.train_model_name), state.seed, 'parallel_eval')


async def evaluate_trained_model_parallel(state):
  """Evaluate the most recently trained model against the current best model.

  Args:
    state: the RL loop State instance.
  """
  all_tasks = []
  loop = asyncio.get_event_loop()
  for i in range(FLAGS.num_parallel_eval):
    all_tasks.append(loop.create_task(evaluate_model_parallel(state.train_model_path_eval, state.best_model_path_eval,
                                                              os.path.join(fsdb.eval_dir(), state.train_model_name), state.seed, i, 'parallel_eval')))
  all_lines = await asyncio.gather(*all_tasks, return_exceptions=True)

  total_games = 0
  total_wins = 0
  for lines in all_lines:
    if type(lines) == RuntimeError or type(lines) == OSError:
      raise lines
    total_games = total_games + lines[0]
    total_wins  = total_wins  + lines[1]

  print('Iter = {} Eval of {} against best={}, games={}, wins={}'.format(state.iter_num, state.train_model_name, state.best_model_name, total_games, total_wins))

  win_rate = total_wins/total_games
  return win_rate


async def train_eval_convert(state, tf_records):
  await train(state, tf_records)

  loop = asyncio.get_event_loop()
  eval_task = loop.create_task(evaluate_trained_model(state))
  convert_task = loop.create_task(convert(state))

  model_win_rate = await eval_task
  await convert_task

  return model_win_rate

async def eval_convert(state):
  loop = asyncio.get_event_loop()
  #eval_task = loop.create_task(evaluate_trained_model(state))
  eval_task = loop.create_task(evaluate_trained_model_parallel(state))
  convert_task = loop.create_task(convert(state))
  model_win_rate = await eval_task
  await convert_task
  return model_win_rate

async def eval(state):
  loop = asyncio.get_event_loop()
  eval_task = loop.create_task(evaluate_trained_model(state))
  model_win_rate = await eval_task
  await eval_task
  return model_win_rate

def eval_noasync(state):
  model_win_rate = evaluate_trained_model_noasync(state)
  return model_win_rate

def set_env_variables():
    os.environ['HOROVOD_CYCLE_TIME'] = '0.1'
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE'] = '1'
    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'

def put_files_exchange(state, mpi_rank, fileout=None):
    dst_dir  = os.path.join(FLAGS.shared_dir_exchange)

    if mpi_rank == FLAGS.train_rank:
        ##copy model_path to shared FS
        model_files = glob.glob(os.path.join(fsdb.models_dir(), state.train_model_name + '.pb*'))
        for src_file in model_files:
          print('Rank = {}, Putting file={} iter={} to SharedFS'.format(mpi_rank, src_file, state.iter_num))
          shutil.copy(src_file, dst_dir)

        ##train times as well
        #src_file = os.path.join(fsdb.models_dir(), 'train_times.txt')
        #shutil.copy(src_file, dst_dir)
    else:
        src_files = glob.glob(fileout)
        for src_file in src_files:
          print('Rank = {}, Putting file={} iter={} to SharedFS'.format(mpi_rank, src_file, state.iter_num))
          shutil.copy(src_file, dst_dir)

def get_files_exchange(state, mpi_rank):
    ##-->Train gets selfplay
    ##-->Self-play gets eval-model
    if mpi_rank == FLAGS.train_rank:
        selfplay_files = glob.glob(os.path.join(FLAGS.shared_dir_exchange, state.output_model_name + '-mpirank-*.zz*'))
        for filename in selfplay_files:
            print('Rank = {}, Getting file={} iter={} from SharedFS'.format(mpi_rank, filename, state.iter_num))
            shutil.copy(filename, fsdb.golden_chunk_dir())
    else:
        ##self-play needs to get training eval model
        dst_dir  = os.path.join(fsdb.models_dir())

        src_file = os.path.join(FLAGS.shared_dir_exchange, state.train_model_name + '.pb')
        print('Rank = {}, Getting file={} iter={} from SharedFS'.format(mpi_rank, src_file, state.iter_num))
        shutil.copy(src_file, dst_dir)

        src_file = os.path.join(FLAGS.shared_dir_exchange, state.train_model_name + '.pb' + '.og')
        print('Rank = {}, Getting file={} iter={} from SharedFS'.format(mpi_rank, src_file, state.iter_num))
        shutil.copy(src_file, dst_dir)

        ##train times as well
        #src_file = os.path.join(FLAGS.shared_dir_exchange, 'train_times.txt')
        #print('Rank = {}, Getting file={} iter={} from SharedFS'.format(mpi_rank, src_file, state.iter_num))
        #shutil.copy(src_file, dst_dir)


def mpi_all_ranks_sync(state, mpi_comm, mpi_rank, mpi_size, model_win_result):
    ##Ideally, a barrier and a broadcast should have worked but it doesn't
    ##mpi_comm.Barrier()
    #model_win_result[state.iter_num] = mpi_comm.bcast(model_win_result[state.iter_num], root=FLAGS.train_rank)
    print('Wait at barrier/broadcast iter = {} mpi_rank = {} took {} seconds from start'.format(state.iter_num, mpi_rank, (time.time()-state.start_time)))
    if mpi_rank == FLAGS.train_rank:
        for i in range(mpi_size-1):
            mpi_comm.send(model_win_result[state.iter_num], dest=(i+1), tag=state.iter_num)
            tmp =  mpi_comm.recv(source=(i+1), tag=state.iter_num)
    else:
      model_win_result[state.iter_num] = mpi_comm.recv(source=0, tag=state.iter_num)
      mpi_comm.send(int(-1), dest=0, tag=state.iter_num)

    print('COMPLETED WAIT at broadcast iter = {} mpi_rank = {} took {} seconds from start'.format(state.iter_num, mpi_rank, (time.time()-state.start_time)))


def rl_loop(mpi_comm, mpi_rank, mpi_size):
  """The main reinforcement learning (RL) loop."""

  # init_start is once per worker
  minigo_print(key=constants.INIT_START)
  state = State()

  ##-set environment variables
  set_env_variables()

  if FLAGS.use_multinode:
    model_win_result = [-1] * (FLAGS.iterations+2)
    print('At rank = {} start_time was {} seconds, before reset after init_stop'.format(mpi_rank, state.start_time))

  if FLAGS.checkpoint_dir:
    # Start from a partially trained model.
    initialize_from_checkpoint(state)
  else:
    raise RuntimeError('Expected to train from checkpoint, train from random init disabled.')

  selfplay_flags = parse_flags_file(os.path.join(FLAGS.flags_dir, 'selfplay.flags'))
  train_flags = parse_flags_file(os.path.join(FLAGS.flags_dir, 'train.flags'))

  if mpi_rank == 0:
     mlperf_log.MINIGO_TAG_SET.update(['save_model'])

     minigo_print(key=constants.OPT_BASE_LR, value=train_flags['lr_rates'])
     minigo_print(key=constants.OPT_LR_DECAY_BOUNDARY_STEPS, value=train_flags['lr_boundaries'])
     minigo_print(key=constants.GLOBAL_BATCH_SIZE, value=train_flags['train_batch_size'])

     mlperf_log.MINIGO_TAG_SET.update(['virtual_losses'])
     minigo_print(key='virtual_losses', value=selfplay_flags['virtual_losses'])

  # spawn workers if multinode
  if FLAGS.use_multinode:
    if mpi_rank == FLAGS.train_rank:
      i_comm = spawn_train_workers(state)

  while state.iter_num < FLAGS.iterations:
    state.iter_num += 1
    with logged_timer('Iteration time'):
      if FLAGS.use_multinode:
        if mpi_rank == FLAGS.train_rank:
          # data ready, do the barrier to start training
          # assume train rank find the correct data to use without passing in
          i_comm.barrier()
          #print tags after first init
          if state.iter_num == 1:
            minigo_print(key=constants.INIT_STOP)
            # Barrier After INIT_STOP
            mpi_comm.barrier()
            minigo_print(key=constants.RUN_START)
            # restart timer since we are now training from here
            state.start_time = time.time()
            # Add Barrier after RUN_START to be rule compliant
            mpi_comm.barrier()
            minigo_print(key=mlperf_log.TRAIN_LOOP)

          minigo_print(key=constants.EPOCH_START, metadata={'epoch_num': state.iter_num})

          # wait on workers to finish
          i_comm.barrier()
          print('Train rank = {} done in iter = {} and {} seconds from start'.format(mpi_rank, state.iter_num, (time.time()-state.start_time)))

          # save model log
          minigo_print(key='save_model', value={'iteration': state.iter_num})

          elapsed = time.time() - state.start_time
          timestamps_path = os.path.join(fsdb.models_dir(), 'train_times.txt')
          with gfile.Open(timestamps_path, 'a') as f:
            print('{:.3f} {}'.format(elapsed, state.train_model_name), file=f)

          # add copy here to copy trained model from tmp
          shutil.move(os.path.join(fsdb.models_dir(), 'new_model.pb.og'), os.path.join(fsdb.models_dir(), state.train_model_name+'.pb.og'))
          shutil.move(os.path.join(fsdb.models_dir(), 'new_model.index'), os.path.join(fsdb.models_dir(), state.train_model_name+'.index'))
          shutil.move(os.path.join(fsdb.models_dir(), 'new_model.meta'), os.path.join(fsdb.models_dir(), state.train_model_name+'.meta'))
          shutil.move(os.path.join(fsdb.models_dir(), 'new_model.data-00000-of-00001'), os.path.join(fsdb.models_dir(), state.train_model_name+'.data-00000-of-00001'))

          convert_icomm = convert_noasync(state)
          model_win_rate = eval_noasync(state)
          # sync on convert
          convert_icomm.Barrier()
          convert_icomm.Disconnect()

          put_files_exchange(state, mpi_rank) ##put it in freeze
          model_win_result[state.iter_num] = 1 if (model_win_rate >= FLAGS.gating_win_rate) else 0
          print('Freeze and Model-eval rank = {} done in iter = {} and {} seconds from start'.format(mpi_rank, state.iter_num, time.time()-state.start_time))
        else:
          #print tags after first init
          if state.iter_num == 1:
            # Barrier After INIT_STOP
            mpi_comm.barrier()
            # Add Barrier after RUN_START to be rule compliant
            mpi_comm.barrier()
          selfplay_noasync(state)
          print('Selfplay rank = {} done in iter = {} and {} seconds from start'.format(mpi_rank, state.iter_num, (time.time()-state.start_time)))

        ##sync first
        mpi_all_ranks_sync(state, mpi_comm, mpi_rank, mpi_size, model_win_result)

        ##exchange_files; no failsafe if the files don't appear in time; sync first
        get_files_exchange(state, mpi_rank)

        ##print some model statistics for model-win-result
        print('Model-win-result[{}] is {}, from mpi_rank={} best-model={}, gen_num={}'.format(state.iter_num, model_win_result[state.iter_num],
                                                                                              state.best_model_name, state.gen_num, mpi_rank))
        if model_win_result[state.iter_num] == 1:
          # Promote the trained model to the best model and increment the generation number.
          state.best_model_name = state.train_model_name
          state.gen_num += 1
      else:
        #print tags after first init
        if state.iter_num == 1:
          minigo_print(key=constants.INIT_STOP)
          minigo_print(key=constants.RUN_START)
          # restart timer since we are now training from here
          state.start_time = time.time()
          minigo_print(key=mlperf_log.TRAIN_LOOP)
        minigo_print(key=constants.EPOCH_START, metadata={'epoch_num': state.iter_num})
        ##-->single-node
        # Train on shuffled game data from recent selfplay rounds.
        tf_records = get_golden_chunk_records(state)

        # Run train -> (eval, validation) & selfplay in parallel.
        model_win_rate, _ = wait([
          train_eval_convert(state, tf_records),
          selfplay(state)])

        if model_win_rate >= FLAGS.gating_win_rate:
          # Promote the trained model to the best model and increment the generation
          # number.
          state.best_model_name = state.train_model_name
          state.gen_num += 1

      if mpi_rank == 0:
        minigo_print(key=constants.EPOCH_STOP, metadata={'epoch_num': state.iter_num})
  if FLAGS.use_multinode:
    if mpi_rank == FLAGS.train_rank:
      i_comm.barrier()
      i_comm.Disconnect()

def main(unused_argv):
  """Run the reinforcement learning loop."""
  logging.getLogger('mlperf_compliance').propagate = False

  ##-->multi-node setup
  if FLAGS.use_multinode:
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    print('[MPI Init] MPI rank {}, mpi size is {} host is {}'.format(mpi_rank, mpi_size, socket.gethostname()))
  else:
    mpi_comm = None
    mpi_rank = 0
    mpi_size = 1

  print('Wiping dir %s' % FLAGS.base_dir, flush=True)
  shutil.rmtree(FLAGS.base_dir, ignore_errors=True)
  dirs = [fsdb.models_dir(), fsdb.selfplay_dir(), fsdb.holdout_dir(),
          fsdb.eval_dir(), fsdb.golden_chunk_dir(), fsdb.working_dir()]

  ##-->sharedFS for dataExchange. tmp solution 5/6/2019
  if FLAGS.use_multinode:
      ensure_dir_exists(FLAGS.shared_dir_exchange)
  for d in dirs:
    ensure_dir_exists(d);

  # Copy the flag files so there's no chance of them getting accidentally
  # overwritten while the RL loop is running.
  flags_dir = os.path.join(FLAGS.base_dir, 'flags')
  shutil.copytree(FLAGS.flags_dir, flags_dir)
  FLAGS.flags_dir = flags_dir

  # Copy the target model to the models directory so we can find it easily.
  shutil.copy(FLAGS.target_path, os.path.join(fsdb.models_dir(), 'target.pb'))
  shutil.copy(FLAGS.target_path+'.og', os.path.join(fsdb.models_dir(), 'target.pb.og'))

  with logged_timer('Total time from mpi_rank={}'.format(mpi_rank)):
    try:
      rl_loop(mpi_comm, mpi_rank, mpi_size)
    finally:
      asyncio.get_event_loop().close()


if __name__ == '__main__':
  app.run(main)
