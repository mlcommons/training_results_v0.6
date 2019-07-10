# Here need some words

import os
import random
import functools
import shutil

import numpy as np
import tensorflow as tf
import threading

from mpi4py import MPI
from absl import app, flags
from rl_loop import example_buffer

flags.DEFINE_string('read_path', '/tmp/minigo',
                    'Path to the read origin data.')

flags.DEFINE_string('write_path', '/tmp/minigo/output',
                    'Path to the read origin data.')

flags.DEFINE_integer('out_files_number', 2,
                     'Num of files to produce.')

flags.DEFINE_integer('physical_cores', 56,
                     'Num of cores.')

flags.DEFINE_integer('seed', 0,
                     'Random seed.')

FLAGS = flags.FLAGS


def main(unused_argv):
  mpi_comm = MPI.COMM_WORLD
  mpi_rank = mpi_comm.Get_rank()
  mpi_size = mpi_comm.Get_size()
  # avoid seed out of range
  random.seed(FLAGS.seed % 1048576)
  tf.set_random_seed(FLAGS.seed % 1048576)
  np.random.seed(FLAGS.seed % 1048576)

  pattern = os.path.join(FLAGS.read_path, '*.zz')
  files = tf.gfile.Glob(pattern)

  buffer = example_buffer.ExampleBuffer(sampling_frac=1.0)
  example_num = buffer.parallel_fill(files, threads=FLAGS.physical_cores)
  # make sure all nodes generate same number of examples
  example_num = int(mpi_comm.allreduce(example_num, op=MPI.MIN))
  buffer.flush_new(FLAGS.write_path+'_{}'.format(mpi_rank), example_num, FLAGS.out_files_number, threads=1)

  shutil.rmtree('/tmp/minigo/home', ignore_errors=True)

if __name__ == '__main__':
  app.run(main)






