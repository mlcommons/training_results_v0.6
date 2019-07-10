# Copyright 2018 Google LLC
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

"""Train a network.

Usage:
  BOARD_SIZE=19 python train.py tfrecord1 tfrecord2 tfrecord3
"""

import logging

from absl import app, flags
import numpy as np
import tensorflow as tf

import bigtable_input
import dual_net
import preprocessing
import utils
import os

##import horovod separately?
import horovod.tensorflow as hvd
from mpi4py import MPI
import multiprocessing as mp

# See www.moderndescartes.com/essays/shuffle_viz for discussion on sizing
flags.DEFINE_integer('shuffle_buffer_size', 2000,
                     'Size of buffer used to shuffle train examples.')

flags.DEFINE_integer('steps_to_train', None,
                     'Number of training steps to take. If not set, iterates '
                     'once over training data.')

flags.DEFINE_integer('window_size', 500000,
                     'Number of games to include in the window')

flags.DEFINE_float('filter_amount', 1.0,
                   'Fraction of positions to filter from golden chunks,'
                   'default, 1.0 (no filter)')

flags.DEFINE_string('export_path', None,
                    'Where to export the model after training.')

flags.DEFINE_bool('use_bt', False,
                  'Whether to use Bigtable as input.  '
                  '(Only supported with --use_tpu, currently.)')

flags.DEFINE_bool('freeze', False,
                  'Whether to freeze the graph at the end of training.')

flags.DEFINE_integer('trt_batch', 0,
                     'Batch size to create TRT graph. 0 means disable')

flags.DEFINE_bool('use_multinode', False,
                  'Whether using multinode configuration or not.')

flags.DEFINE_string('golden_chunk_pattern', None,
                    'pattern to find training data.')

flags.DEFINE_integer('window_iters', 10,
                     'Number of iterations to look back for data')

flags.DEFINE_integer('num_selfplays', 1,
                     'Number of nodes to do selfplay')

flags.DEFINE_integer('total_iters', 50,
                     'Total iteration to train in persist case')

flags.register_multi_flags_validator(
    ['use_bt', 'use_tpu'],
    lambda flags: flags['use_tpu'] if flags['use_bt'] else True,
    '`use_bt` flag only valid with `use_tpu` as well')


@flags.multi_flags_validator(
    ['use_bt', 'cbt_project', 'cbt_instance', 'cbt_table'],
    message='Cloud Bigtable configuration flags not correct')
def _bt_checker(flags_dict):
    if not flags_dict['use_bt']:
        return True
    return (flags_dict['cbt_project']
            and flags_dict['cbt_instance']
            and flags_dict['cbt_table'])


# From dual_net.py
flags.declare_key_flag('work_dir')
flags.declare_key_flag('train_batch_size')
flags.declare_key_flag('num_tpu_cores')
flags.declare_key_flag('use_tpu')
flags.declare_key_flag('use_mgpu_horovod')

FLAGS = flags.FLAGS

class _PrefillStagingAreasHook(tf.train.SessionRunHook):
    def after_create_session(self, session, coord):
        # TODO: This assumes TF collections are ordered; is this safe?
        enqueue_ops = tf.get_collection('STAGING_AREA_PUTS')
        for i in range(len(enqueue_ops)):
            session.run(enqueue_ops[:i+1])

class EchoStepCounterHook(tf.train.StepCounterHook):
    """A hook that logs steps per second."""

    def _log_and_record(self, elapsed_steps, elapsed_time, global_step):
        s_per_sec = elapsed_steps / elapsed_time
        logging.info("{}: {:.3f} steps per second".format(global_step, s_per_sec))
        super()._log_and_record(elapsed_steps, elapsed_time, global_step)


def compute_update_ratio(weight_tensors, before_weights, after_weights):
    """Compute the ratio of gradient norm to weight norm."""
    deltas = [after - before for after,
              before in zip(after_weights, before_weights)]
    delta_norms = [np.linalg.norm(d.ravel()) for d in deltas]
    weight_norms = [np.linalg.norm(w.ravel()) for w in before_weights]
    ratios = [d / w for d, w in zip(delta_norms, weight_norms)]
    all_summaries = [
        tf.Summary.Value(tag='update_ratios/' +
                         tensor.name, simple_value=ratio)
        for tensor, ratio in zip(weight_tensors, ratios)]
    return tf.Summary(value=all_summaries)

class UpdateRatioSessionHook(tf.train.SessionRunHook):
    """A hook that computes ||grad|| / ||weights|| (using frobenius norm)."""

    def __init__(self, output_dir, every_n_steps=1000):
        self.output_dir = output_dir
        self.every_n_steps = every_n_steps
        self.before_weights = None
        self.file_writer = None
        self.weight_tensors = None
        self.global_step = None

    def begin(self):
        # These calls only works because the SessionRunHook api guarantees this
        # will get called within a graph context containing our model graph.

        self.file_writer = tf.summary.FileWriterCache.get(self.output_dir)
        self.weight_tensors = tf.trainable_variables()
        self.global_step = tf.train.get_or_create_global_step()

    def before_run(self, run_context):
        global_step = run_context.session.run(self.global_step)
        if global_step % self.every_n_steps == 0:
            self.before_weights = run_context.session.run(self.weight_tensors)

    def after_run(self, run_context, run_values):
        global_step = run_context.session.run(self.global_step)
        if self.before_weights is not None:
            after_weights = run_context.session.run(self.weight_tensors)
            weight_update_summaries = compute_update_ratio(
                self.weight_tensors, self.before_weights, after_weights)
            self.file_writer.add_summary(
                weight_update_summaries, global_step)
            self.before_weights = None


def train(*tf_records: "Records to train on"):
    """Train on examples."""
    tf.logging.set_verbosity(tf.logging.INFO)
    estimator = dual_net.get_estimator()

    ##batch-size arithmetic
    effective_batch_size = FLAGS.train_batch_size

    if FLAGS.use_mgpu_horovod:
        batch_size_per_gpu = effective_batch_size // hvd.size()
    else:
        batch_size_per_gpu = effective_batch_size

    if FLAGS.use_tpu:
        effective_batch_size *= FLAGS.num_tpu_cores

    if FLAGS.use_tpu:
        if FLAGS.use_bt:
            def _input_fn(params):
                games = bigtable_input.GameQueue(
                    FLAGS.cbt_project, FLAGS.cbt_instance, FLAGS.cbt_table)
                games_nr = bigtable_input.GameQueue(
                    FLAGS.cbt_project, FLAGS.cbt_instance, FLAGS.cbt_table + '-nr')
                return preprocessing.get_tpu_bt_input_tensors(
                    games,
                    games_nr,
                    params['batch_size'],
                    number_of_games=FLAGS.window_size,
                    random_rotation=True)
        else:
            def _input_fn(params):
                return preprocessing.get_tpu_input_tensors(
                    params['batch_size'],
                    tf_records,
                    random_rotation=True)
        # Hooks are broken with TPUestimator at the moment.
        hooks = []
    else:
        def _input_fn():
            return preprocessing.get_input_tensors(
                batch_size_per_gpu, #/FLAGS.train_batch_size,
                tf_records,
                filter_amount=FLAGS.filter_amount,
                shuffle_buffer_size=FLAGS.shuffle_buffer_size,
                random_rotation=True)

        if FLAGS.use_mgpu_horovod:
            ##add profiler hook
            ##export LD_LIBRARY_PATH=/usr/local/cuda-10.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH
            if False:
                outdir_hvd=os.path.join("./profile/rank_%d" % hvd.rank())
                profiler_hook = tf.train.ProfilerHook(output_dir=outdir_hvd, save_steps=100, show_dataflow=True, show_memory=False)
                hooks = [hvd.BroadcastGlobalVariablesHook(0),
                        profiler_hook,
                         _PrefillStagingAreasHook()]
            else:
                hooks = [hvd.BroadcastGlobalVariablesHook(0),
                         _PrefillStagingAreasHook()]
        else:
            hooks = []#UpdateRatioSessionHook(FLAGS.work_dir),
            #EchoStepCounterHook(output_dir=FLAGS.work_dir)]

    steps = FLAGS.steps_to_train
    logging.info("Training, steps = %s, batch = %s -> %s examples",
                 steps or '?', effective_batch_size,
                 (steps * effective_batch_size) if steps else '?')

    if FLAGS.use_bt:
        games = bigtable_input.GameQueue(
            FLAGS.cbt_project, FLAGS.cbt_instance, FLAGS.cbt_table)
        if not games.read_wait_cell():
            games.require_fresh_games(20000)
        latest_game = games.latest_game_number
        index_from = max(latest_game, games.read_wait_cell())
        print("== Last game before training:", latest_game, flush=True)
        print("== Wait cell:", games.read_wait_cell(), flush=True)

    try:
        estimator.train(_input_fn, steps=steps, hooks=hooks)
        if FLAGS.use_bt:
            bigtable_input.set_fresh_watermark(games, index_from,
                                               FLAGS.window_size)
    except:
        if FLAGS.use_bt:
            games.require_fresh_games(0)
        raise

def init_train_loop():
    """Init training loop"""
    tf.logging.set_verbosity(tf.logging.INFO)
    return dual_net.get_estimator()

def train_loop(estimator, *tf_records):
    """Train on examples."""
    ##batch-size arithmetic
    effective_batch_size = FLAGS.train_batch_size
    batch_size_per_gpu = effective_batch_size // hvd.size()
    def _input_fn():
        return preprocessing.get_input_tensors(
            batch_size_per_gpu,
            tf_records,
            filter_amount=FLAGS.filter_amount,
            shuffle_buffer_size=FLAGS.shuffle_buffer_size,
            random_rotation=True)

    hooks = [hvd.BroadcastGlobalVariablesHook(0),
             _PrefillStagingAreasHook()]

    steps = FLAGS.steps_to_train
    logging.info("Training, steps = %s, batch = %s -> %s examples",
                 steps or '?', effective_batch_size,
                 (steps * effective_batch_size) if steps else '?')
    try:
        estimator.train(_input_fn, steps=steps, hooks=hooks)
    except:
        raise


def export_and_freeze_model():
    ##random stuff
    if FLAGS.export_path:
        dual_net.export_model(FLAGS.export_path)
    if FLAGS.freeze:
        if FLAGS.use_tpu:
            dual_net.freeze_graph_tpu(FLAGS.export_path)
        elif FLAGS.trt_batch > 0:
            dual_net.freeze_graph(FLAGS.export_path, True, FLAGS.trt_batch)
        else:
            dual_net.freeze_graph(FLAGS.export_path)


def get_golden_chunk_records(pattern, num_selfplays, iter_num, window_size, num_shard):
    """Return up to num_records of golden chunks to train on.

    Returns:
    A list of golden chunks up to num_records in length, sorted by path.
    """
    if iter_num <= window_size:
        win_size=(iter_num)*num_selfplays + (window_size-iter_num)
    else:
        win_size=(window_size)*num_selfplays
    print('Train get_golden_chunks at iter = {} has win_size = {}'.format(iter_num, win_size))

    return sorted(tf.gfile.Glob(pattern), reverse=True)[:win_size*num_shard]


def main(argv):
    """Train on examples and export the updated model weights."""
    tf_records = argv[1:]
    logging.info("Training on %s records: %s to %s",
                 len(tf_records), tf_records[0], tf_records[-1])

    if FLAGS.use_mgpu_horovod:
        ##-->multi-node setup
        if FLAGS.use_multinode:
            icomm = MPI.Comm.Get_parent()
            comm_world = MPI.COMM_WORLD
            hvd.init(comm_world)
            print('hvd init from rank = {} of {}'.format(hvd.rank(), hvd.size()))
        else:
            hvd.init()

    if FLAGS.use_mgpu_horovod:
        if FLAGS.use_multinode:
            tf.logging.set_verbosity(tf.logging.INFO)

            tf_records_ph = tf.placeholder(tf.string)
            data_iter = preprocessing.get_input_tensors_new(FLAGS.train_batch_size // hvd.size(),
                                                            tf_records_ph,
                                                            filter_amount=FLAGS.filter_amount,
                                                            shuffle_buffer_size=FLAGS.shuffle_buffer_size,
                                                            random_rotation=True)

            features, labels = data_iter.get_next()
            train_op = dual_net.model_fn_sess(features, labels, tf.estimator.ModeKeys.TRAIN, FLAGS.flag_values_dict())

            config = tf.ConfigProto()
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
            config.gpu_options.per_process_gpu_memory_fraction = 0.6
            config.gpu_options.visible_device_list = str(hvd.local_rank())
            config.gpu_options.force_gpu_compatible = True
            config.intra_op_parallelism_threads = 1
            config.inter_op_parallelism_threads = 4
            sess = tf.Session(config=config)

            tf.train.Saver().restore(sess, '/opt/reinforcement/minigo/ml_perf/checkpoint/9/work_dir/model.ckpt-9383')
            for i in range(FLAGS.total_iters):
                # sync all before start iteration
                icomm.barrier()
                # get newest tf record use same function here
                tf_records = get_golden_chunk_records(FLAGS.golden_chunk_pattern, FLAGS.num_selfplays, i+1, FLAGS.window_iters, hvd.size())
                sess.run(data_iter.initializer, {tf_records_ph: tf_records})
                while True:
                    try:
                        sess.run(train_op)
                    except tf.errors.OutOfRangeError:
                        break
                #print info
                if hvd.rank() == 0:
                    for rrr in tf_records:
                        print(rrr)
                if hvd.rank() == 0:
                    tf.train.Saver().save(sess, FLAGS.export_path)
                    tmp_n = dual_net.DualNetwork(FLAGS.export_path)
                    out_graph = tf.graph_util.convert_variables_to_constants(
                        tmp_n.sess, tmp_n.sess.graph.as_graph_def(), ["policy_output", "value_output"])
                    with tf.gfile.GFile(FLAGS.export_path + '.pb.og', 'wb') as f:
                        f.write(out_graph.SerializeToString())
                print('rank={} icomm.Barrier()'.format(comm_world.Get_rank()))
                # sync all after finish
                icomm.barrier()

            sess.close()
            print('rank={} icomm.Barrier()'.format(comm_world.Get_rank()))
            icomm.barrier()
            icomm.Disconnect()
        else:
            with utils.logged_timer("Training"):
                train(*tf_records)
            if hvd.rank() == 0: export_and_freeze_model()
        ##--horovod shuts down
        print('rank={} hvd.shutdown'.format(hvd.rank()))
        hvd.shutdown()
    else:
        with utils.logged_timer("Training"):
            train(*tf_records)
            export_and_freeze_model()


if __name__ == "__main__":
    app.run(main)
