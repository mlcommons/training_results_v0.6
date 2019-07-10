# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

""" example train fit utility """
import logging
import os
import time
import re
import math
import mxnet as mx
import numpy as np

#### imports needed for fit monkeypatch
from mxnet.initializer import Uniform
from mxnet.context import cpu
from mxnet.model import BatchEndParam
from mxnet.initializer import Uniform
from mxnet.io import DataDesc, DataIter, DataBatch
from mxnet.base import _as_list
from mxnet.explorer import Explorer
import copy
from distutils.util import strtobool
#####

from mlperf_compliance import constants as mlperf_constants
from mlperf_compliance import tags as mlperf_log
from mlperf_log_utils import mx_resnet_print, all_reduce, mpiwrapper

def get_epoch_size(args, kv):
    return math.ceil(int(args.num_examples / kv.num_workers) / args.batch_size)

def _get_gpu(gpus):
    idx = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    gpu = gpus.split(",")[idx]
    return gpu

def _get_lr_scheduler(args, kv):
    if 'lr_factor' not in args or args.lr_factor >= 1:
        return (args.lr, None)
    epoch_size = get_epoch_size(args, kv)
    begin_epoch = 0
    mx_resnet_print(key=mlperf_constants.OPT_BASE_LR, val=args.lr)
    mx_resnet_print(key=mlperf_constants.OPT_LR_WARMUP_EPOCHS, val=args.warmup_epochs)

    if 'pow' in args.lr_step_epochs:
        lr = args.lr
        max_up = args.num_epochs * epoch_size
        pwr = float(re.sub('pow[- ]*', '', args.lr_step_epochs))
        poly_sched = mx.lr_scheduler.PolyScheduler(max_up, lr, pwr)
        return (lr, poly_sched)
    step_epochs = [int(l) for l in args.lr_step_epochs.split(',')] if len(args.lr_step_epochs.strip()) else []
    lr = args.lr
    for s in step_epochs:
        if begin_epoch >= s:
            lr *= args.lr_factor
    if lr != args.lr:
        logging.info('Adjust learning rate to %e for epoch %d',
                     lr, begin_epoch)

    steps = [epoch_size * (x - begin_epoch)
             for x in step_epochs if x - begin_epoch > 0]
    if steps:
        if kv:
            num_workers = kv.num_workers
        else:
            num_workers = 1
        epoch_size = math.ceil(int(args.num_examples/num_workers)/args.batch_size)
        return (lr, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=args.lr_factor,
                                                         base_lr=args.lr, warmup_steps=epoch_size * args.warmup_epochs,
                                                         warmup_mode=args.warmup_strategy))
    else:
        return (lr, None)


def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    train = parser.add_argument_group('Training', 'model training')
    train.add_argument('--network', type=str,
                       help='the neural network to use')
    train.add_argument('--num-layers', type=int,
                       help='number of layers in the neural network, \
                             required by some networks such as resnet')
    train.add_argument('--gpus', type=str,
                       help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu')
    train.add_argument('--num-epochs', type=int, default=100,
                       help='max num of epochs')
    train.add_argument('--lr', type=float, default=0.1,
                       help='initial learning rate')
    train.add_argument('--lr-factor', type=float, default=0.1,
                       help='the ratio to reduce lr on each step')
    train.add_argument('--lr-step-epochs', type=str,
                       help='the epochs to reduce the lr, e.g. 30,60')
    train.add_argument('--wd_step_epochs', type=str,
                       help='the epochs to reduce the wd, e.g. 30,60')
    train.add_argument('--wd_factor', type=float, default=0.1,
                       help='the ratio to reduce wd on each step')
    train.add_argument('--initializer', type=str, default='default',
                       help='the initializer type')
    train.add_argument('--optimizer', type=str, default='sgd',
                       help='the optimizer type')
    train.add_argument('--explorer', type=str, default='linear',
                       help='the explorer type')
    train.add_argument('--burn_in', type=int, default=0,
                       help='iteration number of burn_in behavior')
    train.add_argument('--lr_range_max', type=float, default=1.0,
                       help='learning rate range_max for explorer')
    train.add_argument('--lr_range_min', type=float, default=0.0,
                       help='learning rate range_min for explorer')
    train.add_argument('--wd_range_max', type=float, default=0.001,
                       help='weight decay range_max for explorer')
    train.add_argument('--wd_range_min', type=float, default=0.00001,
                       help='weight decay range_min for explorer')
    train.add_argument('--cg_range_max', type=float, default=-1,
                       help='clip gradient range_max for explorer')
    train.add_argument('--cg_range_min', type=float, default=-1,
                       help='clip gradient range_min for explorer')
    train.add_argument('--num_grps', type=float, default=1,
                       help='the number of MPI process groups, which uses same momentum coefficient, for explorer. 0 means that momentum coefficient is equal to LR coefficient.')
    train.add_argument('--num_cg_grps', type=float, default=1,
                       help='the number of MPI process groups, which uses same clip gradient, for explorer.')
    train.add_argument('--lr_decay', type=float, default=2.0,
                       help='lr_decay is decay speed of learning rate')
    train.add_argument('--wd_decay', type=float, default=2.0,
                       help='wd_decay is weight decay speed of learning rate')
    train.add_argument('--lr_decay_mode', type=str, default=None,
                       help='learning decay mode')
    train.add_argument('--wd_decay_mode', type=str, default='step',
                       help='learning decay mode')
    train.add_argument('--natan_turn_epoch', type=int, default=50,
                       help='change lr according to this epoch. only use natan lr_decay_mode')
    train.add_argument('--natan_final_ratio', type=float, default=-1,
                       help='target final ratio of natan decay. No effect if value < 0')
    train.add_argument('--ds_fix_min', type=int, default=1,
                       help='min range value will not move')
    train.add_argument('--ds_fix_max', type=int, default=1,
                       help='max range value will not move')
    train.add_argument('--ds_upper_factor', type=float, default=1.0,
                       help='This is used by dynamic search. xx_range_max can move to xx_range_max * ds_upper_factor')
    train.add_argument('--ds_lower_factor', type=float, default=1.0,
                       help='This is used by dynamic search. xx_range_min can move to xx_range_min * ds_lower_factor')
    train.add_argument('--explore_freq', type=int, default=1,
                       help='Frequency of explorer which include try/revert update and allgather')
    train.add_argument('--explore_start_epoch', type=int, default=0,
                       help='starting epoch of explorer')
    train.add_argument('--smooth_decay', type=strtobool, default=False,
                       help='decay every iteration. False means decay every epoch')
    train.add_argument('--decay_after_warmup', type=strtobool, default=False,
                       help='start to decay after warmup.')
    train.add_argument('--end_lr_ratio', type=float, default=0.0,
                       help='learning rate will be large than lr * end_lr_ratio')
    train.add_argument('--mom', type=float, default=0.9,
                       help='momentum for sgd')
    train.add_argument('--mom_end', type=float, default=None,
                       help='final momentum for sgd')
    train.add_argument('--wd', type=float, default=0.0001,
                       help='weight decay for sgd')
    train.add_argument('--batch-size', type=int, default=128,
                       help='the batch size')
    train.add_argument('--disp-batches', type=int, default=20,
                       help='show progress for every n batches')
    train.add_argument('--model-prefix', type=str,
                       help='model prefix')
    train.add_argument('--save-period', type=int, default=1, help='params saving period')
    train.add_argument('--eval-period', type=int, default=1, help='evaluation every N epochs')
    train.add_argument('--eval-offset', type=int, default=0, help='first evaluation on epoch N')

    train.add_argument('--top-k', type=int, default=0,
                       help='report the top-k accuracy. 0 means no report.')
    train.add_argument('--dtype', type=str, default='float32',
                       help='precision: float32 or float16')
    # additional parameters for large batch sgd
    train.add_argument('--macrobatch-size', type=int, default=0,
                       help='distributed effective batch size')
    train.add_argument('--warmup-epochs', type=int, default=5,
                       help='the epochs to ramp-up lr to scaled large-batch value')
    train.add_argument('--warmup-strategy', type=str, default='linear',
                       help='the ramping-up strategy for large batch sgd')
    train.add_argument('--wd_warmup', type=bool, default=True)
    train.add_argument('--logging-dir', type=str, default='logs')
    train.add_argument('--log', type=str, default='')
    train.add_argument('--bn-gamma-init0', action='store_true')
    train.add_argument('--epoch-size',type=int, default=0,
                       help='set number of batches in an epoch. useful for debugging')
    train.add_argument('--profile-worker-suffix', type=str, default='',
                       help='profile workers actions into this file. During distributed training\
                             filename saved will be rank1_ followed by this suffix')
    train.add_argument('--profile-server-suffix', type=str, default='',
                       help='profile server actions into a file with name like rank1_ followed by this suffix \
                             during distributed training')
    train.add_argument('--accuracy-threshold', default=1.0, type=float,
                       help='stop training after top1 reaches this value')
    train.add_argument('--smooth_alpha', type=float, default=0.0,
                        help='label smoothing constant')
    train.add_argument('--add_one_fwd_epoch', type=int, default=None,
                        help='start epoch for additional forward per iteration')
    train.add_argument('--no_augument_epoch', type=int, default=None,
                        help='start epoch for using non augumented batch')
    train.add_argument('--weight_seed', type=int, default=1234,
                        help='seed for weight initialization')
    train.add_argument('--bias_wd', type=strtobool, default=False,
                       help='use weight decay for fc bias layer')
    train.add_argument('--decay_steps', type=int, default=0,
                        help='steps for decay. decay_steps <= 0 means that decay_steps is calculated from num-epochs')
    train.add_argument('--bn_lr_decay', type=strtobool, default=True,
                       help='decay learning rate for BN layer')
    return train


class CorrectCount(mx.metric.Accuracy):
    def __init__(self, axis=1, name='correct-count',
                 output_names=None, label_names=None):
        super(CorrectCount, self).__init__(
                name=name, axis=axis,
                output_names=output_names, label_names=label_names)
        self.axis = axis

    def get(self):
        return (self.name, self.sum_metric)


class TotalCount(mx.metric.Accuracy):
    def __init__(self, axis=1, name='total-count',
                 output_names=None, label_names=None):
        super(TotalCount, self).__init__(
                name=name, axis=axis,
                output_names=output_names, label_names=label_names)
        self.axis = axis

    def get(self):
        return (self.name, self.num_inst)

def mlperf_fit(self, args, data_loader, epoch_size, eval_metric='acc',
               epoch_end_callback=None, batch_end_callback=None, kvstore='local',
               optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
               explorer='linear', explorer_params=None,
               eval_end_callback=None,
               eval_batch_end_callback=None, initializer=Uniform(0.01),
               arg_params=None, aux_params=None, allow_missing=False,
               force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
               validation_metric=None, monitor=None, sparse_row_id_fn=None,
               eval_offset=0, eval_period=1,
               accuracy_threshold=1.0):

    assert num_epoch is not None, 'please specify number of epochs'

    if monitor is not None:
        self.install_monitor(monitor)

    self.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                        optimizer_params=optimizer_params)

    explorer = Explorer.create_explorer(name=explorer, optimizer=self._optimizer,
                                        explorer_params=explorer_params)
    #This mxnet can not use several optimizers without sgd series
    explorer.set_best_coeff(0)
    explorer.set_best_wd_coeff(0)
    explorer.set_best_cg(0)
    exp_freq = explorer_params['explore_freq']
    exp_start_epoch = explorer_params['explore_start_epoch']

    if validation_metric is None:
        validation_metric = eval_metric
    ###########################################################################
    # Adding Correct and Total Count metrics
    ###########################################################################
    if not isinstance(validation_metric, list):
        validation_metric = [validation_metric]

    validation_metric = mx.metric.create(validation_metric)

    if not isinstance(validation_metric, mx.metric.CompositeEvalMetric):
        vm = mx.metric.CompositeEvalMetric()
        vm.append(validation_metric)
        validation_metric = vm

    for m in [CorrectCount(), TotalCount()]:
        validation_metric.metrics.append(m)
    ###########################################################################

    if not isinstance(eval_metric, mx.metric.EvalMetric):
        eval_metric = mx.metric.create(eval_metric)

    try:
        world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    except:
        world_rank = 0
        world_size = 1

    use_cval_data =    explorer_params['add_one_fwd_epoch'] < num_epoch \
                    or explorer_params['no_augument_epoch'] < num_epoch

    best_rank = 0
    self.prepare_states()

    mx_resnet_print(key=mlperf_constants.INIT_STOP, sync=True)
    mx_resnet_print(key=mlperf_constants.RUN_START, sync=True)

    # data iterators
    (train_data, eval_data, cval_data) = data_loader(args, kvstore)
    if 'dist' in args.kv_store and not 'async' in args.kv_store:
        logging.info('Resizing training data to %d batches per machine', epoch_size)
        # resize train iter to ensure each machine has same number of batches per epoch
        # if not, dist_sync can hang at the end with one machine waiting for other machines
        if not args.use_dali:
            train = mx.io.ResizeIter(train_data, epoch_size)

    block_epoch_start = begin_epoch
    block_epoch_count = eval_offset + 1 - (begin_epoch % eval_period)
    if block_epoch_count < 0:
        block_epoch_count += eval_period
    mx_resnet_print(key=mlperf_constants.BLOCK_START,
        metadata={'first_epoch_num': block_epoch_start + 1, 'epoch_count': block_epoch_count})
    ################################################################################
    # training loop
    ################################################################################

    for epoch in range(begin_epoch, num_epoch):
        mx_resnet_print(key=mlperf_constants.EPOCH_START, metadata={'epoch_num': epoch + 1})
        tic = time.time()
        eval_metric.reset()
        nbatch = 0

        use_normal_data_batch = epoch < explorer_params['no_augument_epoch']
        if not use_normal_data_batch:
            if world_rank == 0:
                self.logger.info('use non-augumented batch')

        end_of_batch = False

        if use_normal_data_batch:
            data_iter = iter(train_data)
            next_data_batch = next(data_iter)
        else:
            cval_iter = iter(cval_data)
            next_cval_batch = next(cval_iter)

        smooth_decay = explorer_params['smooth_decay']

        if not smooth_decay:
            explorer.apply_lr_decay_epoch(epoch)
            explorer.apply_wd_decay_epoch(epoch)
        explorer.set_mom(epoch)


        while not end_of_batch:
            if use_normal_data_batch:
                data_batch = next_data_batch
            else:
                cval_batch = next_cval_batch
            if monitor is not None:
                monitor.tic()

            if use_normal_data_batch:
                self.forward_backward(data_batch)
            else:
                self.forward_backward(cval_batch)

            if smooth_decay:
                explorer.apply_lr_decay_iter()
                explorer.apply_wd_decay_iter()
            explorer.apply_wd_warmup()
            explorer.apply_burn_in()

            use_explorer = (epoch == 0 and nbatch == 0) or (epoch >= exp_start_epoch and nbatch % exp_freq == 0)
            if use_explorer:
                explorer.set_tmp_coeff(world_rank)
                explorer.set_tmp_wd_coeff(world_rank)
                explorer.set_tmp_cg(world_rank)

            explorer.set_best_coeff(0)
            explorer.set_best_wd_coeff(0)
            explorer.set_best_cg(world_rank)
            self.update()

            if use_normal_data_batch:
                if isinstance(data_batch, list):
                    self.update_metric(eval_metric,
                                       [db.label for db in data_batch],
                                       pre_sliced=True)
                else:
                    self.update_metric(eval_metric, data_batch.label)
            else:
                if isinstance(cval_batch, list):
                    self.update_metric(eval_metric,
                                       [db.label for db in cval_batch],
                                       pre_sliced=True)
                else:
                    self.update_metric(eval_metric, cval_batch.label)

            if use_normal_data_batch:
                try:
                    # pre fetch next batch
                    next_data_batch = next(data_iter)
                except StopIteration:
                    end_of_batch = True
            else:
                try:
                    # pre fetch next cval batch
                    next_cval_batch = next(cval_iter)
                except StopIteration:
                    end_of_batch = True

            if use_normal_data_batch:
                if not end_of_batch:
                    self.prepare(next_data_batch, sparse_row_id_fn=sparse_row_id_fn)
            else:
                if not end_of_batch:
                    self.prepare(next_cval_batch, sparse_row_id_fn=sparse_row_id_fn)

            if monitor is not None:
                monitor.toc_print()

            if batch_end_callback is not None:
                batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                 eval_metric=eval_metric,
                                                 locals=locals())
                for callback in _as_list(batch_end_callback):
                    callback(batch_end_params)
            nbatch += 1

        mx_resnet_print(key=mlperf_constants.EPOCH_STOP, metadata={"epoch_num": epoch + 1})
        # one epoch of training is finished
        toc = time.time()
        if kvstore:
            if kvstore.rank == 0:
                self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))
        else:
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

        # sync aux params across devices
        #arg_params, aux_params = self.get_params()
        #self.set_params(arg_params, aux_params)

        if epoch_end_callback is not None:
            for callback in _as_list(epoch_end_callback):
                callback(epoch, self.symbol, arg_params, aux_params)

        #----------------------------------------
        # evaluation on validation set
        if eval_data and epoch >= eval_offset and (epoch - eval_offset) % eval_period == 0:
            mx_resnet_print(key=mlperf_constants.EVAL_START, metadata={'epoch_num': epoch + 1})
            res = self.score(eval_data, validation_metric,
                             score_end_callback=eval_end_callback,
                             batch_end_callback=eval_batch_end_callback, epoch=epoch)
            #TODO: pull this into default
            if kvstore:
                if kvstore.rank == 0:
                    for name, val in res:
                        self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)
            else:
                for name, val in res:
                    self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)
            res = dict(res)

            acc = [res['correct-count'], res['total-count']]
            acc = all_reduce(acc)
            acc = acc[0]/acc[1]
            mx_resnet_print(key=mlperf_constants.EVAL_STOP, metadata={'epoch_num': epoch + 1})

            mx_resnet_print(key=mlperf_constants.EVAL_ACCURACY, val=acc,
                            metadata={'epoch_num': epoch + 1})

            mx_resnet_print(key=mlperf_constants.BLOCK_STOP,
                            metadata={'first_epoch_num': block_epoch_start + 1})
            if acc > accuracy_threshold:
                mx_resnet_print(key=mlperf_constants.RUN_STOP,
                                metadata={'status': 'success'})

                return epoch

            if epoch < (num_epoch - 1):
                block_epoch_start = epoch + 1
                block_epoch_count = num_epoch - epoch - 1
                if block_epoch_count > eval_period:
                    block_epoch_count = eval_period
                mx_resnet_print(key=mlperf_constants.BLOCK_START,
                    metadata={'first_epoch_num': block_epoch_start + 1,
                              'epoch_count': block_epoch_count})

        # end of 1 epoch, reset the data-iter for another epoch
        if use_normal_data_batch:
            train_data.reset()
        else:
            cval_data.reset()

    mx_resnet_print(key=mlperf_constants.RUN_STOP,
                    metadata={'status': 'aborted'})
    return num_epoch

def fit(args, kv, model, initializer, data_loader, devs, arg_params, aux_params, **kwargs):
    """
    train a model
    args : argparse returns
    model : loaded model of the neural network
    initializer : weight initializer
    data_loader : function that returns the train and val data iterators
    devs : devices for training
    arg_params : model parameters
    aux_params : model parameters
    """
    if args.profile_server_suffix:
        mx.profiler.set_config(filename=args.profile_server_suffix, profile_all=True, profile_process='server')
        mx.profiler.set_state(state='run', profile_process='server')

    if args.profile_worker_suffix:
        if kv.num_workers > 1:
            filename = 'rank' + str(kv.rank) + '_' + args.profile_worker_suffix
        else:
            filename = args.profile_worker_suffix
        mx.profiler.set_config(filename=filename, profile_all=True, profile_process='worker')
        mx.profiler.set_state(state='run', profile_process='worker')

    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments %s', args)

    epoch_size = get_epoch_size(args, kv)

    # save model
    epoch_end_callbacks = []

    # learning rate
    lr, lr_scheduler = _get_lr_scheduler(args, kv)

    total_steps = math.ceil(args.num_examples * args.num_epochs / kv.num_workers / args.batch_size)
    warmup_steps = get_epoch_size(args,kv) * args.warmup_epochs

    if args.decay_steps > 0:
        decay_steps = args.decay_steps
        decay_epochs = math.ceil(args.decay_steps / get_epoch_size(args,kv))
    else:
        if args.decay_after_warmup:
            decay_steps = total_steps - warmup_steps
            decay_epochs = args.num_epochs - args.warmup_epochs
        else:
            decay_steps = total_steps
            decay_epochs = args.num_epochs

    explorer_params = {
        'burn_in_iter' : args.burn_in,
        'lr_range_max' : args.lr_range_max,
        'lr_range_min' : args.lr_range_min,
        'wd_range_max' : args.wd_range_max,
        'wd_range_min' : args.wd_range_min,
        'cg_range_max' : args.cg_range_max,
        'cg_range_min' : args.cg_range_min,
        'start_epoch' : 0,
        'end_epoch' : args.num_epochs,
        'lr_decay' : args.lr_decay,
        'wd_decay' : args.wd_decay if args.wd_decay != -1 else args.lr_decay,
        'num_grps' : args.num_grps,
        'num_cg_grps' : args.num_cg_grps,
        'lr_decay_mode' : args.lr_decay_mode,
        'wd_decay_mode' : args.wd_decay_mode,
        'wd_step_epochs' : [int(s.strip()) for s in args.wd_step_epochs.split(',')] if len(args.wd_step_epochs.strip()) else [],
        'wd_factor' : args.wd_factor,
        'lr_rate' : args.lr,
        'warmup_step' : warmup_steps,
        'warmup_epochs' : args.warmup_epochs,
        'wd_warmup' : args.wd_warmup,
        'ds_upper_factor' : args.ds_upper_factor,
        'ds_lower_factor' : args.ds_lower_factor,
        'ds_fix_min' : args.ds_fix_min,
        'ds_fix_max' : args.ds_fix_max,
        'explore_freq' : args.explore_freq,
        'natan_turn_epoch' : args.natan_turn_epoch,
        'natan_final_ratio' : args.natan_final_ratio,
        'explore_start_epoch' : args.explore_start_epoch,
        'momentum' : args.mom,
        'momentum_end' : args.mom_end if args.mom_end else args.mom,
        'epoch_size' : epoch_size,
        'smooth_decay' : args.smooth_decay,
        'add_one_fwd_epoch' : args.add_one_fwd_epoch if args.add_one_fwd_epoch is not None else args.num_epochs,
        'no_augument_epoch' : args.no_augument_epoch if args.no_augument_epoch is not None else args.num_epochs,
        'decay_after_warmup' : args.decay_after_warmup,
        'end_lr_ratio' :  args.end_lr_ratio,
        'total_steps' : total_steps,
        'decay_steps' : decay_steps,
        'decay_epochs' : decay_epochs,
    }

    optimizer_params = {
        'learning_rate': lr,
        'wd': args.wd * args.lr,
        'lr_scheduler': lr_scheduler,
        'multi_precision': True}

    mx_resnet_print(
            key=mlperf_constants.OPT_NAME,
            val='lars') #args.optimizer

    mx_resnet_print(
            key=mlperf_constants.LARS_EPSILON,
            val=1e-9)

    mx_resnet_print(
            key=mlperf_constants.LARS_OPT_WEIGHT_DECAY,
            val=args.wd)

    mx_resnet_print(
            key=mlperf_constants.LARS_OPT_LR_DECAY_POLY_POWER,
            val=args.lr_decay)

    mx_resnet_print(
            key=mlperf_constants.LARS_OPT_END_LR,
            val=args.lr*args.end_lr_ratio)

    mx_resnet_print(
            key=mlperf_constants.LARS_OPT_LR_DECAY_STEPS,
            val=decay_steps)


    ##########################################################################
    # MXNet excludes BN layers from L2 Penalty by default,
    # so this won't be explicitly stated anywhere in the code
    ##########################################################################
    mx_resnet_print(key=mlperf_log.MODEL_EXCLUDE_BN_FROM_L2, val=True)

    
    # Only a limited number of optimizers have 'momentum' property
    has_momentum = {'sgd', 'dcasgd', 'nag', 'signum', 'lbsgd'}
    if args.optimizer in has_momentum:
        optimizer_params['momentum'] = args.mom

    if args.optimizer == 'sgd':
        optimizer_params['bias_wd'] = args.bias_wd
        optimizer_params['bn_lr_decay'] = args.bn_lr_decay

    ### copy from nvidia-mxnet/3rdparty/horovod/example/mxnet/common/fit.py
    # A limited number of optimizers have a warmup period
    has_warmup = {'lbsgd', 'lbnag'}
    if args.optimizer in has_warmup:
        nworkers = kv.num_workers
        epoch_size = args.num_examples / args.batch_size / nworkers

        if epoch_size < 1:
            epoch_size = 1
        macrobatch_size = args.macrobatch_size
        if macrobatch_size < args.batch_size * nworkers:
            macrobatch_size = args.batch_size * nworkers
        #batch_scale = round(float(macrobatch_size) / args.batch_size / nworkers +0.4999)
        batch_scale = math.ceil(
            float(macrobatch_size) / args.batch_size / nworkers)
        optimizer_params['updates_per_epoch'] = epoch_size
        #optimizer_params['begin_epoch'] = args.load_epoch if args.load_epoch else 0
        optimizer_params['batch_scale'] = batch_scale
        optimizer_params['warmup_strategy'] = args.warmup_strategy
        optimizer_params['warmup_epochs'] = args.warmup_epochs
        optimizer_params['num_epochs'] = args.num_epochs
    ###

    # evaluation metrices
    eval_metrics = ['accuracy']
    if args.top_k > 0:
        eval_metrics.append(mx.metric.create(
            'top_k_accuracy', top_k=args.top_k))

    # callbacks that run after each batch
    batch_end_callbacks = []
    if 'horovod' in args.kv_store:
        # if using horovod, only report on rank 0 with global batch size
        if kv.rank == 0:
            batch_end_callbacks.append(mx.callback.Speedometer(
                kv.num_workers*args.batch_size, args.disp_batches))
        mx_resnet_print(key=mlperf_constants.GLOBAL_BATCH_SIZE,
                        val=kv.num_workers * args.batch_size)
    else:
        batch_end_callbacks.append(mx.callback.Speedometer(
            args.batch_size, args.disp_batches))
        mx_resnet_print(key=mlperf_constants.GLOBAL_BATCH_SIZE,
                        val=args.batch_size)


    mx_resnet_print(key=mlperf_log.EVAL_TARGET, val=args.accuracy_threshold)

    # run
    last_epoch = mlperf_fit(model,
                            args,
                            data_loader,
                            epoch_size,
                            begin_epoch=0,
                            num_epoch=args.num_epochs,
                            eval_metric=eval_metrics,
                            kvstore=kv,
                            optimizer=args.optimizer,
                            optimizer_params=optimizer_params,
                            explorer=args.explorer,
                            explorer_params=explorer_params,
                            initializer=initializer,
                            arg_params=arg_params,
                            aux_params=aux_params,
                            batch_end_callback=batch_end_callbacks,
                            epoch_end_callback=epoch_end_callbacks, #checkpoint if args.use_dali else ,,
                            allow_missing=True, 
                            eval_offset=args.eval_offset,
                            eval_period=args.eval_period,
                            accuracy_threshold=args.accuracy_threshold)

    if ('horovod' not in args.kv_store) or kv.rank == 0:
        arg_params, aux_params = model.get_params()
        model.set_params(arg_params, aux_params)
        model.save_checkpoint('MLPerf-RN50v15', last_epoch, save_optimizer_states=False)

    # When using horovod, ensure all ops scheduled by the engine complete before exiting
    if 'horovod' in args.kv_store:
        mx.ndarray.waitall()

    if args.profile_server_suffix:
        mx.profiler.set_state(state='run', profile_process='server')
    if args.profile_worker_suffix:
        mx.profiler.set_state(state='run', profile_process='worker')


