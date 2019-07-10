# coding: utf-8
# Copyright FUJITSU LIMITED 2019

# pylint: disable=too-many-lines
"""Gradient exploring functions."""
import logging
import math
import pickle
import warnings
import os
from ..optimizer import optimizer

__all__ = [
    'Explorer', 'Linear','register','create'
]

class Explorer(object):
    def __init__(self, optimizer=None, explorer_params=None):
        self.start_epoch = explorer_params['start_epoch']
        self.end_epoch = explorer_params['end_epoch']
        self.epoch_size = explorer_params['epoch_size'] # number of iterations per epoch
        self.lr_decay = explorer_params['lr_decay']
        self.wd_decay = explorer_params['wd_decay']
        self.base_lr_range_max = explorer_params['lr_range_max']
        self.cur_lr_range_max = self.base_lr_range_max
        self.lr_range_max = self.base_lr_range_max
        self.base_lr_range_min = explorer_params['lr_range_min']
        self.cur_lr_range_min = self.base_lr_range_min
        self.lr_range_min = self.base_lr_range_min
        self.base_wd_range_max = explorer_params['wd_range_max']
        self.wd_range_max = self.base_wd_range_max
        self.cur_wd_range_max = self.base_wd_range_max
        self.base_wd_range_min = explorer_params['wd_range_min']
        self.wd_range_min = self.base_wd_range_min
        self.cur_wd_range_min = self.base_wd_range_min
        self.burn_in_iter = explorer_params['burn_in_iter']
        self.burn_in_ratio = 1.0
        self.current_iter = 0
        self.wd_factor = explorer_params['wd_factor']
        self.wd_step_epochs = explorer_params['wd_step_epochs']
        self.current_wd_factor = 1.0
        self.wd_warmup_ratio = 1.0
        self.wd_warmup = explorer_params['wd_warmup']
        self.warmup_step = explorer_params['warmup_step']
        self.ds_upper_factor = explorer_params['ds_upper_factor']
        self.ds_lower_factor = explorer_params['ds_lower_factor']
        self.ds_fix_max = explorer_params['ds_fix_max']
        self.ds_fix_min = explorer_params['ds_fix_min']
        self.natan_turn_epoch = explorer_params['natan_turn_epoch']
        if self.wd_warmup and self.warmup_step > 0:
            self.warmup_stride = 1.0 / self.warmup_step
            #self.warmup_stride = explorer_params['wd_base'] / self.warmup_step
        else:
            self.warmup_stride = 0
        self.warmup_epochs = explorer_params['warmup_epochs']
        self.lr_stride_range = self.lr_range_max - self.lr_range_min
        self.wd_stride_range = self.wd_range_max - self.wd_range_min
        assert(self.lr_stride_range >= 0)
        assert(self.wd_stride_range >= 0)
        self.comm_world_rank = int(os.getenv('OMPI_COMM_WORLD_RANK', 0))
        self.comm_world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE', 1))

        num_grps = explorer_params['num_grps']
        num_cg_grps = explorer_params['num_cg_grps']

        if num_grps == 0:
            self.same_wd_coeff = True
        else:
            self.same_wd_coeff = False

        if self.same_wd_coeff:
            assert(self.comm_world_size % num_cg_grps == 0)
            num_grps = 1
            self.lr_size = self.comm_world_size / num_cg_grps
            self.wd_size = self.lr_size
            self.cg_size = num_cg_gps
        else:
            assert(self.comm_world_size % (num_grps * num_cg_grps) == 0)
            self.lr_size = self.comm_world_size / (num_grps * num_cg_grps)
            self.wd_size = num_grps
            self.cg_size = num_cg_grps

        # ranks = self._get_ranks(self.comm_world_rank)
        # print ("(%d,%d,%d) in (%d,%d,%d)" % (self._get_lr_rank(self.comm_world_rank),
        #                                      self._get_wd_rank(self.comm_world_rank),
        #                                      self._get_cg_rank(self.comm_world_rank),
        #                                      self.lr_size, self.wd_size, self.cg_size))

        self.optimizer = optimizer

        self.decay_ratio = 1.0
        self.wd_decay_ratio = 1.0

        self.base_cg_range_min = explorer_params['cg_range_min']
        self.cur_cg_range_min = self.base_cg_range_min
        self.base_cg_range_max = explorer_params['cg_range_max']
        self.cur_cg_range_max = self.base_cg_range_max
        self.cg_stride_range = self.base_cg_range_max - self.base_cg_range_min

        self.print_values = int(os.getenv('PRINT_FJSGD_VALUES',0)) != 0 and int(os.getenv('OMPI_COMM_WORLD_RANK', 0)) == 0

        self.natan_final_ratio = explorer_params['natan_final_ratio']
        self.explore_start_epoch = explorer_params['explore_start_epoch']
        self.momentum = explorer_params['momentum']
        self.momentum_end = explorer_params['momentum_end']
        self.lr_decay_mode = explorer_params['lr_decay_mode']
        self.wd_decay_mode = explorer_params['wd_decay_mode']
        self.total_iter = explorer_params['total_steps'] #(self.end_epoch - self.start_epoch) * self.epoch_size

        self.decay_after_warmup = explorer_params['decay_after_warmup']
        self.end_lr_ratio = explorer_params['end_lr_ratio']
        self.decay_steps = explorer_params['decay_steps']
        self.decay_epochs = explorer_params['decay_epochs']

    exp_registry = {}

    @staticmethod
    def register(klass):
        assert(isinstance(klass, type))
        name = klass.__name__.lower()

        if name in Explorer.exp_registry:
            warnings.warn('WARNING: New optimizer %s.%s is overriding '
                          'existing explorer %s.%s' %
                          (klass.__module__, klass.__name__,
                           Explorer.exp_registry[name].__module__,
                           Explorer.exp_registry[name].__name__))
        Explorer.exp_registry[name] = klass
        return klass

    @staticmethod
    def create_explorer(name, optimizer, explorer_params=None):
        if name.lower() in Explorer.exp_registry:
            return Explorer.exp_registry[name.lower()](optimizer,
                                                       explorer_params=explorer_params)
        else:
            raise ValueError('Cannot find explorer %s' % name)

    def _get_lr_rank(self, world_rank):
        if self.same_wd_coeff:
            lr_rank = world_rank % self.lr_size
        else:
            lr_rank = world_rank % self.lr_size
        return lr_rank

    def _get_wd_rank(self, world_rank):
        if self.same_wd_coeff:
            wd_rank = world_rank % self.lr_size
        else:
            wd_rank = (world_rank // self.lr_size) % self.wd_size
        return wd_rank

    def _get_cg_rank(self, world_rank):
        if self.same_wd_coeff:
            cg_rank = world_rank // self.lr_size
        else:
            cg_rank = world_rank // (self.lr_size * self.wd_size)
        return cg_rank

    def decay_natan_base(self, decay, df):
        if df > 0:
            df = df * 2
        rad = (df / decay) + ( df * df * df  / 1000.0)
        ratio = 0.5 - (math.atan(rad) / math.pi)
        return ratio

    def decay_natan(self, decay, current, total, turn, is_iter):
        df = (current - turn)
        if is_iter:
            df = df / self.epoch_size
        ratio = self.decay_natan_base(decay, df)
        if self.natan_final_ratio >= 0.0 and current >= turn:
            df_end = total - turn
            if is_iter:
                df_end = df_end / self.epoch_size
            coeff = (0.5 - self.natan_final_ratio) / (0.5 - self.decay_natan_base(decay, df_end))
            ratio = 0.5 - (0.5 - ratio) * coeff
        return ratio

    def decay_linear(self, decay, current, total):
        p = current / total
        ratio = 1.0 - (1.0 - (1.0 / decay) ) * p
        return ratio

    def decay_power(self, decay, current, total):
        # tf.train.polynomial_decay
        # https://www.tensorflow.org/api_docs/python/tf/train/polynomial_decay
        #
        # global_step = min(global_step, decay_steps)
        # decayed_learning_rate = (learning_rate - end_learning_rate) *
        #                         (1 - global_step / decay_steps) ^ (power) +
        #                         end_learning_rate

        assert(current <= total)
        p = current / total
        return (1.0 - self.end_lr_ratio) * math.pow(1.0 - p, decay) + self.end_lr_ratio

    def wd_decay_step(self, current, steps):
        if current in steps:
            self.current_wd_factor = self.current_wd_factor * self.wd_factor
        return self.current_wd_factor

    def apply_lr_decay(self, is_iter, current, total):
        if current > total:
            current = total

        if self.lr_decay_mode == None:
            return
        elif 'step' in self.wd_decay_mode:
            return
        elif 'linear' in self.lr_decay_mode:
            decay_ratio = self.decay_linear(self.lr_decay, current, total)
        elif 'power' in self.lr_decay_mode:
            decay_ratio = self.decay_power(self.lr_decay, current, total)
        elif 'natan' in self.lr_decay_mode:
            turn = self.natan_turn_epoch
            if self.decay_after_warmup:
                turn = turn - self.warmup_epochs
            if is_iter:
                turn = turn * self.epoch_size
            decay_ratio = self.decay_natan(self.lr_decay, current, total, turn, is_iter)
        else:
            assert(False)
        if self.current_iter > self.warmup_step:
            # this is not need for power
            self.decay_ratio = max(self.end_lr_ratio, decay_ratio)
        else:
            self.decay_ratio = decay_ratio

    def apply_wd_decay(self, is_iter, current, total):
        if current > total:
            current = total

        if self.wd_decay_mode == None:
            return
        elif 'step' in self.wd_decay_mode:
            steps = self.wd_step_epochs
            if is_iter:
                steps = [i * self.epoch_size for i in steps]
            decay_ratio = self.wd_decay_step(current, steps)
        elif 'linear' in self.wd_decay_mode:
            decay_ratio = self.decay_linear(self.wd_decay, current, total)
        elif 'power' in self.wd_decay_mode:
            decay_ratio = self.decay_power(self.wd_decay,current, total)
        elif 'natan' in self.wd_decay_mode:
            turn = self.natan_turn_epoch
            if self.decay_after_warmup:
                turn = turn - self.warmup_epochs
            if is_iter:
                turn = turn * self.epoch_size
            decay_ratio = self.decay_natan(self.wd_decay, current, total, turn, is_iter)
        else:
            assert(False)
        if self.current_iter > self.warmup_step:
            # this is not need for power
            self.wd_decay_ratio = max(self.end_lr_ratio, decay_ratio)
        else:
            self.wd_decay_ratio = decay_ratio

    def apply_lr_decay_iter(self):
        decay_cur_step = self.current_iter
        if self.decay_after_warmup:
            decay_cur_step -= self.warmup_step
            if decay_cur_step < 0:
                decay_cur_step = 0
        self.apply_lr_decay(True, decay_cur_step, self.decay_steps)

    def apply_lr_decay_epoch(self, current_epoch):
        decay_cur_epoch = current_epoch - self.start_epoch
        if self.decay_after_warmup:
            decay_cur_epoch -= self.warmup_epochs
            if decay_cur_epoch < 0:
                decay_cur_epoch = 0
        self.apply_lr_decay(False, decay_cur_epoch, self.decay_epochs)

    def apply_wd_decay_iter(self):
        decay_cur_step = self.current_iter
        if self.decay_after_warmup:
            decay_cur_step -= self.warmup_step
            if decay_cur_step < 0:
                decay_cur_step = 0
        self.apply_wd_decay(True, decay_cur_step, self.decay_steps)

    def apply_wd_decay_epoch(self, current_epoch):
        decay_cur_epoch = current_epoch - self.start_epoch
        if self.decay_after_warmup:
            decay_cur_epoch -= self.warmup_epochs
            if decay_cur_epoch < 0:
                decay_cur_epoch = 0
        self.apply_wd_decay(False, decay_cur_epoch, self.decay_epochs)

    def apply_wd_warmup(self):
        if self.wd_warmup == True:
           if self.current_iter < self.warmup_step:
              self.wd_warmup_ratio = self.current_iter * self.warmup_stride
        else:
           self.wd_warmup_ratio = 1.0

    def apply_burn_in(self):
        if self.current_iter < self.burn_in_iter:
           self.burn_in_ratio = math.pow(self.current_iter / self.burn_in_iter,4)
        else:
           self.burn_in_ratio = 1.0
        self.current_iter += 1

    def set_tmp_coeff(self, rank=0):
        coeff = self.coeff(rank) * self.burn_in_ratio * self.decay_ratio
        self.optimizer._set_tmp_coeff(coeff)

    def set_tmp_wd_coeff(self, rank=0):
        wd_coeff = self.coeff(rank) if self.same_wd_coeff else self.wd_coeff(rank)
        wd_coeff = wd_coeff * self.burn_in_ratio * self.wd_warmup_ratio * self.wd_decay_ratio
        self.optimizer._set_tmp_wd_coeff(wd_coeff)

    def set_tmp_cg(self, rank=0):
        coeff = self.cg_coeff(rank)
        self.optimizer._set_cg(coeff)

    def set_best_coeff(self, rank=0):
        base_coeff = self.coeff(rank)
        coeff = base_coeff * self.burn_in_ratio * self.decay_ratio

        self.optimizer._set_best_coeff(coeff)

    def set_best_wd_coeff(self, rank=0):
        base_wd_coeff = self.coeff(rank) if self.same_wd_coeff else self.wd_coeff(rank)
        wd_coeff = base_wd_coeff * self.burn_in_ratio * self.wd_warmup_ratio * self.wd_decay_ratio

        self.optimizer._set_best_wd_coeff(wd_coeff)

    def set_best_cg(self, rank=0):
        coeff = self.cg_coeff(rank)

        self.optimizer._set_cg(coeff)

    def set_mom(self, current_epoch):
        if self.momentum == self.momentum_end:
            return

        p = (current_epoch - self.start_epoch ) / (self.end_epoch - self.start_epoch)
        mom = self.momentum + (self.momentum_end - self.momentum) * p
        if self.print_values:
            logging.info('set momentum {:.10e}'.format(mom))
        self.optimizer.momentum = mom


    def coeff(self, rank=0):
        raise NotImplementedError()

    def wd_coeff(self, rank=0):
        raise NotImplementedError()

    def cg_coeff(self, rank=0):
        raise NotImplementedError()



# convenience wrapper for Explorer.Register
register = Explorer.register   # pylint: disable=invalid-name
create = Explorer.create_explorer  # pylint: disable=invalid-name

# pylint: disable=line-too-long
@register
class Linear(Explorer):
    def __init__(self,optimizer,explorer_params=None):
        super(Linear, self).__init__(optimizer=optimizer,explorer_params=explorer_params)
        self.set_tmp_coeff(self.comm_world_rank)
        self.set_tmp_wd_coeff(self.comm_world_rank)
        self.set_tmp_cg(self.comm_world_rank)

    def coeff(self, rank=0):
        target_rank = self._get_lr_rank(rank)
        return self.linear_ratio(target_rank, self.lr_size) * self.lr_stride_range + self.cur_lr_range_min

    def wd_coeff(self, rank=0):
        target_rank = self._get_wd_rank(rank)
        return self.linear_ratio(target_rank, self.wd_size) * self.wd_stride_range + self.cur_wd_range_min

    def cg_coeff(self, rank=0):
        target_rank = self._get_cg_rank(rank)
        return self.linear_ratio(target_rank, self.cg_size) * self.cg_stride_range + self.cur_cg_range_min

    def linear_ratio(self, rank, size):
        assert(0 <= rank and rank < size)
        return (rank + 1) / size
