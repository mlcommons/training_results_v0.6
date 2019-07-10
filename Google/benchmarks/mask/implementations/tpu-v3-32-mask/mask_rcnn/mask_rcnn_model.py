# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Model defination for the Mask-RCNN Model.

Defines model_fn of Mask-RCNN for TF Estimator. The model_fn includes Mask-RCNN
model architecture, loss function, learning rate schedule, and evaluation
procedure.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import six
import tensorflow as tf

import anchors
import core_assignment_utils
import fpn
import losses
import lr_policy
import mask_rcnn_architecture
import mask_rcnn_params
import post_processing
from mlp_log import mlp_log

_WEIGHT_DECAY = 1e-4


class MaskRcnnModelFn(object):
  """Mask-Rcnn model function."""

  def remove_variables(self, variables, resnet_depth=50):
    """Removes low-level variables from the training.

    Removing low-level parameters (e.g., initial convolution layer) from
    training usually leads to higher training speed and slightly better testing
    accuracy. The intuition is that the low-level architecture
    (e.g., ResNet-50) is able to capture low-level features such as edges;
    therefore, it does not need to be fine-tuned for the detection task.

    Args:
      variables: all the variables in training
      resnet_depth: the depth of ResNet model

    Returns:
      A list containing variables for training.

    """
    # Freeze at conv2 based on reference model.
    # Reference: https://github.com/ddkang/Detectron/blob/80f329530843e66d07ca39e19901d5f3e5daf009/lib/modeling/ResNet.py  # pylint: disable=line-too-long
    remove_list = []
    prefix = 'resnet{}/'.format(resnet_depth)
    remove_list.append(prefix + 'conv2d/')
    for i in range(1, 11):
      remove_list.append(prefix + 'conv2d_{}/'.format(i))

    # All batch normalization variables are frozen during training.
    def _is_kept(variable):
      return (all(rm_str not in variable.name for rm_str in remove_list) and
              'batch_normalization' not in variable.name)

    return list(filter(_is_kept, variables))

  def get_learning_rate(self, params, global_step):
    """Sets up learning rate schedule."""
    learning_rate = lr_policy.learning_rate_schedule(
        params['learning_rate'], params['lr_warmup_init'],
        params['lr_warmup_step'], params['first_lr_drop_step'],
        params['second_lr_drop_step'], global_step)
    mlp_log.mlperf_print(key='opt_base_learning_rate',
                         value=params['learning_rate'])
    mlp_log.mlperf_print(key='opt_learning_rate_warmup_steps',
                         value=params['lr_warmup_step'])
    mlp_log.mlperf_print(key='opt_learning_rate_warmup_factor',
                         value=params['lr_warmup_init']/params['learning_rate'])
    return learning_rate

  def get_optimizer(self, params, learning_rate):
    """Defines the optimizer."""
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, momentum=params['momentum'])
    if params['use_tpu']:
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
    return optimizer

  def get_scaffold_fn(self, params):
    """Loads pretrained model from checkpoint."""
    if params['resnet_checkpoint']:
      def scaffold_fn():
        """Loads pretrained model through scaffold function."""
        tf.train.init_from_checkpoint(params['resnet_checkpoint'], {
            '/': 'resnet%s/' % params['resnet_depth'],
        })
        return tf.train.Scaffold()
      return scaffold_fn
    else:
      return None

  def _model_outputs(self, features, labels, image_size, mode, params):
    """Generates outputs from the model."""
    all_anchors = anchors.Anchors(
        params['min_level'], params['max_level'], params['num_scales'],
        params['aspect_ratios'], params['anchor_scale'], image_size)

    if params['conv0_space_to_depth_block_size'] != 0:
      image_size = tuple(x // params['conv0_space_to_depth_block_size']
                         for x in image_size)

    if params['transpose_input']:
      images = tf.reshape(
          features['images'],
          [image_size[0], image_size[1], params['batch_size'], -1])
      images = tf.transpose(images, [2, 0, 1, 3])
    else:
      images = tf.reshape(
          features['images'],
          [params['batch_size'], image_size[0], image_size[1], -1])

    fpn_feats = fpn.resnet_fpn(images, params['min_level'],
                               params['max_level'], params['resnet_depth'],
                               params['conv0_kernel_size'],
                               params['conv0_space_to_depth_block_size'],
                               params['is_training_bn'])

    rpn_score_outputs, rpn_box_outputs = mask_rcnn_architecture.rpn_net(
        fpn_feats, params['min_level'], params['max_level'],
        len(params['aspect_ratios'] * params['num_scales']))

    if mode == tf.estimator.ModeKeys.PREDICT:
      # Use TEST.NMS in the reference for this value. Reference: https://github.com/ddkang/Detectron/blob/80f329530843e66d07ca39e19901d5f3e5daf009/lib/core/config.py#L227  # pylint: disable=line-too-long

      # The mask branch takes inputs from different places in training vs in
      # eval/predict. In training, the mask branch uses proposals combined
      # with labels to produce both mask outputs and targets. At test time,
      # it uses the post-processed predictions to generate masks.
      # Generate detections one image at a time.
      (class_outputs, box_outputs,
       box_rois) = mask_rcnn_architecture.faster_rcnn(
           fpn_feats, rpn_score_outputs, rpn_box_outputs, all_anchors,
           features['image_info'], mode, params)
      batch_size, _, _ = class_outputs.get_shape().as_list()
      detections = []
      softmax_class_outputs = tf.nn.softmax(class_outputs)
      for i in range(batch_size):
        detections.append(
            post_processing.generate_detections_per_image_op(
                softmax_class_outputs[i], box_outputs[i], box_rois[i],
                features['source_ids'][i], features['image_info'][i],
                params['test_detections_per_image'],
                params['test_rpn_post_nms_topn'], params['test_nms'],
                params['bbox_reg_weights'])
            )
      detections = tf.stack(detections, axis=0)
      mask_outputs = mask_rcnn_architecture.mask_rcnn(
          fpn_feats, mode, params, detections=detections)
      return {'detections': detections,
              'mask_outputs': mask_outputs}
    else:
      (class_outputs, box_outputs, box_rois, class_targets, box_targets,
       proposal_to_label_map) = mask_rcnn_architecture.faster_rcnn(
           fpn_feats, rpn_score_outputs, rpn_box_outputs, all_anchors,
           features['image_info'], mode, params, labels)
      encoded_box_targets = mask_rcnn_architecture.encode_box_targets(
          box_rois, box_targets, class_targets, params['bbox_reg_weights'])
      (mask_outputs, select_class_targets,
       mask_targets) = mask_rcnn_architecture.mask_rcnn(
           fpn_feats, mode, params, labels, class_targets, box_targets,
           box_rois, proposal_to_label_map)
      return {
          'rpn_score_outputs': rpn_score_outputs,
          'rpn_box_outputs': rpn_box_outputs,
          'class_outputs': class_outputs,
          'box_outputs': box_outputs,
          'class_targets': class_targets,
          'box_targets': encoded_box_targets,
          'box_rois': box_rois,
          'select_class_targets': select_class_targets,
          'mask_outputs': mask_outputs,
          'mask_targets': mask_targets,}

  def get_model_outputs(self, features, labels, image_size, mode, params):
    """A wrapper to generate outputs from the model.

    Args:
      features: the input image tensor and auxiliary information, such as
        `image_info` and `source_ids`. The image tensor has a shape of
        [batch_size, height, width, 3]. The height and width are fixed and
        equal.
      labels: the input labels in a dictionary. The labels include score targets
        and box targets which are dense label maps. See dataloader.py for more
        details.
      image_size: an integer tuple (height, width) representing the image shape.
      mode: the mode of TPUEstimator including TRAIN, EVAL, and PREDICT.
      params: the dictionary defines hyperparameters of model. The default
        settings are in default_hparams function in this file.

    Returns:
      The outputs from model (all casted to tf.float32).
    """

    if params['use_bfloat16']:
      with tf.contrib.tpu.bfloat16_scope():
        outputs = self._model_outputs(features, labels, image_size, mode,
                                      params)
        def _cast_outputs_to_float(d):
          for k, v in six.iteritems(d):
            if isinstance(v, dict):
              _cast_outputs_to_float(v)
            else:
              d[k] = tf.cast(v, tf.float32)
        _cast_outputs_to_float(outputs)
    else:
      outputs = self._model_outputs(features, labels, image_size, mode, params)
    return outputs

  def predict(self, features, labels, mode, params):
    """Generates predicitons."""
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
      def branch_fn(image_size):
        return self.get_model_outputs(
            features, labels, image_size, mode, params)

      model_outputs = tf.cond(
          tf.less(
              features['image_info'][0][3],
              features['image_info'][0][4]),
          lambda: branch_fn(params['image_size']),
          lambda: branch_fn(params['image_size'][::-1]))

      def scale_detections_to_original_image_size(detections, image_info):
        """Maps [y1, x1, y2, x2] -> [x1, y1, w, h] and scales detections."""
        batch_size, _, _ = detections.get_shape().as_list()
        image_ids, y_min, x_min, y_max, x_max, scores, classes = tf.split(
            value=detections, num_or_size_splits=7, axis=2)
        image_scale = tf.reshape(image_info[:, 2], [batch_size, 1, 1])
        scaled_height = (y_max - y_min) * image_scale
        scaled_width = (x_max - x_min) * image_scale
        scaled_y = y_min * image_scale
        scaled_x = x_min * image_scale
        detections = tf.concat(
            [image_ids, scaled_x, scaled_y, scaled_width, scaled_height, scores,
             classes],
            axis=2)
        return detections

      predictions = {}
      predictions['detections'] = scale_detections_to_original_image_size(
          model_outputs['detections'], features['image_info'])
      predictions['mask_outputs'] = tf.nn.sigmoid(model_outputs['mask_outputs'])
      predictions['image_info'] = features['image_info']
      if mask_rcnn_params.IS_PADDING in features:
        predictions[mask_rcnn_params.IS_PADDING] = features[
            mask_rcnn_params.IS_PADDING]
      else:
        predictions[mask_rcnn_params.IS_PADDING] = tf.constant(
            False, dtype=tf.bool, shape=[64])

      return predictions

  def get_loss(self, model_outputs, labels, params, var_list):
    """Generates the loss function."""
    # score_loss and box_loss are for logging. only total_loss is optimized.
    total_rpn_loss, rpn_score_loss, rpn_box_loss = losses.rpn_loss(
        model_outputs['rpn_score_outputs'], model_outputs['rpn_box_outputs'],
        labels, params)

    device = core_assignment_utils.get_core_assignment(
        core_assignment_utils.CORE_2, params['num_cores_per_replica'])
    with tf.device(device):
      total_fast_rcnn_loss, class_loss, box_loss = losses.fast_rcnn_loss(
          model_outputs['class_outputs'], model_outputs['box_outputs'],
          model_outputs['class_targets'], model_outputs['box_targets'], params)

    device = core_assignment_utils.get_core_assignment(
        core_assignment_utils.CORE_1, params['num_cores_per_replica'])
    with tf.device(device):
      mask_loss = losses.mask_rcnn_loss(
          model_outputs['mask_outputs'], model_outputs['mask_targets'],
          model_outputs['select_class_targets'], params)

    l2_weight_loss = _WEIGHT_DECAY * tf.add_n([
        tf.nn.l2_loss(v)
        for v in var_list
        if 'batch_normalization' not in v.name and 'bias' not in v.name
    ])
    total_loss = (total_rpn_loss + total_fast_rcnn_loss + mask_loss +
                  l2_weight_loss)

    return [total_loss, mask_loss, total_fast_rcnn_loss, class_loss, box_loss,
            total_rpn_loss, rpn_score_loss, rpn_box_loss]

  def train_op(self, features, labels, image_size, mode, params):
    """Generates train op."""
    model_outputs = self.get_model_outputs(
        features, labels, image_size, mode, params)

    var_list = self.remove_variables(tf.trainable_variables(),
                                     params['resnet_depth'])
    all_losses = self.get_loss(model_outputs, labels, params, var_list)
    global_step = tf.train.get_or_create_global_step()
    learning_rate = self.get_learning_rate(params, global_step)
    optimizer = self.get_optimizer(params, learning_rate)

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    grads_and_vars = optimizer.compute_gradients(
        all_losses[0], var_list, colocate_gradients_with_ops=True)
    gradients, variables = zip(*grads_and_vars)
    grads_and_vars = []
    # Special treatment for biases (beta is named as bias in reference model)
    # Reference: https://github.com/ddkang/Detectron/blob/80f329530843e66d07ca39e19901d5f3e5daf009/lib/modeling/optimizer.py#L109  # pylint: disable=line-too-long
    for grad, var in zip(gradients, variables):
      if 'beta' in var.name or 'bias' in var.name:
        grad = 2.0 * grad
      grads_and_vars.append((grad, var))
    minimize_op = optimizer.apply_gradients(grads_and_vars,
                                            global_step=global_step)

    with tf.control_dependencies(update_ops):
      train_op = minimize_op
    return all_losses, train_op

  def train(self, features, labels, mode, params):
    """A wrapper for tf.cond."""

    with tf.variable_scope('', reuse=tf.AUTO_REUSE):

      def branch_fn(image_size):
        return self.train_op(
            features, labels, image_size, mode, params)

      (all_losses, train_op) = tf.cond(
          tf.less(features['image_info'][0][3], features['image_info'][0][4]),
          lambda: branch_fn(params['image_size']),
          lambda: branch_fn(params['image_size'][::-1]))
      return all_losses, train_op

  def __call__(self, features, labels, mode, params):
    """Model defination for the Mask-RCNN model based on ResNet.

    Args:
      features: the input image tensor and auxiliary information, such as
        `image_info` and `source_ids`. The image tensor has a shape of
        [batch_size, height, width, 3]. The height and width are fixed and
        equal.
      labels: the input labels in a dictionary. The labels include score targets
        and box targets which are dense label maps. The labels are generated
        from get_input_fn function in data/dataloader.py
      mode: the mode of TPUEstimator including TRAIN, EVAL, and PREDICT.
      params: the dictionary defines hyperparameters of model. The default
        settings are in default_hparams function in this file.

    Returns:
      TPUEstimatorSpec to run training or prediction.
    Raises:
      If `mode` is not tf.estimator.ModeKeys.TRAIN or PREDICT.
    """
    if mode not in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.PREDICT):
      raise ValueError('MaskRcnnModelFn supports either TRAIN or PREDICT.')

    if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = self.predict(features, labels, mode, params)
      if params['use_tpu']:
        return tf.contrib.tpu.TPUEstimatorSpec(mode=mode,
                                               predictions=predictions)
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    elif mode == tf.estimator.ModeKeys.TRAIN:
      all_losses, train_op = self.train(features, labels, mode, params)

      if params['use_host_call']:
        global_step = tf.train.get_or_create_global_step()
        # To log the loss, current learning rate, and epoch for Tensorboard, the
        # summary op needs to be run on the host CPU via host_call. host_call
        # expects [batch_size, ...] Tensors, not scalar. Reshape the losses
        # tensors to [1] tensors. These Tensors are implicitly concatenated to
        # [batch_size, ...].
        global_step_t = tf.reshape(global_step, [1])
        for i, loss in enumerate(all_losses):
          all_losses[i] = tf.reshape(loss, [1])
        host_call_func = functools.partial(host_call_fn, params['model_dir'],
                                           params['iterations_per_loop'])
        all_losses.append(global_step_t)
        host_call = (host_call_func, all_losses)
      else:
        host_call = None

      return tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=all_losses[0],
          train_op=train_op,
          host_call=host_call,
          scaffold_fn=self.get_scaffold_fn(params))


def host_call_fn(model_dir, iterations_per_loop, total_loss, total_rpn_loss,
                 rpn_score_loss, rpn_box_loss, total_fast_rcnn_loss,
                 fast_rcnn_class_loss, fast_rcnn_box_loss, mask_loss,
                 global_step):
  """Training host call. Creates scalar summaries for training metrics.

  This function is executed on the CPU and should not directly reference
  any Tensors in the rest of the `model_fn`. To pass Tensors from the
  model to the `metric_fn`, provide as part of the `host_call`. See
  https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
  for more information.

  Arguments should match the list of `Tensor` objects passed as the second
  element in the tuple passed to `host_call`.

  Args:
    model_dir: The directory of summaries.
    iterations_per_loop: Maximum queued summaries.
    total_loss: `Tensor` with shape `[batch, ]` for the training loss.
    total_rpn_loss: `Tensor` with shape `[batch, ]` for the training RPN
      loss.
    rpn_score_loss: `Tensor` with shape `[batch, ]` for the training RPN
      score loss.
    rpn_box_loss: `Tensor` with shape `[batch, ]` for the training RPN
      box loss.
    total_fast_rcnn_loss: `Tensor` with shape `[batch, ]` for the
      training Mask-RCNN loss.
    fast_rcnn_class_loss: `Tensor` with shape `[batch, ]` for the
      training Mask-RCNN class loss.
    fast_rcnn_box_loss: `Tensor` with shape `[batch, ]` for the
      training Mask-RCNN box loss.
    mask_loss: `Tensor` with shape `[batch, ]` for the training Mask-RCNN
      mask loss.
    global_step: `Tensor with shape `[batch, ]` for the global_step.

  Returns:
    List of summary ops to run on the CPU host.
  """
  # Outfeed supports int32 but global_step is expected to be int64.
  global_step = tf.reduce_mean(global_step)
  # Host call fns are executed FLAGS.iterations_per_loop times after one
  # TPU loop is finished, setting max_queue value to the same as number of
  # iterations will make the summary writer only flush the data to storage
  # once per loop.
  with (tf.contrib.summary.create_file_writer(
      model_dir, max_queue=iterations_per_loop).as_default()):
    with tf.contrib.summary.always_record_summaries():
      tf.contrib.summary.scalar(
          'loss_total', tf.reduce_mean(total_loss), step=global_step)
      tf.contrib.summary.scalar(
          'loss_rpn_total', tf.reduce_mean(total_rpn_loss),
          step=global_step)
      tf.contrib.summary.scalar(
          'loss_rpn_score', tf.reduce_mean(rpn_score_loss),
          step=global_step)
      tf.contrib.summary.scalar(
          'loss_rpn_box', tf.reduce_mean(rpn_box_loss), step=global_step)
      tf.contrib.summary.scalar(
          'loss_fast_rcnn_total', tf.reduce_mean(total_fast_rcnn_loss),
          step=global_step)
      tf.contrib.summary.scalar(
          'loss_fast_rcnn_class', tf.reduce_mean(fast_rcnn_class_loss),
          step=global_step)
      tf.contrib.summary.scalar(
          'loss_fast_rcnn_box', tf.reduce_mean(fast_rcnn_box_loss),
          step=global_step)
      tf.contrib.summary.scalar(
          'loss_mask', tf.reduce_mean(mask_loss), step=global_step)

      return tf.contrib.summary.all_summary_ops()
