# Copyright 2018 Google. All Rights Reserved.
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
"""Use multiprocess to perform COCO metric evaluation.
"""
import multiprocessing
import six

import mask_rcnn_params
import segm_utils


def post_processing(q_in, q_out):
  """Batch-processes the predictions."""
  boxes, masks, image_info, is_padding = q_in.get()
  while boxes is not None:
    detections = []
    segmentations = []
    for i, box in enumerate(boxes):
      if is_padding[i]:
        continue
      segms = segm_utils.segm_results(
          masks[i], box[:, 1:5], int(image_info[i][3]), int(image_info[i][4]))
      detections.extend(box)
      segmentations.append(segms)
    q_out.put((detections, segmentations))
    boxes, masks, image_info, is_padding = q_in.get()
  # signal the parent process that we have completed all work.
  q_out.put((None, None))


def update_eval_metric(q_out, eval_metric, exited_process):
  detections, segmentations = q_out.get()
  if detections is None and segmentations is None:
    exited_process += 1
  else:
    eval_metric.update(detections, segmentations)
  return exited_process


def eval_multiprocessing(num_batches,
                         predictor,
                         eval_metric,
                         eval_worker_count,
                         queue_size=mask_rcnn_params.QUEUE_SIZE):
  """Enables multiprocessing to update eval metrics."""
  # pylint: disable=line-too-long
  q_in = multiprocessing.Queue(maxsize=queue_size)
  q_out = multiprocessing.Queue(maxsize=queue_size)
  processes = [
      multiprocessing.Process(target=post_processing, args=(q_in, q_out))
      for _ in range(eval_worker_count)
  ]
  # pylint: enable=line-too-long
  for p in processes:
    p.start()

  # TODO(b/129410706): investigate whether threading improves speed.
  # Every predictor.next() gets a batch of prediction (a dictionary).
  exited_process = 0
  for _ in range(num_batches):
    while q_in.full() or q_out.qsize() > queue_size // 4:
      exited_process = update_eval_metric(q_out, eval_metric, exited_process)

    predictions = six.next(predictor)

    q_in.put((predictions['detections'],
              predictions['mask_outputs'],
              predictions['image_info'],
              predictions[mask_rcnn_params.IS_PADDING]))

  # Adds empty items to signal the children to quit.
  for _ in processes:
    q_in.put((None, None, None, None))

  # Cleans up q_out and waits for all the processes to finish work.
  while not q_out.empty() or exited_process < eval_worker_count:
    exited_process = update_eval_metric(q_out, eval_metric, exited_process)

  for p in processes:
    # actively terminate all processes (to work around the multiprocessing
    # deadlock issue in Cloud)
    p.terminate()
    p.join()
