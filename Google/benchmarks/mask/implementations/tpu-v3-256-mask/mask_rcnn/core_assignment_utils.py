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
"""Core assignment utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


# The core assignments may change the performance of the model parallelism.
# An ideal case is to locate the op at the core that its input is located and
# avoid communication among cores.
CORE_0 = 0
CORE_1 = 1
CORE_2 = 2


def get_core_assignment(core_assignment, num_cores_per_replica=None):
  """Returns core assignment based on the number of cores in a replica.

  When the model runs with model parallelism (i.e., multiple cores for one
  replica), the core assignment is a modular of the number of available cores.
  When the model runs with out model parallelism (`num_cores_per_replica` is
  None), the function returns `None` so that device placement is a no-op.

  Args:
    core_assignment: An `int` that represents the core number.
    num_cores_per_replica: An `int` that represents the number of cores. `None`
      means no model parallelism.

  Returns:
    The core assignment based on whether the model runs with model parallelism
    and the number of cores per replica.
  """
  if num_cores_per_replica is not None:
    return tf.contrib.tpu.core(core_assignment % num_cores_per_replica)
  else:
    return None
