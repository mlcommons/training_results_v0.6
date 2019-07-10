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

from __future__ import print_function
import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
import mxnet as mx
from common import find_mxnet, modelzoo, data, fit
from score import score_trt
from common.util import download_file


VAL_DATA='cifar10_val.rec'
def download_data():
    return mx.test_utils.download(
        'http://data.mxnet.io/data/cifar10/cifar10_val.rec', VAL_DATA)

def test_cifar10_resnet(**kwargs):
    acc = mx.metric.create('acc')
    m = './resnet50'
    g = 0.89
    (speed,) = score_trt(model=m,
                     data_val=VAL_DATA,
                     rgb_mean='123.68,116.779,103.939', metrics=acc, **kwargs)
    r = acc.get()[1]
    print('Tested %s acc = %f, speed = %f img/sec' % (m, r, speed))
    assert r > g and r < g + .1

if __name__ == '__main__':
    gpus = '1'
    assert len(gpus) > 0
    batch_size = 32 * len(gpus)
    gpus = ','.join([str(i) for i in gpus])

    kwargs = {'gpus':gpus, 'batch_size':batch_size, 'max_num_examples':500}
    download_data()
    test_cifar10_resnet(**kwargs)
