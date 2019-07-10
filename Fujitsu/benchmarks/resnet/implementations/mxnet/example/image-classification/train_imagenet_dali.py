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

import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, data, dali, fit
from common.util import download_file
import mxnet as mx
import numpy as np
from gluoncv.model_zoo import get_model
if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="train imagenet-1k",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    dali.add_dali_args(parser)
    data.add_data_aug_args(parser)
    # set to 0 for standard resnet augmentation
    data.set_data_aug_level(parser, 0)
    parser.set_defaults(
        # network
        network          = 'resnet',
        num_layers       = 50,
        # data
        num_classes      = 1000,
        num_examples     = 1281167,
        image_shape      = '4,224,224',
        min_random_scale = 1, # if input image has min size k, suggest to use
                              # 256.0/x, e.g. 0.533 for 480
        # train
        num_epochs       = 90,
        lr_step_epochs   = '30,60,80',
        dtype            = 'float32'
    )
    args = parser.parse_args()

    # load network
    from importlib import import_module
    if args.network == "resnet-v1b" and args.num_layers == 50:
        net = get_model('resnet50_v1b', ctx=[mx.gpu(int(i)) for i in args.gpus.split(',')],
                        pretrained=False, classes=args.num_classes, last_gamma=args.bn_gamma_init0)
        d = mx.sym.var('data')
        if args.dtype == 'float16':
            d = mx.sym.Cast(data=d, dtype=np.float16)
        net.cast(args.dtype)
        out = net(d)
        if args.dtype == 'float16':
            out = mx.sym.Cast(data=out, dtype=np.float32)
        sym = mx.sym.SoftmaxOutput(out, name='softmax')
    else:
        net = import_module('symbols.'+args.network)
        sym = net.get_symbol(**vars(args))

    # train
    fit.fit(args, sym, dali.get_rec_iter)

