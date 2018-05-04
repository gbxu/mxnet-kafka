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
from common import find_mxnet, data, fit
from common.util import download_file
import mxnet as mx


def resized_iter(args, kv=None):
    """
    for reduce the data size. Created by youhui Bai
    :param args:
    :param kv:
    :return:
    """
    data_iter, val_iter = data.get_rec_iter(args, kv)
    if kv:
        nworkers = kv.num_workers
    else:
        nworkers = 1
    return mx.io.ResizeIter(data_iter, args.num_examples // args.batch_size // nworkers), val_iter


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="train imagenet-1k",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    # use a large aug level
    data.set_data_aug_level(parser, 3)
    parser.set_defaults(
        # network
        network          = 'resnet',
        # num_layers       = 50,
        num_layers       = 18,
        # data
        num_classes      = 1000,
        #num_examples     = 1281167, #gbxu
        num_example      = 51200, # reduce the num example
        image_shape      = '3,224,224',
        min_random_scale = 1, # if input image has min size k, suggest to use
        # 256.0/x, e.g. 0.533 for 480
        # train
        num_epochs       = 80,
        lr_step_epochs   = '30,60',
        dtype            = 'float32'
    )
    args = parser.parse_args()

    # load network
    from importlib import import_module
    net = import_module('symbols.'+args.network)
    sym = net.get_symbol(**vars(args))

    # train
    fit.fit(args, sym, resized_iter) # reduce the examples
    # fit.fit(args, sym, data.get_rec_iter)
