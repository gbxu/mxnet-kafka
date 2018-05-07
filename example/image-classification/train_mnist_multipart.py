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

"""
Train mnist, see more explanation at http://mxnet.io/tutorials/python/mnist.html
"""
import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, fit
from common.util import download_file
import mxnet as mx
import numpy as np
import gzip, struct


def download_upzip_data(label, image, ulabel, uimage):
    """
    download and read data into numpy
    """
    ufl = os.path.join('data',ulabel)
    ufi = os.path.join('data',uimage)
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    flbl = gzip.open(download_file(base_url+label, os.path.join('data',label)))
    fimg = gzip.open(download_file(base_url+image, os.path.join('data',image)), 'rb')
    ulabel_out = open(ufl, 'wb')
    ulabel_out.write(flbl.read())
    ulabel_out.close()
    flbl.close()

    uimage_out = open(ufi, 'wb')
    uimage_out.write(fimg.read())
    uimage_out.close()
    fimg.close()
    return (ufl, ufi)

def get_mnist_iter(args, kv):
    """
    create data iterator with NDArrayIter
    """
    (train_lbl, train_img) = download_upzip_data(
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte', 'train-images-idx3-ubyte')
    (val_lbl, val_img) = download_upzip_data(
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte', 't10k-images-idx3-ubyte')
    # multi parts
    if kv:
        (rank, nworker) = (kv.rank, kv.num_workers)
    else:
        (rank, nworker) = (0, 1)

    train = mx.io.MNISTIter(
        image               = train_img,
        label               = train_lbl,
        batch_size          = args.batch_size,
        num_parts           = nworker,
        part_index          = rank
    )
    val = mx.io.MNISTIter(
        image               = val_img,
        label               = val_lbl,
        batch_size          = args.batch_size,
        num_parts           = nworker,
        part_index          = rank
    )
    return (train, val)

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="train mnist",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-classes', type=int, default=10,
                        help='the number of classes')
    parser.add_argument('--num-examples', type=int, default=60000,
                        help='the number of training examples')

    parser.add_argument('--add_stn',  action="store_true", default=False, help='Add Spatial Transformer Network Layer (lenet only)')

    fit.add_fit_args(parser)
    parser.set_defaults(
        # network
        network        = 'mlp',
        # train
        num_examples  = 60000,
        gpus           = None,
        batch_size     = 64,
        disp_batches   = 100,
        num_epochs     = 20,
        lr             = .05,
        lr_step_epochs = '10'
    )
    args = parser.parse_args()

    # load network
    from importlib import import_module
    net = import_module('symbols.'+args.network)
    sym = net.get_symbol(**vars(args))

    # train
    fit.fit(args, sym, get_mnist_iter)
