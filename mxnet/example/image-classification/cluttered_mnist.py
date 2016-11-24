# Copyright 2016 MXNET LiuHao. All Rights Reserved.
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
# =============================================================================
import find_mxnet
import mxnet as mx
import argparse
import os, sys
import train_model
import numpy as np
import scipy.io as si
import importlib


#def get_loc(data, attr={'lr_mult':'0.01'}):
def get_loc(data):
    """
    the localisation network in lenet-stn, it will increase acc about more than 1%,
    when num-epoch >=15
    """
    # Downsampling
    loc = mx.symbol.Pooling(data=data, kernel=(2, 2), stride=(2, 2), pool_type='avg')
    
    # fully convolutional network
    loc = mx.symbol.Convolution(data= loc, num_filter=20, kernel=(5, 5), stride=(1,1))
    loc = mx.symbol.Activation(data = loc, act_type='relu')
    loc = mx.symbol.Pooling(data= loc, kernel=(2, 2), stride=(2, 2), pool_type='max')

    loc = mx.symbol.Convolution(data= loc, num_filter=20, kernel=(5, 5), stride=(1,1))
    loc = mx.symbol.Activation(data = loc, act_type='relu')
    #loc = mx.symbol.Pooling(data= loc, kernel=(2, 2), stride=(2, 2), pool_type='max')
    
    #loc = mx.symbol.Convolution(data= loc, num_filter=50, kernel=(9, 9), stride=(1,1))
    #loc = mx.symbol.Activation(data = loc, act_type='relu')
    loc = mx.symbol.Flatten(data= loc)
    loc = mx.symbol.FullyConnected(data=loc, num_hidden=50)
    loc = mx.symbol.Activation(data = loc, act_type='relu')
    loc = mx.symbol.FullyConnected(data=loc, num_hidden=6)
    return loc


def get_loc2(data):
    """
    the localisation network in lenet-stn, it will increase acc about more than 1%,
    when num-epoch >=15
    """
    # place holder
    #data = mx.symbol.Variable('data') 
    # average pooling  60/2 =30
    #avg_pool = mx.symbol.Pooling(data=data, pool_type="avg",
                              #kernel=(2,2), stride=(2,2))
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(7,7), num_filter=32)
    relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    pool1 = mx.symbol.Pooling(data=relu1, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=48)
    relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # first fullc
    flatten = mx.symbol.Flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=256)
    relu3 = mx.symbol.Activation(data=fc1, act_type="relu")
    # second fullc
    fc2 = mx.symbol.FullyConnected(data=relu3, num_hidden=6)
    
    return fc2

def get_mlp():
    """
    multi-layer perceptron
    """
    data = mx.symbol.Variable('data')
    fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
    fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
    fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)
    mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')
    return mlp

def get_lenet(add_stn=True):
    """
    LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick
    Haffner. "Gradient-based learning applied to document recognition."
    Proceedings of the IEEE (1998)
    """
    # place holderFalse
    data = mx.symbol.Variable('data') 
    if(add_stn):
        data = mx.sym.SpatialTransformer(data=data, loc=get_loc(data), target_shape = (60,60),
                                         transform_type="affine", sampler_type="bilinear")
    
    # average pooling  60/2 =30
    avg_pool = mx.symbol.Pooling(data=data, pool_type="avg",
                              kernel=(2,2), stride=(2,2))
    # first conv
    conv1 = mx.symbol.Convolution(data=avg_pool, kernel=(7,7), num_filter=32)
    relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    pool1 = mx.symbol.Pooling(data=relu1, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=48)
    relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # first fullc
    flatten = mx.symbol.Flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=256)
    relu3 = mx.symbol.Activation(data=fc1, act_type="relu")
    
    # second fullc
    fc2 = mx.symbol.FullyConnected(data=relu3, num_hidden=10)
    
    # loss
    lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    return lenet
False
def get_iterator(data_shape):

    def to4d(img):
        return np.transpose(img,(3,2,1,0))
        
    def to1d(label):
        return label.reshape(label.shape[1])

    
    def get_iterator_impl(args, kv):
    
        mat_data = si.loadmat('./mnist/cluttered-mnist.mat') # dict
        X_train = mat_data['x_tr']
        y_train = mat_data['y_tr']
        X_valid = mat_data['x_vl']
        y_valid = mat_data['y_vl']
    
        train = mx.io.NDArrayIter(
            to4d(X_train),
            to1d(y_train),
            args.batch_size)
            #shuffle = True)

        val = mx.io.NDArrayIter(
            to4d(X_valid),
            to1d(y_valid),
            args.batch_size)

        return (train, val)
    return get_iterator_impl

def parse_args():
    parser = argparse.ArgumentParser(description='train an image classifer on mnist')
    parser.add_argument('--network', type=str, default='lenet-stn',
                        choices = ['mlp', 'lenet', 'lenet-stn'],
                        help = 'the cnn to use')
    parser.add_argument('--data-dir', type=str, default='mnist/',
                        help='the input data directory')
    parser.add_argument('--gpus', type=str, default='0',
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--num-examples', type=int, default=50000,
                        help='the number of training examples')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='the batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='the initial learning rate')
    parser.add_argument('--model-prefix', type=str, 
                        help='the prefix of the model to load/save')
    parser.add_argument('--save-model-prefix', type=str, 
                        help='the prefix of the model to save')
    parser.add_argument('--num-epochs', type=int, default=60,
                        help='the number of training epochs')
    parser.add_argument('--load-epoch', type=int, 
                        help="load the model on an epoch using the model-prefix")
    parser.add_argument('--kv-store', type=str, default='local',
                        help='the kvstore type')
    parser.add_argument('--lr-factor', type=float, default=1,
                        help='times the lr with a factor for every lr-factor-epoch epoch')
    parser.add_argument('--lr-factor-epoch', type=float, default=1,
                        help='the number of epoch to factor the lr, could be .5')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.network == 'mlp':
        data_shape = (784, )
        net = get_mlp()
    elif args.network == 'lenet-stn':
        data_shape = (1, 60, 60)
        net = get_lenet(True)
    else:
        data_shape = (1, 60, 60)
        net = get_lenet()
    
    #net = importlib.import_module("symbol_resnet").get_symbol(num_class=10)
    # visualize the residual
    #vis = mx.viz.plot_network(symbol=net, shape={"data" : (128, 1, 60, 60)})
    #vis.render('stn')
    
    # train
    train_model.fit(args, net, get_iterator(data_shape))
    
    
    