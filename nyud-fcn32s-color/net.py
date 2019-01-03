#!/usr/bin/env python
# encoding: utf-8
'''
@author: lele Ye
@contact: 1750112338@qq.com
@software: pycharm 2018.2
@file: net.py
@time: 2019/1/2 15:59
@desc: 网络结构的定义文件，目的是用于生成trainval.prototxt和test.prototxt文件
'''
CAFFE_ROOT = "/home/bxx-yll/caffe"
import sys
sys.path.insert(0, CAFFE_ROOT + '/python')
import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop


def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad,
                         param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    return conv, L.ReLU(conv, in_place=True)


def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


def fcn(split, tops):
    n = caffe.NetSpec()
    n.data, n.label = L.Python(module='nyud_layers',
                               layer='NYUDSegDataLayer', ntop=2,
                               param_str=str(dict(nyud_dir='../data/nyud', split=split,
                                                  tops=tops, seed=1337)))

    # the base net
    # 从conv1_1到pool5照搬了vggnet的网络结构
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64, pad=100)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64)
    n.pool1 = max_pool(n.relu1_2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128)
    n.pool2 = max_pool(n.relu2_2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256)
    n.pool3 = max_pool(n.relu3_3)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512)
    n.pool4 = max_pool(n.relu4_3)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512)
    n.pool5 = max_pool(n.relu5_3)

    # fully conv
    # fc6与fc7将vgg的全连接层改成了1×1的卷积层.(各自还加了一个Dropout层).fc7的输出score_fr作为分类score.
    n.fc6, n.relu6 = conv_relu(n.pool5, 4096, ks=7, pad=0)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
    n.fc7, n.relu7 = conv_relu(n.drop6, 4096, ks=1, pad=0)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)

    n.score_fr = L.Convolution(n.drop7, num_output=40, kernel_size=1, pad=0,
                               param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])

    # 32倍的FCN直接将score_fr通过ConvolutionTranspose, 或人们习惯的Deconvolution将其放大32倍, 然后crop到原图大小.
    n.upscore = L.Deconvolution(n.score_fr, convolution_param=dict(num_output=40, kernel_size=64, stride=32,
                                                                   bias_term=False), param=[dict(lr_mult=0)])
    n.score = crop(n.upscore, n.data)

    n.loss = L.SoftmaxWithLoss(n.score, n.label,
                               loss_param=dict(normalize=False, ignore_label=255))

    return n.to_proto()


# 经过对比发现，trainval.prototxt与test.prototxt就是只有在第一层Python层python_param.param_str.split参数不一样
def make_net():
    tops = ['color', 'label']
    with open('trainval.prototxt', 'w') as f:
        f.write(str(fcn('trainval', tops)))

    with open('test.prototxt', 'w') as f:
        f.write(str(fcn('test', tops)))


if __name__ == '__main__':
    make_net()
