#!/usr/bin/env python
# encoding: utf-8
'''
@author: lele Ye
@contact: 1750112338@qq.com
@software: pycharm 2018.2
@file: net.py.py
@time: 2019/1/2 17:21
@desc:nyud-fcn16s-color 网络结构的定义文件，目的是用于生成trainval.prototxt和test.prototxt文件
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


# n=caffe.NetSpec() 是获取Caffe的一个Net，我们只需不断的填充这个n，最后面把n输出到文件就会使我们在Caffe学习里面看到的Net的protobuf的定义
def fcn(split, tops):
    n = caffe.NetSpec()
    n.data, n.label = L.Python(module="nyud_layers", layer="NYUDSegDataLayer", ntop=2,
                               param_str=str(dict(nyud_dir='../data/nyud', split=split,
                                                  tops=tops, seed=1337)))
    # 基本的网络结构
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

    # 全连接层
    n.fc6, n.relu6 = conv_relu(n.pool5, 4096, ks=7, pad=0)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
    n.fc7, n.relu7 = conv_relu(n.drop6, 4096, ks=1, pad=0)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)

    # 将第七层的dropout层进行1x1卷积，输出40个类别
    n.score_fr = L.Convolution(n.drop7, num_output=40, kernel_size=1, pad=0,
                               param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    '''
    第一次合并
    '''

    # 将第七层卷积放大两倍得到upscore2
    n.upscore2 = L.Deconvolution(n.score_fr,
                                 convolution_param=dict(num_output=40, kernel_size=4, stride=2,
                                                        bias_term=False),
                                 param=[dict(lr_mult=0)])
    # 在pool4上加一个1×1的conv层, 得到score_pool4
    n.score_pool4 = L.Convolution(n.pool4, num_output=40, kernel_size=1, pad=0,
                                  param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    # 将score_pool4 crop到与upscore2一样的形状, 得到score_pool4c.crop的对象是score_pool4, 因为它可能比upscore2要大.
    n.score_pool4c = crop(n.score_pool4, n.upscore2)
    # fuse_pool4 = score_pool4c + upscore2
    n.fuse_pool4 = L.Eltwise(n.upscore2, n.score_pool4c,
                             operation=P.Eltwise.SUM)
    '''
    第二次合并
    '''
    # 将第一次合并结果放大两倍得到upscore_pool4
    n.upscore_pool4 = L.Deconvolution(n.fuse_pool4,
                                      convolution_param=dict(num_output=40, kernel_size=4, stride=2,
                                                             bias_term=False),
                                      param=[dict(lr_mult=0)])

    # 在pool3上加一个1×1的conv层, 得到score_pool3
    n.score_pool3 = L.Convolution(n.pool3, num_output=40, kernel_size=1, pad=0,
                                   param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    # 将score_pool3 crop到与upscore4一样的形状
    n.score_pool3c = crop(n.score_pool3, n.upscore_pool4)
    # fuse_pool3 = upscore_pool4 + score_pool3c
    n.fuse_pool3 = L.Eltwise(n.upscore_pool4, n.score_pool3c,
                             operation=P.Eltwise.SUM)

    '''
    第三次后处理预测
    '''
    # 将fuse_pool3放大8倍, 然后crop, 得到与原图大小相同的score
    n.upscore8 = L.Deconvolution(n.fuse_pool3,
                                  convolution_param=dict(num_output=40, kernel_size=16, stride=8,
                                                         bias_term=False),
                                  param=[dict(lr_mult=0)])

    n.score = crop(n.upscore8, n.data)

    # ignore_label：int型变量，默认为空。
    # 如果指定值，则label等于ignore_label的样本将不参与Loss计算，并且反向传播时梯度直接置0
    n.loss = L.SoftmaxWithLoss(n.score, n.label,
                               loss_param=dict(normalize=False, ignore_label=255))

    return n.to_proto()


def make_net():
    tops = ['color', 'label']
    with open('trainval.prototxt', 'w') as f:
        f.write(str(fcn('trainval', tops)))

    with open('test.prototxt', 'w') as f:
        f.write(str(fcn('test', tops)))


if __name__ == '__main__':
    make_net()
