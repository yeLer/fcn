#!/usr/bin/env python
# encoding: utf-8
'''
@author: lele Ye
@contact: 1750112338@qq.com
@software: pycharm 2018.2
@file: infer.py
@time: 2019/1/2 15:59
@desc:  这个文件用于部署性测试，输入单张图片得到其分割的结果
'''
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io

CAFFE_ROOT = "/home/bxx-yll/caffe"
import sys

sys.path.insert(0, CAFFE_ROOT + '/python')
import caffe

# the demo image is "2007_000129" from PASCAL VOC

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
im = Image.open('../demo/image.jpg')
in_ = np.array(im, dtype=np.float32)
in_ = in_[:, :, ::-1]
in_ -= np.array((104.00698793, 116.66876762, 122.67891434))
in_ = in_.transpose((2, 0, 1))

# 加载网络文件，与模型文件，并设置为测试模式
net = caffe.Net('./deploy.prototxt', './snapshot/train_iter_100000.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_

# run net and take argmax for prediction
net.forward()
out = net.blobs['score'].data[0].argmax(axis=0)
scipy.io.savemat('./out.mat', {'X': out})

# visualize segmentation in PASCAL VOC colors
plt.imshow(out)
plt.show()
plt.axis('off')
plt.savefig('../demo/testout_32s.png')
