#!/usr/bin/env python
# encoding: utf-8
'''
@author: lele Ye
@contact: 1750112338@qq.com
@software: pycharm 2018.2
@file: solve.py
@time: 2019/1/2 17:21
@desc:nyud-fcn16s-color 网络结构的定义文件，目的是用于生成trainval.prototxt和test.prototxt文件
'''
CAFFE_ROOT = "/home/bxx-yll/caffe"
import sys

sys.path.insert(0, CAFFE_ROOT + '/python')
import caffe
import surgery, score

import numpy as np
import os

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
# 获得当前路径(返回最后的文件名)
# 比如os.getcwd（）获得的当前路径为/home/bxx-yll/fcn,则os.path.basename()为fcn；
# setproctitle是用来修改进程入口名称，如C++中入口为main()函数
except:
    pass

# vgg_weights = '../ilsvrc-nets/VGG_ILSVRC_16_layers.caffemodel'  # 用来fine-tune的FCN参数
# vgg_proto = '../ilsvrc-nets/VGG_ILSVRC_16_layers_deploy.prototxt'  # VGGNet模型
# 这次我们用fcn16s的模型微调训练
weights = '../nyud-fcn32s-color/snapshot/train_iter_100000.caffemodel'

# init
# caffe.set_device(int(sys.argv[1]))
# 获取命令行参数，其中sys.argv[0]为文件名，argv[1]为紧随其后的那个参数
caffe.set_device(2)  # GPU型号id,这里指定第3块GPU
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')  # 调用SGD（随即梯度下降）Solver方法，solver.prototxt为所需参数
solver.net.copy_from(weights)  # 这个方法仅仅是从vgg-16模型中拷贝参数，但是并没有改造原先的网络，这才是不收敛的根源

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]  # interp_layers为upscore层
surgery.interp(solver.net, interp_layers)  # 将upscore层中每层的权重初始化为双线性内核插值。

# scoring
test = np.loadtxt('../data/nyud/test.txt', dtype=str)  # 载入测试图片信息

for _ in range(50):
    solver.step(2000)  # 每2000次训练迭代执行后面的函数
    score.seg_tests(solver, False, test, layer='score')  # 测试图片
