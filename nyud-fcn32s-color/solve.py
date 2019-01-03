# coding: utf-8
import caffe
import surgery, score

import numpy as np
import os
import sys

try:
    import setproctitle

    setproctitle.setproctitle(os.path.basename(os.getcwd()))
# 获得当前路径(返回最后的文件名)
# 比如os.getcwd（）获得的当前路径为/home/zhangrf/fcn,则os.path.basename()为fcn；
# setproctitle是用来修改进程入口名称，如C++中入口为main()函数
except:
    pass

# weights = '../ilsvrc-nets/vgg16-fcn.caffemodel'
vgg_weights = '../ilsvrc-nets/VGG_ILSVRC_16_layers.caffemodel'  # 用来fine-tune的FCN参数
vgg_proto = '../ilsvrc-nets/VGG_ILSVRC_16_layers_deploy.prototxt'  # VGGNet模型

# init
# caffe.set_device(int(sys.argv[1]))
# 获取命令行参数，其中sys.argv[0]为文件名，argv[1]为紧随其后的那个参数
caffe.set_device(1)  # GPU型号id,这里指定第一块GPU
caffe.set_mode_gpu()

# solver = caffe.SGDSolver('solver.prototxt')
# solver.net.copy_from(weights)  # 这个方法仅仅是从vgg-16模型中拷贝参数，但是并没有改造原先的网络，这才是不收敛的根源
solver = caffe.SGDSolver('solver.prototxt')  # 调用SGD（随即梯度下降）Solver方法，solver.prototxt为所需参数
vgg_net = caffe.Net(vgg_proto, vgg_weights, caffe.TRAIN)  # vgg_net是原来的VGGNet模型（包括训练好的参数）
surgery.transplant(solver.net, vgg_net)  # FCN模型（参数）与原来的VGGNet模型之间的转化
del vgg_net  # 删除VGGNet模型

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]  # interp_layers为upscore层
surgery.interp(solver.net, interp_layers)  # 将upscore层中每层的权重初始化为双线性内核插值。

# scoring
test = np.loadtxt('../data/nyud/test.txt', dtype=str)  # 载入测试图片信息

for _ in range(50):
    solver.step(2000)  # 每2000次训练迭代执行后面的函数
    score.seg_tests(solver, False, test, layer='score')  # 测试图片
