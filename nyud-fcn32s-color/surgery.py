# coding: utf-8
from __future__ import division  # 导入python未来支持的语言特征division(精确除法)
import caffe
import numpy as np


def transplant(new_net, net, suffix=''):  # 用于将VGGNet的参数转化给FCN（包括全连接层的参数）
    """
    Transfer weights by copying matching parameters, coercing parameters of
    incompatible shape, and dropping unmatched parameters.
	
	通过复制匹配的参数，强制转换不兼容形状的参数和丢弃不匹配的参数来达到传输（转化）权重的目的；

    The coercion is useful to convert fully connected layers to their
    equivalent convolutional layers, since the weights are the same and only
    the shapes are different.  
	因为权重的个数是一样的仅仅是Blob的形状不一样，所以强制转换对于将全连接层转换为等效的卷积层是有用的；
	
	In particular, equivalent fully connected and
    convolution layers have shapes O x I and O x I x H x W respectively for O
    outputs channels, I input channels, H kernel height, and W kernel width.
	参数数量为O*I*H*W
    Both  `net` to `new_net` arguments must be instantiated `caffe.Net`s.
    """
    for p in net.params:  # net.params是字典形式，存放了所有的key-value，p为key
        p_new = p + suffix  # 将p赋给p_new
        if p_new not in new_net.params:  # 用来丢弃fc8（因为FCN中没有fc8）
            print 'dropping', p
            continue
        for i in range(len(net.params[p])):
            if i > (len(new_net.params[p_new]) - 1):
                print 'dropping', p, i
                break
            if net.params[p][i].data.shape != new_net.params[p_new][i].data.shape:
                # Blob不一样转换（这边就是全连接层和卷积层的转换，很精髓！！！）
                print 'coercing', p, i, 'from', net.params[p][i].data.shape, 'to', new_net.params[p_new][i].data.shape
            else:  # 形状一样则直接copy
                print 'copying', p, ' -> ', p_new, i
            new_net.params[p_new][i].data.flat = net.params[p][i].data.flat  # 将参数按顺序赋值（flat函数只要保证参数个数相同，不用保证数组形状完全一样）


def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]  # 生成一列向量和一行向量
    # （64*1）的列向量和（1*64）行向量相乘则得到一个64*64的数组
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)


def interp(net, layers):
    """
    Set weights of each layer in layers to bilinear kernels for interpolation.
	将每一层的权重设置为双线性内核插值。
    """
    for l in layers:
        try:
            m, k, h, w = net.params[l][0].data.shape
        except AssertionError as err:
            if m != k and k != 1:
                print 'input + output channels need to be the same or |output| == 1'
                raise err
            if h != w:
                print 'filters need to be square'
                raise err
        filt = upsample_filt(h)  # 初始化卷积核的参数（64*64*1）
        net.params[l][0].data[range(m), range(k), :,
        :] = filt  # 这边很关键！！！只有对于对应层的那层filter有参数，其余都为0，而且有filter参数的那层还都是相等的~
        # 因为前一层已经是个分类器了，对分类器进行特征组合没有任何意义！所以这一层的上采样效果上而言只是对应的上采样（属于猴子还是属于猴子）


def expand_score(new_net, new_layer, net, layer):  # 这个函数干啥用的没看懂- -貌似solve.py里没有这个函数的调用
    """
    Transplant an old score layer's parameters, with k < k' classes, into a new
    score layer with k classes s.t. the first k' are the old classes.
    """
    old_cl = net.params[layer][0].num
    new_net.params[new_layer][0].data[:old_cl][...] = net.params[layer][0].data
    new_net.params[new_layer][1].data[0, 0, 0, :old_cl][...] = net.params[layer][1].data
