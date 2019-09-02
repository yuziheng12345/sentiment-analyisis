# -*- coding:utf-8 -*-
class Parameters(object):

    #训练数据和测试数据路径
    train_posdata = './data/wx-train/pos_data.txt'
    train_negdata = './data/wx-train/neg_data.txt'
    test_data = './data/test_wx.txt'
    #各词典路径
    stop_word = './data/停用词.txt'
    deny_word = './data/Worddict/否定词'
    posdict = './data/Worddict/positive'
    negdict = './data/Worddict/negative'
    degree_word = './data/Worddict/程度级别词语'
    link_word = './data/Worddict/转折连词'
    dj_word = './data/Worddict/递进连词'
    #神经网络参数
    batch_size = 100
    epochs = 20
    loss = 'binary_crossentropy'
    optimizer = 'rmsprop'
    hidden_dim = 128
    filters = 100
    kernel_sizes = (3,4,5)
    strides = 1
    dropout = 0.5
