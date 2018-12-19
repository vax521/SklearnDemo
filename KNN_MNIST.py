# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 14:17:35 2018

@author: xingxf03
"""

import numpy as np
from sklearn import datasets,model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

mnist = datasets.fetch_mldata('MNIST original')
data,target = mnist.data,mnist.target
print('data.shape:{},target.shape:{}'.format(data.shape,target.shape))


index = np.random.choice(len(target), 70000, replace=False)
#获取特定大小的数据集
def mk_dataset(size):
    train_img = [data[i] for i in index[:size]]
    train_img = np.array(train_img)
    train_target = [target[i] for i in index[:size]]
cond a    train_target = np.array(train_target)
    return train_img,train_target