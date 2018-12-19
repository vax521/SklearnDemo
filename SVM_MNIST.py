# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 14:17:35 2018

@author: xingxf03
"""

import numpy as np
from sklearn import datasets
from sklearn.metrics import classification_report
import matplotlib
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics

mnist = datasets.fetch_mldata('MNIST original')
data, target = mnist.data, mnist.target
print('data.shape:{},target.shape:{}'.format(data.shape, target.shape))
index = np.random.choice(len(target), 70000, replace=False)


#获取特定大小的数据集
def mk_dataset(size):
    train_img = [data[i] for i in index[:size]]
    train_img = np.array(train_img)
    train_target = [target[i] for i in index[:size]]
    train_target = np.array(train_target)
    return train_img,train_target


def show_random_image():
    random_index = np.random.choice(len(target), 1, replace=False)
    digit = data[random_index].reshape(28,28)
    print(target[random_index])
    plt.imshow(digit,cmap=matplotlib.cm.binary,interpolation="nearest")
    plt.show()


# showRandomImage()
train_x, train_y, test_x, test_y = data[index[:60000]], target[index[:60000]], data[index[60000:]], target[index[60000:]]
model = SVC(kernel='rbf', probability=True)
model.fit(train_x, train_y)
predict = model.predict(test_x)
print(classification_report(test_y, predict))

precision = metrics.precision_score(test_y, predict)
recall = metrics.recall_score(test_y, predict)
print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))

accuracy = metrics.accuracy_score(test_y, predict)
print('accuracy: %.2f%%' % (100 * accuracy))

