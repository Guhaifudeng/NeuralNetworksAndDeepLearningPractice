#!python35
# -*- coding: utf-8 -*-
### Libraries
# standard library
"""
error:
Traceback (most recent call last):
   File "mnist.py", line 7, in <module>
     train_set, valid_set, test_set = pickle.load(f)
UnicodeDecodeError: 'ascii' codec can't decode byte 0x90 in position 614: ordinal not in range(128)
解决办法：
1.使用python27
2.自己构造mnist.pkl
3.http://stackoverflow.com/questions/11305790/pickle-incompatability-of-numpy-arrays-between-python-2-and-3
"""
import pickle
import gzip

#Third-party libraries
import numpy as np

#从本地加载数据 数据pickle 什么鬼？
def  load_data():
    #python3
    #with gzip.open('../data/mnist.pkl.gz', 'rb') as f:
    #    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    #f.close()
    with gzip.open('../data/mnist.pkl.gz', 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        training_data, validation_data, test_data = u.load()
    f.close()
    return (training_data, validation_data, test_data)
#封装数据
def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    #print("training_data shape",np.shape(tr_d))

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    print(type(te_d[1]),np.shape(te_d[1]))
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)
#1-10 常数变向量
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
