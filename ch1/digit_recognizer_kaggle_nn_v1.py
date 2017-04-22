#!python27
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import network
def digit_recognize_test():
    #载入数据
    train = pd.read_csv('../../input/train.csv').sample(2000)
    print('train: ' + str(train.shape))
    test = pd.read_csv('../../input/test.csv')
    print('test: ' + str(test.shape))
    train.head()
    #train: (2000, 785)
    #test: (28000, 784)

    # feature matrix
    X = train.ix[:,1:]
    # response vector
    X = X.values
    Y = train['label']
    Y = Y.values
    #训练数据
    training_inputs = [np.reshape(x, (784, 1)) for x in X]
    training_results = [vectorized_result(y) for y in Y]
    training_data = list(zip(training_inputs, training_results))
    print(np.shape(training_results))
    #测试数据
    test_data = list(test.values)
    predit_inputs = [np.reshape(x, (784, 1)) for x in test_data]

    net = network.Network([784,30,10])
    predictions = net.SGD_kaggle(training_data, 70,10,3.0,test_data = predit_inputs)
    #输出数据
    result = pd.DataFrame({'ImageId': list(range(1,len(predictions)+1)), 'Label': predictions})
    result.to_csv('../../output/dr_result.csv', index=False, header=True)
    print("finished!")
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
digit_recognize_test()
