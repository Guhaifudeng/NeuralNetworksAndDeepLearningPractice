
#!python35
# -*- coding: utf-8 -*-
import network3
from network3 import Network, ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer # softmax plus log-likelihood cost is more common in modern image classification networks.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
def digit_recognize_test():
    #载入数据
    train = pd.read_csv('../../input/train.csv').sample(20000)
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
    training_inputs =  [x for x in X]
    training_results = [y for y in Y]
    training_data = [training_inputs[:10000], training_results[:10000]]
    validation_data =[training_inputs[10000:], training_results[10000:]]
    print(np.shape(training_results))
    #测试数据
    test_data = test.values
    #predit_inputs = [np.reshape(x, (784, 1)) for x in test_data]

    training_data =shared(training_data)
    validation_data = shared(validation_data)
    test_data = theano.shared(
        np.asarray(test_data, dtype = theano.config.floatX),borrow = True
        )

    mini_batch_size = 10
    net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28),
                      filter_shape=(20, 1, 5, 5),
                      poolsize=(2, 2)),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12),
                      filter_shape=(40, 20, 5, 5),
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=40*4*4, n_out=100),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
    predictions = net.SGD_kaggle(training_data, 20, mini_batch_size, 0.1, validation_data, test_data)
    ####尚未解决


    #输出数据
    digit_preds = pd.Series(predictions)
    image_ids = pd.Series(np.arange(1, len(digit_preds) + 1))
    submission = pd.DataFrame([image_ids, digit_preds]).T
    submission.columns = ['ImageId', 'Label']
    submission.to_csv('../../output/dr_result.csv', index=False ,header=True)
    print("finished!")
def  shared(data):
    shared_x = theano.shared(
        np.asarray(data[0], dtype = theano.config.floatX),borrow = True
        )
    shared_y = theano.shared(
        np.asarray(data[1], dtype = theano.config.floatX),borrow = True
        )
    return shared_x, T.cast(shared_y,'int32')#

digit_recognize_test()
