#!python35
# -*- coding: utf-8 -*-
import mnist_loader
import network2
training_data ,validation_data, test_data = \
    mnist_loader.load_data_wrapper()
net = network2.Network([784,30,10])
net.SGD(training_data, 30,10,10.0,lmbda = 1000.0,
        evaluation_data = validation_data,monitor_evaluation_accuracy = True)
