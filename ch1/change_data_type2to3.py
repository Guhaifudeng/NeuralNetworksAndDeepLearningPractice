# standard library
import cPickle
import gzip

#Third-party libraries
# -*- coding: utf-8 -*-
import numpy as np
import cPickle
def  load_data():
    f = open('../data/mnist.pkl','rb')
    training_data , validation_data, test_data = cPickle.load(f) #python3

    f.close()

    output = open('data.pkl', 'wb')
    cPickle.dump((training_data , validation_data, test_data ),output,-1)
    output.close()
    return (training_data, validation_data, test_data)
load_data()
