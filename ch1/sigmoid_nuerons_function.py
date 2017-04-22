#!python35
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1.0/(1+np.exp(-z))
def test_sigmoid():
    z = np.arange(-5, 5, .02)
    sigma = sigmoid(z)
    print(type(sigma))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(z, sigma)
    ax.set_ylim([-0.5,1.5])
    ax.set_xlim([-5,5])
    ax.grid(True)
    ax.set_xlabel('z')
    ax.set_title('step function')
    plt.show()
