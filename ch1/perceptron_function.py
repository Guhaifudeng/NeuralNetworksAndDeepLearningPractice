#!python35
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def step(z):
#    if z >= np.zeros(np.shape(z)):
#       return np.zeros(np.shape(z))
#    else:
#        return np.ones(np.shape(z))
    step_fn = np.vectorize(lambda z: 1.0 if z >= 0.0 else 0.0)
    return step_fn(z)
def  sign(x):
    if x >= 0:
        return 1
    else:
        return 0
def  step_2(z):
     return [sign(x) for x in z]
def test_step():
    z = np.arange(-5, 5, .02)
    step_fn = np.vectorize(lambda z: 1.0 if z >= 0.0 else 0.0)
    print(type(step_fn))
    step = step_fn(z)
    print(type(step))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(z, step)
    ax.set_ylim([-0.5,1.5])
    ax.set_xlim([-5,5])
    ax.grid(True)
    ax.set_xlabel('z')
    ax.set_title('step function')
    plt.show()
