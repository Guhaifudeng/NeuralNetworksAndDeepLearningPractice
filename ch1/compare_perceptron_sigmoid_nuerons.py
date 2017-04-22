#!python35
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sigmoid_nuerons_function as snf
# 引入snf时，也会运行其中的语句
import perceptron_function as pf
z = np.arange(-5, 5, .02)
sigma = snf.sigmoid(z)
step = pf.step_2(z)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(z, sigma)
ax.plot(z, step)
ax.set_ylim([-0.5,1.5])
ax.set_xlim([-5,5])
ax.grid(True)
ax.set_xlabel('z')
ax.set_title('sigmoid nuerons and perceptron')
ax.legend(['sigmoid nuerons','perceptron'])
plt.show()

