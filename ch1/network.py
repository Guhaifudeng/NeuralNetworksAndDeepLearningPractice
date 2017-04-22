#!python35
# -*- coding: utf-8 -*-
### Libraries
# standard library
import random
# Third-party libraries
import numpy as np

class Network(object):
    def __init__(self,sizes):
        #神经网络层数
        self.num_layers = len(sizes)
        #每一层神经单元个数
        self.sizes = sizes
        #针对每一个层神经元偏置
        self.biases = [np.random.randn(y,1) for y in sizes[1:]] # y-1,1..2
        #两层神经元权重
        self.weights = [np.random.randn(y,x) for x ,y in zip(sizes[:-1], sizes[1:])]# y-x,1..2

    ###前向传播
    #a 输入层数据 x-1 针对于实际数据 m-x
    def feedforward(self,a):
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        #返回输出层神经元
        return a


    ###随机梯度下降
    #self 神经网络参数
    #training_data 训练数据 m*x
    #epochs 迭代次数 1500
    #mini_batch_size 小样本数据集 m
    #eta 学习率
    #test_data 测试数据
    def SGD(self, training_data, epochs,
     mini_batch_size, eta, test_data = None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches =[
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
                ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch{0}:{1} /{2}".format(j, self.evaluate(test_data),n_test))# 大错误,符号搞错{}中文输入
            else:
                print("Epoch{0}complete".format(j))
    ###随机梯度下降
    #self 神经网络参数
    #training_data 训练数据 m*x
    #epochs 迭代次数 1500
    #mini_batch_size 小样本数据集 m
    #eta 学习率
    #test_data_x 测试数据:向量
    def SGD_kaggle(self, training_data, epochs,
     mini_batch_size, eta, test_data = None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches =[
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
                ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
        test_results =[np.argmax(self.feedforward(x)) for x in test_data]
        return test_results
    #评估函数
    def evaluate(self, test_data):
    #返回测试数据预测正确个数
        test_results = [(np.argmax(self.feedforward(x)),y)
                    for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    ###未完成
    ###利用随机样本，调整神经网络参数
    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
    ###反向传播函数
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))#IndentationError: expected an indented block

#S型神经元激活函数
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

