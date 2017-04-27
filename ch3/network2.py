#!python35
# -*- coding: utf-8 -*-
### Libraries
# standard library
import random
import json
import sys
# Third-party libraries
import numpy as np
###二次代价函数和交叉熵代价函数
###计算差量、误差
class QuadraticCost(object):
    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2
    @staticmethod
    def delta(z, a, y):
        return (a - y) * sigmoid_prime(z)

class CrossEntropycost(object):
    """docstring for CrossEntropycost"""
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a) - (l-y)* np.log(1-a)))

    @staticmethod
    def  delta(z,a,y):
        return (a-y)
### Main Network class
class Network(object):
    def __init__(self,sizes,cost = CrossEntropycost):
        #神经网络层数
        self.num_layers = len(sizes)
        #每一层神经单元个数
        self.sizes = sizes
        #默认初始化权重方法
        self.default_weight_initializer()
        #代价函数(对象表示)
        self.cost = cost
    ###初始化函数
    #标准差 1/size^0.5
    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x ,y in zip(self.sizes[:-1],self.sizes[1:])]
    #标准差 1
    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x ,y in zip(self.sizes[:-1],self.sizes[1:])]

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
    #lmbda = 0 正则化
    #
    def SGD(self, training_data, epochs,
     mini_batch_size, eta,
     lmbda = 0.0,
     evaluation_data = None,
     monitor_evaluation_cost = False,
     monitor_evaluation_accuracy = False,
     monitor_training_cost = False,
     monitor_training_accuracy = False
     ):
        if evaluation_data:
            n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost , evaluation_accuracy = [],[]
        training_cost , training_accuracy = [],[]


        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches =[
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
                ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda,len(training_data))
            print("Epoch %s training complete" % j)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(
                    accuracy, n))

            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(
                    self.accuracy(evaluation_data), n_data))
            print("")
        return evaluation_cost, evaluation_accuracy, \
                training_cost, training_accuracy
    ###随机梯度下降
    #self 神经网络参数
    #training_data 训练数据 m*x
    #epochs 迭代次数 1500
    #mini_batch_size 小样本数据集 m
    #eta 学习率
    #lmbda = 0 正则化
    #
    def SGD_kaggle(self, training_data, epochs,
     mini_batch_size, eta,
     lmbda = 0.0,
     test_data = None
     ):
        if test_data:
            n_data = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches =[
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
                ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda,len(training_data))
        labels = [np.argmax(self.feedforward(x)) for x in test_data]
        return labels
    ###利用随机样本，调整神经网络参数
    ###参照算法p2.6-小样本集合
    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        #初始化
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            #得到单样本 w\b误差
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        #L2规范化
        self.weights = [w-(eta/len(mini_batch))*nw-eta*(lmbda/n)*w
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]
    ###反向传播函数
    ###参照算法p2.6-单样本
    def backprop(self, x, y):
        #初始化
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        #前向传播求 z、a
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # 输出层反向传播 BP1
        delta = (self.cost).delta(zs[-1],activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        #中间层反向传播 BP2-BP3-BP4
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    #评估函数
    def accuracy(self, data, convert=False):
        if convert:
            # when the type of y is vector
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)
    #代价函数
    #convert = false when data set is training data
    #convert = true when data set is validation or test data
    def total_cost(self, data, lmbda, convert=False):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost
    def save(self, filename):
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net
#### 辅助函数
def vectorized_result(j):

    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
def sigmoid_prime(z):
    #sigmoid函数导数
    return sigmoid(z)*(1-sigmoid(z))#IndentationError: expected an indented block

#S型神经元激活函数
def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

