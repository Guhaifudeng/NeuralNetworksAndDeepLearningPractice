#!python35
# _*_ coding:utf-8 _*_
# Standard library
import pickle
import gzip

#Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
#the 'downsample' module has been moved to pool
from theano.tensor.signal import pool

#activation functions  for neurons
def linear(z):return z
def ReLu(z):return T.maxinum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh
#### 常数
GPU = False
if GPU:
    print("Trying to run under a GPU.  If this is not desired, then modify "+\
        "network3.py\nto set the GPU flag to False.")
    try: theano.config.device = 'gpu'
    except: pass # it's already set
    theano.config.floatX = 'float32'
else:
    print("Running with a CPU.  If this is not desired, then the modify "+\
        "network3.py to set\nthe GPU flag to True.")

### Load the mnist data
def load_data_shared(filename = 'mnist.pkl.gz'):
    f = gzip.open(filename, 'rb')
    training_data, validation_data , test_data = pickle.load(f, encoding = 'latin1')
    f.close()

    print(type(training_data[0]),np.shape(training_data[0]), np.shape(validation_data[0]), np.shape(test_data[0]))
    print(np.shape(training_data[1]), np.shape(validation_data[1]), np.shape(test_data[1]))
    def  shared(data):
        shared_x = theano.shared(
            np.asarray(data[0], dtype = theano.config.floatX),borrow = True
            )
        shared_y = theano.shared(
            np.asarray(data[1], dtype = theano.config.floatX),borrow = True
            )
        return shared_x, T.cast(shared_y,'int32')#
    return [shared(training_data), shared(validation_data), shared(test_data)]

####Main class used to construct and train networks

class Network(object):
    """docstring for ClassName"""
    def __init__(self, layers , mini_batch_size):
        self.layers = layers
        self.mini_batch_size = mini_batch_size
        self.params = [param for layer in self.layers for param in layer.params]
        self.x = T.matrix('x')
        self.y = T.ivector('y') # 写错，wrong
        init_layer = self.layers[0]
        init_layer.set_inpt(self.x, self.x,self.mini_batch_size)

        for j in range(1, len(self.layers)):
            prev_layer, layer = self.layers[j-1], self.layers[j]
            layer.set_inpt(
                prev_layer.output, prev_layer.output_dropout,self.mini_batch_size)
        self.output = self.layers[-1].output
        self.output_dropout = self.layers[-1].output_dropout
    def SGD(self, training_data, epochs, mini_batch_size,eta,
            validation_data, test_data, lmbda = 0.0):
        training_x, training_y = training_data
        validation_x , validation_y = validation_data
        test_x ,test_y= test_data

        #计算训练、验证、测试数据的小样本个数
        num_training_batches = int(size(training_data)/mini_batch_size)
        num_validation_batches = int(size(validation_data)/mini_batch_size)
        num_test_batches = int(size(test_data)/mini_batch_size)

        #定义L2规范、代价函数、梯度、参数更新
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self) + 0.5*lmbda*l2_norm_squared/num_training_batches
        grads = T.grad(cost, self.params)
        updates = [(param,param-eta*grad)
                    for param, grad in zip(self.params, grads)]

        ###训练，计算精确率
        #小样本
        i = T.lscalar()# mini-batch index
        #训练集：小样本数据、代价、参数更新
        train_mb = theano.function(
            [i], cost, updates = updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        #验证集：小样本数据、精确率
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={

                self.x:
                validation_x[i*self.mini_batch_size: (i+1)* self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)* self.mini_batch_size]
            })
        #测试集：小样本数据、精确率
        test_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        #测试集：小样本数据、输出
        self.test_mb_predictions = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
         # 训练
        best_validation_accuracy = 0.0
        for epoch in range(epochs):
            iteration = 0
            for minibatch_index in range(num_training_batches):
                iteration = num_training_batches*epoch+minibatch_index
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                cost_ij = train_mb(minibatch_index)

            if (iteration+1) % num_training_batches == 0:
                validation_accuracy = np.mean(
                    [validate_mb_accuracy(j) for j in range(num_validation_batches)])
                print("Epoch {0}: validation accuracy {1:.2%}".format(
                    epoch, validation_accuracy))
                if validation_accuracy >= best_validation_accuracy:
                    print("This is the best validation accuracy to date.")
                    best_validation_accuracy = validation_accuracy
                    best_iteration = iteration
                    if test_data:
                        test_accuracy = np.mean(
                            [test_mb_accuracy(j) for j in range(num_test_batches)])
                        print('The corresponding test accuracy is {0:.2%}'.format(
                            test_accuracy))

    def SGD_kaggle(self, training_data, epochs, mini_batch_size,eta,
            validation_data, test_data, lmbda = 0.0):
        training_x, training_y = training_data
        validation_x , validation_y = validation_data
        test_x = test_data

        #计算训练、验证、测试数据的小样本个数
        num_training_batches = int(size(training_data)/mini_batch_size)
        num_validation_batches = int(size(validation_data)/mini_batch_size)
        num_test_batches = int((test_data.get_value().shape[0])/mini_batch_size)
        #定义L2规范、代价函数、梯度、参数更新
        l2_norm_squared = sum([(layer.w**2).sum() for layer in self.layers])
        cost = self.layers[-1].cost(self) + 0.5*lmbda*l2_norm_squared/num_training_batches
        grads = T.grad(cost, self.params)
        updates = [(param,param-eta*grad)
                    for param, grad in zip(self.params, grads)]
                ###训练，计算精确率
        #小样本
        i = T.lscalar()# mini-batch index
        #训练集：小样本数据、代价、参数更新
        train_mb = theano.function(
            [i], cost, updates = updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            })
        #验证集：小样本数据、精确率
        validate_mb_accuracy = theano.function(
            [i], self.layers[-1].accuracy(self.y),
            givens={

                self.x:
                validation_x[i*self.mini_batch_size: (i+1)* self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)* self.mini_batch_size]
            })
        model_predict = theano.function(
            [i], self.layers[-1].y_out,
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)* self.mini_batch_size]
            })


         # 训练
        best_validation_accuracy = 0.0
        for epoch in range(epochs):
            iteration = 0
            for minibatch_index in range(num_training_batches):
                iteration = num_training_batches*epoch+minibatch_index
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                cost_ij = train_mb(minibatch_index)

            if (iteration+1) % num_training_batches == 0:
                validation_accuracy = np.mean(
                    [validate_mb_accuracy(j) for j in range(num_validation_batches)])
                print("Epoch {0}: validation accuracy {1:.2%}".format(
                    epoch, validation_accuracy))
                if validation_accuracy >= best_validation_accuracy:
                    print("This is the best validation accuracy to date.")
                    best_validation_accuracy = validation_accuracy
                    best_iteration = iteration
        digit_preds = np.concatenate([model_predict(i) for i in range(num_test_batches)])
        return digit_preds
####定义隐含层
#卷积层：卷积+池化
class ConvPoolLayer(object):


    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
                 activation_fn=sigmoid):

        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn=activation_fn
        # initialize weights and biases
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])/np.prod(poolsize))
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
                dtype=theano.config.floatX),
            borrow=True)
        self.b = theano.shared(
            np.asarray(
                np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
                dtype=theano.config.floatX),
            borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape(self.image_shape)
        conv_out = conv.conv2d(
            input=self.inpt, filters=self.w, filter_shape=self.filter_shape,
            image_shape=self.image_shape)
        pooled_out = pool.pool_2d(
            input=conv_out, ws=self.poolsize, ignore_border=True)
        self.output = self.activation_fn(
            pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_dropout = self.output # no dropout in the convolutional layers
#全连接
class FullyConnectedLayer(object):

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.asarray(
                np.random.normal(
                    loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
                       dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn(
            (1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = self.activation_fn(
            T.dot(self.inpt_dropout, self.w) + self.b)

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))
#输出层
class SoftmaxLayer(object):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = theano.shared(
            np.zeros((n_in, n_out), dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.zeros((n_out,), dtype=theano.config.floatX),
            name='b', borrow=True)
        self.params = [self.w, self.b]

    def set_inpt(self, inpt, inpt_dropout, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = softmax((1-self.p_dropout)*T.dot(self.inpt, self.w) + self.b)
        self.y_out = T.argmax(self.output, axis=1)
        self.inpt_dropout = dropout_layer(
            inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout)
        self.output_dropout = softmax(T.dot(self.inpt_dropout, self.w) + self.b)

    def cost(self, net):
        "Return the log-likelihood cost."
        return -T.mean(T.log(self.output_dropout)[T.arange(net.y.shape[0]), net.y])

    def accuracy(self, y):
        "Return the accuracy for the mini-batch."
        return T.mean(T.eq(y, self.y_out))


#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].get_value(borrow=True).shape[0]

def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(
        np.random.RandomState(0).randint(999999))
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer*T.cast(mask, theano.config.floatX)
