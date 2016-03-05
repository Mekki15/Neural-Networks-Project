# Neural-Networks-Project

import theano
from theano import tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
%matplotlib inline
import numpy as np
import pylab 
from PIL import Image
import os
import matplotlib.pyplot as plt
import glob
import matplotlib.image as mpimg
from sklearn.cross_validation import train_test_split

class LeNetConvPoolLayer:
    def _init_(self,rng,input,filter_shape,image_shape,poolsize:
        assert image_shape[1]==filter_shape[1]
        self.input=input
        fan_in=numpy.prod[filter_shape[1:]]
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        
        conv_out=conv2d(input=input,filters=self.W,filter_shape=filter_shape,image_shape=image_shape)
        
        pooled_out=downsample.maxpool_2d(input=onv_out,ds=poolsize,ignore_border=True)
        
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        
        self.params = [self.W, self.b]
        
        self.input = input

impath1=glob.glob('/home/mekki/upper/img*')
x= np.array([np.array(Image.open(impath1[i])) for i in range(len(impath1))])
print(imarray1.shape)
y=np.zeros(962)

X_train, X_test=train_test_split(x,test_size=0.25)
print(X_train.shape)
Y_train, Y_test=train_test_split(y,test_size=0.25)
print(Y_train.shape)

x=T.tensor3('X_train')
y=T.ivector('y')
rng=np.random.RandomState(23455)

layer0_input = x.reshape((721, 3,480,640))

layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(721, 3, 480, 640),
        filter_shape=(20, 3, 12, 12),
        poolsize=(2,2)
    )

layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(721,20, 12, 12),
        filter_shape=(50,20, 5, 5),
        poolsize=(2,2)
    )
    
    
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

   // train_model = theano.function(
     //   [index],
      //  cost,
      //  updates=updates,
      //  givens={
      //      x: train_set_x[index * batch_size: (index + 1) * batch_size],
      //      y: train_set_y[index * batch_size: (index + 1) * batch_size]
      //  }
    //)
    
