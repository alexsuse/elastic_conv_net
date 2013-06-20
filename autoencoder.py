#!/usr/bin/env python
"""
Implements a simple autoencoder class, with a simple training method.
"""
from theano import tensor as T
from theano import function, shared
from numpy import exp, dot, sum, log
import numpy as np
import theano
import load_data as ld

class dA:
	"""
	The classic autoencoder of yore.
	hidden activation is given by h = logistic(W_vtoh*visible+b_h)
	visible activation is givens by z = logistic(W_htov*hidden+b_v)
	"""
	def __init__(self,nvisible,nhidden,data = None, rng_seed = None):
		self.nhidden = nhidden
		self.nvisible = nvisible

		#hidden to visible matrix
		Wi_htov = np.asarray(np.random.uniform(
			low = -4*np.sqrt(6./(nhidden+nvisible) ),
			high = 4*np.sqrt(6./(nhidden+nvisible) ),
			size = (nhidden,nvisible)),dtype = theano.config.floatX)
		W_htov = shared(value=Wi_htov, name='W_htov')
		self.W_htov = W_htov

		#visible to hidden matrix
		Wi_vtoh = np.asarray(np.random.uniform(
			low = -4*np.sqrt(6./(nhidden+nvisible) ),
			high = 4*np.sqrt(6./(nhidden+nvisible) ),
			size = (nvisible,nhidden)),dtype = theano.config.floatX)
		W_vtoh = shared(value = Wi_vtoh, name = 'W_vtoh')
		self.W_vtoh = W_vtoh

		#biases
		bi_h = np.asarray(np.zeros(nhidden),dtype = theano.config.floatX)
		b_h = shared(value = bi_h,name='b_h')
		self.b_h = b_h
		bi_v = np.asarray(np.zeros(nvisible),dtype = theano.config.floatX)
		b_v = shared(value = bi_v , name= 'b_v')
		self.b_v = b_v
		if data:
			self.data = data
		else:
			self.data = T.dmatrix('data')
		if rng_seed:
			self.rng = np.random.RandomState(rng_seed)
			self.theano_rng = T.shared_randomstreams.RandomStreams(self.rng.randint(2**30))
		else:
			self.rng = np.random.RandomState(1234)
			self.theano_rng = T.shared_randomstreams.RandomStreams(self.rng.randint(2**30))

		self.params=[self.W_vtoh,self.W_htov,self.b_h,self.b_v]

	def get_reconstruction_function(self,input):
		return T.nnet.sigmoid(T.dot(T.nnet.sigmoid(T.dot(input,self.W_vtoh)+self.b_h),self.W_htov)+self.b_v)

	def corrupt_input(self,input,corruption_level):
		return self.theano_rng.binomial(size=input.shape, n=1, p=1 - corruption_level) * input

	def get_cost_and_updates(self,corruptionlevel,learning_rate):

		reconst_x = self.get_reconstruction_function(self.corrupt_input(self.data,corruptionlevel))
		L = -T.sum(self.data*T.log(reconst_x)+(1-self.data)*T.log(1-reconst_x),axis=1)
		cost = T.mean(L)

		gparams = T.grad(cost,self.params)

		updates=[]
		for param,gparams in zip(self.params,gparams):
			updates.append((param,param-learning_rate*gparams))
	
		return (cost,updates)


if __name__=='__main__':

	data,dataset_size = ld.load_data_mnist()

	training_epochs = 50
	training_batches=100
	batch_size = int(dataset_size/training_batches)-2
	
	batches,ys = ld.make_vector_batches(data,training_batches,batch_size)
#	theano_data = logistic_sgd.load_data('/Users/alex//Downloads/mnist.pkl.gz')
#	data_x,data_y = theano_data[0]
	index = T.lscalar()
	x = T.dmatrix('x')

	a = dA(784,500,data=x)
	data_x = theano.shared(value = np.asarray(batches,dtype=theano.config.floatX),name='data_x')

	cost,updates = a.get_cost_and_updates(0.2,0.01)
	train_da = theano.function([index],cost,updates=updates,givens=[(x , data_x[index])],on_unused_input='ignore')
	for epoch in xrange(training_epochs):
		c = []
		for batch in xrange(training_batches):
			#data_x.set_value(np.asarray(batches[batch,:,:],dtype=theano.config.floatX))
			c.append(train_da(batch))
		print 'Training epoch %d, cost ' % epoch, np.mean(c)
