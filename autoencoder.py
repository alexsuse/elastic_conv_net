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
import matplotlib.pyplot as plt
import cPickle as pic


class dA:
	"""
	The classic autoencoder of yore.
	hidden activation is given by h = logistic(W_vtoh*visible+b_h)
	visible activation is givens by z = logistic(W_htov*hidden+b_v)
	"""
	def __init__(self,nvisible,nhidden,data = None, W1 = None, bv= None, bh = None, rng_seed = None, regL = None):
		self.nhidden = nhidden
		self.nvisible = nvisible

		#hidden to visible matrix
		if W1 == None:
			Wi = np.asarray(np.random.uniform(
				low = -4*np.sqrt(6./(nhidden+nvisible) ),
				high = 4*np.sqrt(6./(nhidden+nvisible) ),
				size = (nvisible,nhidden)),dtype = theano.config.floatX)
			W = shared(value = Wi, name = 'W')
		else:
			W = shared(value = W1, name = 'W')
		self.W = W
		self.Wprime = W.T

		#biases
		if bh==None:
			bi_h = np.asarray(np.zeros(nhidden),dtype = theano.config.floatX)
			b_h = shared(value = bi_h , name = 'b_h')
		else:
			b_h = shared(value = bh , name = 'b_h')
		if bv==None:
			bi_v = np.asarray(np.zeros(nvisible),dtype = theano.config.floatX)
			b_v = shared(value = bi_v , name= 'b_v')
		else:
			b_v = shared(value = bv , name = 'b_v')
		self.b_v = b_v
		self.b_h = b_h
		
		#regularization parameter lambda
		
		if regL==None:
			self.lamb = None
		else:
			self.lamb = shared(value = regL, name = 'lamb')
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

		self.params=[self.W,self.b_h,self.b_v]

	def get_reconstruction_function(self,input):
		return T.nnet.sigmoid(T.dot(T.nnet.sigmoid(T.dot(input,self.W)+self.b_h),self.Wprime)+self.b_v)

	def corrupt_input(self,input,corruption_level):
		return self.theano_rng.binomial(size=input.shape, n=1, p=1 - corruption_level) * input

	def get_cost_and_updates(self,corruptionlevel,learning_rate):

		reconst_x = self.get_reconstruction_function(self.corrupt_input(self.data,corruptionlevel))
		L = -T.sum(self.data*T.log(reconst_x)+(1-self.data)*T.log(1-reconst_x),axis=1)
		cost = T.mean(L)
		if self.lamb!=None:
			L += self.lamb*(T.mean(T.dot(self.W,self.W)))
		
		gparams = T.grad(cost,self.params)

		updates=[]
		for param,gparams in zip(self.params,gparams):
			updates.append((param,param-learning_rate*gparams))
	
		return (cost,updates)

class assymetric_dA(dA):
	"""
	Assymetric AE, extend from dA
	hidden activation is given by h = logistic(W_vtoh*visible+b_h)
	visible activation is givens by z = logistic(W_htov*hidden+b_v)
	"""
	def __init__(self,nvisible,nhidden,data = None, W1 = None, W2 = None, b_v = None, b_h = None, rng_seed = None, regL = None):
		dA.__init__(self,nvisible,nhidden,data,W1,b_v,b_h,rng_seed,regL)

		#visible to hidden matrix
		Wi_htov = np.asarray(np.random.uniform(
			low = -4*np.sqrt(6./(nhidden+nvisible) ),
			high = 4*np.sqrt(6./(nhidden+nvisible) ),
			size = (nhidden,nvisible)),dtype = theano.config.floatX)
		Wprime = shared(value = Wi_htov, name = 'Wprime')
		self.Wprime = Wprime
		
		self.params.append(self.Wprime)
	
	def get_cost_and_updates(self,corruptionlevel,learning_rate):

		reconst_x = self.get_reconstruction_function(self.corrupt_input(self.data,corruptionlevel))
		L = -T.sum(self.data*T.log(reconst_x)+(1-self.data)*T.log(1-reconst_x),axis=1)
		cost = T.mean(L)
		if self.lamb!=None:
			L += self.lamb*(T.mean(T.dot(self.W,self.W))+T.mean(T.dot(self.Wprime,self.Wprime)))
		
		gparams = T.grad(cost,self.params)

		updates=[]
		for param,gparams in zip(self.params,gparams):
			updates.append((param,param-learning_rate*gparams))
	
		return (cost,updates)

if __name__=='__main__':
	#Mnist has 70000 examples, we use 50000 for training, set 20000 aside for validation
	train_size = 50000
	train_data,validation_data = ld.load_data_mnist(train_size = train_size)

	#fiddle around, not sure which values to use
	training_epochs = 100
	training_batches=100
	batch_size = int(train_data['images'].shape[0]/training_batches)
	batches,ys = ld.make_vector_batches(train_data,training_batches,batch_size)
	validation_images,validation_ys = ld.make_vector_batches(validation_data,1,validation_data['images'].shape[0])

	index = T.lscalar()
	x = T.dmatrix('x')
	
	#Creates a denoising autoencoder with 500 hidden nodes, could be changed as well
	a = dA(784,500,data=x,regL=0.01)
	
	#sEt theano shared variables for the train and validation data
	data_x = theano.shared(value = np.asarray(batches,dtype=theano.config.floatX),name='data_x')
	validation_x = theano.shared(value = np.asarray(validation_images[0,:,:],dtype=theano.config.floatX),name='validation_x')

	#get cost and update functions for the autoencoder
	cost,updates = a.get_cost_and_updates(0.2,0.01)

	#train_da returns the current cost and updates the dA parameters, index gives the batch index.
	train_da = theano.function([index],cost,updates=updates,givens=[(x , data_x[index])],on_unused_input='ignore')

	#validation_error just returns the cost on the validation set
	validation_error = theano.function([],cost,givens=[(x,validation_x)],on_unused_input='ignore')

	#loop over training epochs
	for epoch in xrange(training_epochs):
		c = []
		ve = validation_error()	
		#loop over batches
		for batch in xrange(training_batches):
			
			#collect costs for this batch
			c.append(train_da(batch))

		#pritn mean training cost in this epoch and final validation cost for checking
		print 'Training epoch %d, cost %lf, validation cost %lf' % (epoch, np.mean(c), ve)
	try:
		finame = raw_input('Output to pickle?')
	except SyntaxError, NameError:
		finame = 'output_pickle'
	fi = open(finame,'w')
	b = [a.W.get_value(),a.b_h.get_value(),a.b_v.get_value()]
	pic.dump(b,fi)
