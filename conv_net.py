#!/usr/bin/env python
import numpy as np
import theano.tensor as T
from theano.tensor.nnet import conv
import theano
from autoencoder import dA
import load_data as ld
import cPickle as pic
import os

class OneLayerConvNet(object):
	def __init__(self,input,filter_shape,image_shape,filters_init=None,bias_init = None, fix_filters = True,rng = None):
		self.fix_filters = fix_filters	
		if rng == None:
			self.rng = np.random.RandomState(23455)
		else:
			self.rng = rng
		assert image_shape[1]==filter_shape[1]
		if filters_init==None:
			fan_in = np.prod(filter_shape[1:])
			W_values = np.asarray(self.rng.uniform(
				low = -numpy.sqrt(3./fan_in),
				high = numpy.sqrt(3./fan_in),
				size = filter_shape),
				dtype = theano.config.floatX)
			self.W = theano.shared(value=W_values,name='W')
		else:
			assert filters_init.shape == filter_shape
			self.W = theano.shared(value=np.asarray(filters_init,dtype=theano.config.floatX),name='W')
		if bias_init==None:
			b_value = numpy.zeros((filter_shape[0],),dtype=theano.config.floatX)
			self.b = theano.shared(value = b_value, name = 'b')
		else:
			assert filters_init.shape[0] == bias_init.shape[0]
			self.b = theano.shared(value = np.asarray(bias_init,dtype=theano.config.floatX), name='b')

		conv_out = conv.conv2d(input,self.W,filter_shape = filter_shape, image_shape=image_shape)	
		self.output = T.tanh(conv_out+self.b.dimshuffle('x',0,'x','x'))
		self.params = [self.W,self.b]
	
	def get_output(self,input, filter_shape, image_shape):
		return T.tanh(conv.conv2d(input,self.W,filter_shape = filter_shape, image_shape = image_shape)+self.b.dimshuffle('x',0,'x','x'))

class LogisticRegression(object):
	'''
	Simple stupid class for logistic regression
	Simply takes the input, calculates the softmax of the log-linear predictions
	and that's it

	self.W
	'''
	def __init__(self, input, n_in, n_out):
		self.W = theano.shared(value = np.zeros((n_in,n_out),dtype=theano.config.floatX), name='W')
		self.b = theano.shared(value = np.zeros((n_out,),dtype=theano.config.floatX), name = 'b')

		self.p_y_given_x = T.nnet.softmax(T.dot(input,self.W)+self.b)

		self.y_pred = T.argmax(self.p_y_given_x,axis=1)

		self.params = [self.W,self.b]

	def negativeLL(self,y):
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])

	def get_cost_and_updates(self,y,learning_rate):
		cost = self.negativeLL(y)
		grads = T.grad(cost,self.params)
		updates = []
		for i,p in enumerate(self.params):
			updates.append((p,p-learning_rate*grads[i]))

		return cost,updates

	def get_cost(self,x,y):
		p_y_given_x = T.nnet.softmax(T.dot(x,self.W)+self.b)
		return -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]),y])

if __name__=='__main__':
	#Mnist has 70000 examples, we use 50000 for training, set 20000 aside for validation
	train_size = 50000
	train_data,validation_data = ld.load_data_mnist(train_size = train_size)

	train_data['images'] = train_data['images'][:20000,:,:,:]
	validation_data['images'] = validation_data['images'][:5000,:,:,:]
	train_data['labels'] = train_data['labels'][:20000]
	validation_data['labels'] = validation_data['labels'][:5000]

	output_file = 'ae_10x10_backup.pikle'

	#fiddle around, not sure which values to use
	training_epochs = 100
	training_batches=100
	patch_size = 10
	batch_size = int(train_data['images'].shape[0]/training_batches)
	n_filters = 30
	filter_shape = (n_filters,1,patch_size,patch_size)
	image_shape = (batch_size,1,28,28)
	validation_shape = (validation_data['images'].shape[0],1,28,28)
	index = T.lscalar()
	if output_file not in os.listdir('.'):

		batches = ld.make_vector_patches(train_data,training_batches,batch_size,patch_size)
		validation_images = ld.make_vector_patches(validation_data,1,validation_data['images'].shape[0],patch_size)
		#batches,ys = ld.make_vector_patches(train_data,training_batches,batch_size,patch_size)
		#validation_images,validation_ys = ld.make_vector_batches(validation_data,1,validation_data['images'].shape[0])

		x = T.dmatrix('x')
		
		#Creates a denoising autoencoder with 500 hidden nodes, could be changed as well
		a = dA(patch_size*patch_size,n_filters,data=x,regL=0.05)
		
		#sEt theano shared variables for the train and validation data
		data_x = theano.shared(value = np.asarray(batches,dtype=theano.config.floatX),name='data_x')
		validation_x = theano.shared(value = np.asarray(validation_images[0,:,:],dtype=theano.config.floatX),name='validation_x')

		#get cost and update functions for the autoencoder
		cost,updates = a.get_cost_and_updates(0.4,0.02)

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
	       #try:
	       #	finame = raw_input('Output to pickle?')
	       #except SyntaxError, NameError:

		W_ae = np.reshape(a.W.get_value(),(n_filters,1,patch_size,patch_size))
		b_ae = np.reshape(a.b_h.get_value(),(n_filters,))

		fi = open(output_file,'w')
		pic.dump([W_ae,b_ae],fi)
		fi.close()
	else:	
		fi = open(output_file,'r')
		[W_ae,b_ae] = pic.load(fi)
		fi.close()
	
	W_ae = np.reshape(W_ae,(n_filters,1,patch_size,patch_size))
	b_ae = np.reshape(b_ae,(n_filters,))

	t_x,t_y = ld.make_vector_batches(train_data,training_batches,batch_size)
	v_x,v_y = ld.make_vector_batches(validation_data,1,validation_data['images'].shape[0])
	

	train_x = theano.shared(value = t_x, name = 'train_x')
	train_y = theano.shared(value = np.asarray(t_y,dtype='int'), name = 'train_y')

	validation_x= theano.shared(value = v_x[0,:,:], name = 'validation_x')
	print v_y.shape
	print v_y[0,:].shape
	validation_y = theano.shared(value = np.asarray(v_y[0,:],dtype='int'), name = 'validation_y')

	inp = T.matrix('inp')
	val_inp = T.matrix('val_inp')
	y = T.ivector('y')
	val_y = T.ivector('val_y')
	data_conv = inp.reshape((batch_size,1,28,28))
	data_val_conv = val_inp.reshape(validation_shape)
	conv_net = OneLayerConvNet(data_conv,filter_shape,image_shape,filters_init = W_ae, bias_init = b_ae )
	log_reg_input = conv_net.output.flatten(2)
	log_reg_validation_input = conv_net.get_output(data_val_conv,filter_shape,validation_shape).flatten(2)
	n_in = n_filters*(28-patch_size+1)**2
	n_out = 10
	log_reg = LogisticRegression(log_reg_input,n_in,n_out)

	cost,updates = log_reg.get_cost_and_updates(y,0.01)
	val_cost = log_reg.get_cost(log_reg_validation_input,val_y)
	train_lr = theano.function([index],cost,
			updates = updates,
			givens=[(y,train_y[index]),(inp,train_x[index])])
	validation_lr = theano.function([],val_cost,
			givens = [(val_y,validation_y),(val_inp,validation_x)])

	print '\n\n------------\nNow training Logistic Regression with ConvNet\n------------\n'
	
	for epoch in xrange(training_epochs):
		c = []
		for batch in xrange(training_batches):
			c.append(train_lr(batch))
		ve = validation_lr()
		print 'LR training epoch %d, cost %lf, validation cost %lf' % (epoch, np.mean(c),ve)

