#!/usr/bin/env python
"""
load mnist data from skdata library
"""

from skdata.mnist import view
from numpy.random import permutation
from numpy import reshape, prod
import numpy as np

def load_data_mnist():
	"""
	load the mnist dataset from the skdata package.
	"""
	print "---->\n.....Loading MNIST data from the skdata package\n"
	dataset = view.OfficialImageClassification()
	data = {}
	data['images'] = dataset.all_images/256.0
	data['labels'] = dataset.all_labels
	print "---->\n.....Done!"
	return data, data['images'].shape[0]


def make_vector_batches(data , nbatches, batch_size):
	"""
	Loads data in data into a list of numpy arrays with individual batches.
	batches are in batch_size x data_dim format 
	"""
	print "---->\n.....Putting into vector-shaped batches"
	assert nbatches*batch_size <= data['images'].shape[0]
	batches = np.asarray([])
	permut = permutation(data['images'].shape[0])
	xdata = []
	ydata = []
	for i in range(nbatches):
		batch = {}
		xs = data['images'][permut[i*batch_size:(i+1)*batch_size],:,:,:]
		xdata.append(reshape(xs,(batch_size,prod(xs.shape)/batch_size)))
		ydata.append(data['labels'][permut[i*batch_size:(i+1)*batch_size]])
	print "---->\n.....Done!
	return [np.reshape(np.asarray(xdata),(nbatches,batch_size,-1)),np.asarray(ydata)]

def make_batches(data , nbatches, batch_size):
	assert nbatches*batch_size <= data['images'].shape[0]
	batches = []
	permut = permutation(data['images'].shape[0])
	for i in range(nbatches):
		batch = {}
		batch['x'] = data['images'][permut[i*batch_size:(i+1)*batch_size],:,:,:]
		batch['y'] = data['labels'][permut[i*batch_size:(i+1)*batch_size]]
		batches = batches+[batch]
	return batches

if __name__=='__main__':
	data,_ = load_data_mnist()
	batches = make_vector_batches(data,10,10)
	print batches[0][0,:]
