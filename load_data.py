#!/usr/bin/env python
"""
load mnist data from skdata library
"""

from skdata.mnist import view
from numpy.random import permutation
from numpy import reshape, prod
import numpy as np

def load_data_mnist(train_size=None):
	"""
	load the mnist dataset from the skdata package.
	"""
	print "---->\n.....Loading MNIST data from the skdata package"
	dataset = view.OfficialImageClassification()
	if train_size:
		train_data = {}
		validation_data = {}
		order = permutation(dataset.all_images.shape[0])
		train_data['images'] = dataset.all_images[order[:train_size]]/256.0
		train_data['labels'] = dataset.all_labels[order[:train_size]]
		validation_data['images'] = dataset.all_images[order[train_size:]]/256.0
		validation_data['labels'] = dataset.all_labels[order[train_size:]]
	else:
		train_data = {}
		validation_data = {}
		train_data['images'] = dataset.all_images/256.0
		train_data['labels'] = dataset.all_labels
	print "---->\n.....Done!"
	return train_data, validation_data


def make_localized_batches(data,nbatches,batch_size,field_size, stride):
	"""
	Makes localized batches of patches of the MNIST data.
	These can be randomly indexed to get a selection of 
	"""
	batches = {}
	for i in xrange(0,data['images'].shape[1]-field_size+1,stride):
		for j in xrange(0,data['images'].shape[2]-field_size+1,stride):
			batches[(i,j)] = make_unsupervised_batches(data['images'][:,i:i+field_size,j:j+field_size,:],nbatches,batch_size)
	return batches

def make_unsupervised_batches(data,nbatches,batch_size):
	"""
	Same as make_vector_batches, but doesn't include label data
	"""
	print '---->\n.....Putting data into vector-shaped batches'
	assert nbatches*batch_size <= data.shape
	permut = permutation(data.shape[0])
	xdata = []
	for i in xrange(nbatches):
		xs = data[permut[i*batch_size:(i+1)*batch_size],:,:,:]
		xdata.append(reshape(xs,(batch_size,prod(xs.shape)/batch_size)))
	return np.reshape(np.asarray(xdata),(nbatches,batch_size,-1))

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
		xs = data['images'][permut[i*batch_size:(i+1)*batch_size],:,:,:]
		xdata.append(reshape(xs,(batch_size,prod(xs.shape)/batch_size)))
		ydata.append(data['labels'][permut[i*batch_size:(i+1)*batch_size]])
	print "---->\n.....Done!"
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
	data,_ = load_data_mnist(train_size = 50000)
	local_batches = make_localized_batches(data,10,10,14,14)
	print local_batches[(0,0)].shape
