#!/usr/bin/env python
import numpy as np
from random import sample
import matplotlib.pyplot as plt
import load_data as ld

class KMeans(object):

    def __init__(self,data, nfeatures):
        '''
        Kmeans pretraining class
        data should be a matrix of nexamples x ndimensions
        '''
        nexamples,dim = data.shape
	self.nfeatures = nfeatures
	#self.prototypes = np.random.uniform(-1,1,(nfeatures,dim))

	self.training_epochs = 10

   	self.prototypes = data[sample(xrange(nexamples),nfeatures),:]

	dists =np.zeros((self.nfeatures,nexamples))
	for i in xrange(self.training_epochs):
		print '---->\n.....training epoch %d'%i
		#dists = np.diag(np.dot(self.prototypes,self.prototypes.T))-np.dot(data,self.prototypes.T)
		for j in xrange(self.nfeatures):
			dists[j,:] = np.sum((data-self.prototypes[j,:])**2,axis=1)
		assignments = np.argmin(dists,axis=0)
		print assignments
		for j in xrange(self.nfeatures):
			self.prototypes[j,:] = np.mean(data[assignments==j,:],axis=0)


if __name__=='__main__':
	data,_ = ld.load_data_mnist(50000)
	batches = ld.make_vector_batches(data,1)
	nfeatures = 30
	km = KMeans(batches[0][0,:,:],nfeatures)
	for i in xrange(nfeatures):
		plt.imshow(km.prototypes[i,:].reshape((28,28)),interpolation='nearest')
		plt.show()
		if raw_input('continue?')!='y':
			break

	

