#!/usr/bin/env python
import numpy as np
from random import sample
import matplotlib.pyplot as plt
import load_data as ld

class KMeans(object):

    def __init__(self,data, nfeatures,show_results=False):
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
		dists = np.diag(np.dot(self.prototypes,self.prototypes.T))-2*np.dot(data,self.prototypes.T)
		assignments = np.argmin(dists.T,axis=0)
		for j in xrange(self.nfeatures):
			self.prototypes[j,:] = np.mean(data[assignments==j,:],axis=0)
	if show_results==True:
		for i in xrange(self.nfeatures):
			plt.imshow(self.prototypes[i,:].reshape((int(np.sqrt(dim)),int(np.sqrt(dim)))),interpolation='nearest')
			plt.show()



if __name__=='__main__':
	data,_ = ld.load_data_mnist(50000)
	batches = ld.make_vector_patches(data,1,50000,10)
	#batches = ld.make_vector_batches(data,1)
	nfeatures = 30
	#km = KMeans(batches[0][0,:,:],nfeatures)
	km = KMeans(batches[0,:,:],nfeatures)
	for i in xrange(nfeatures):
		#plt.imshow(km.prototypes[i,:].reshape((28,28)),interpolation='nearest')
		plt.imshow(km.prototypes[i,:].reshape((10,10)),interpolation='nearest')
		plt.show()
		#if raw_input('continue?')!='y':
		#	break

	

