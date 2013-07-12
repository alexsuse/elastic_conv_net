#!/usr/bin/env python
"""
Implements a Restricted Boltzmann Machine Class
"""
from theano import tensor as T
from theano import shared
import numpy as np
import theano
import load_data as ld
import cPickle as pic
from theano.printing import Print
from theano.tensor.shared_randomstreams import RandomStreams
import matplotlib.pyplot as plt
#theano.config.compute_test_value = 'warn'

class RBM:
    """
    The classic RBM of yore.
    """
    def __init__(self, nvisible, nhidden, data=None, Wi=None, bv=None,
                 bh=None, rng=None, theano_rng=None, sparse = 0.1):
        self.nhidden = nhidden
        self.nvisible = nvisible
        self.sparse = sparse
        #hidden to visible matrix
        if Wi == None:
            Wi = np.asarray(np.random.uniform(
                low=-4 * np.sqrt(6. / (nhidden + nvisible)),
                high=4 * np.sqrt(6. / (nhidden + nvisible)),
                size=(nvisible, nhidden)), dtype=theano.config.floatX)
            W = shared(value=Wi, name='W')
        else:
            W = shared(value=Wi, name='W')
        self.W = W
        self.Wprime = W.T

        #biases
        if bh == None:
            bi_h = np.asarray(np.zeros(nhidden,), dtype=theano.config.floatX)
            b_h = shared(value=bi_h, name='b_h')
        else:
            b_h = shared(value=bh, name='b_h')
        if bv == None:
            bi_v = np.asarray(np.zeros(nvisible,), dtype=theano.config.floatX)
            b_v = shared(value=bi_v, name='b_v')
        else:
            b_v = shared(value=bv, name='b_v')
        self.b_v = b_v
        self.b_h = b_h

        #variances
        vvar = shared(value = np.ones_like(bi_v, dtype = theano.config.floatX), name = 'vvar')
        self.vvar = vvar

        #epsilon
        self.epsilon = 0.01

        if data==None:
            print 'no data'
            self.data = T.matrix('data')
        else:
            self.data = data
        if rng==None:
            self.rng = np.random.RandomState(12345)
        else:
            self.rng = rng
        if theano_rng == None:
            self.theano_rng = T.shared_randomstreams.RandomStreams(
                                                self.rng.randint(2 ** 30))
        else:
            self.theano_rng = theano_rng

        self.params = [self.W, self.b_h, self.b_v, self.vvar]
        #self.params = [self.W, self.b_h, self.b_v]

    def sample_h_given_v(self,vis): 
        pre_sig = T.dot( vis,self.W) + self.b_h
        prob = T.nnet.sigmoid(pre_sig)
        hsample =  self.theano_rng.binomial(size = prob.shape, n = 1, p = prob, dtype= theano.config.floatX)
        #hsample = prob
        return [pre_sig, prob, hsample]

    def sample_v_given_h(self,hid):
        mean = T.dot(hid,self.W.T) + self.b_v
        #v_sample = 0.5*self.theano_rng.normal(size = mean.shape, dtype= theano.config.floatX)*(self.vvar+self.epsilon)**2 + mean
        v_sample = 0.5*T.sqrt(T.sqrt(self.vvar)+self.epsilon)*self.theano_rng.normal(size = mean.shape, dtype=theano.config.floatX)+mean
        #v_sample = mean
        return [mean, v_sample]

    def free_energy(self, vsample):
        #visible_term = T.sum(T.dot(vsample*(self.vvar+self.epsilon)**2,vsample.T)*0.5,axis=1)
        varterm = 1.0/(T.sqr(self.vvar)+self.epsilon)
        visible_term = T.sum(T.sqr((vsample-self.b_v))*varterm,axis=1)*0.5
        exponent = T.dot(vsample,self.W)+self.b_h
        hidden_term = T.sum(T.log(1+T.exp(exponent)),axis=1)
        return visible_term - hidden_term

    def gibbs_hvh(self,hid):
        mean, v_sample = self.sample_v_given_h(hid)
        pre_sig, prob, h_sample = self.sample_h_given_v(v_sample)
        return [mean, v_sample, pre_sig, prob, h_sample]

    def get_cost_and_updates(self, persistent_chain = None, k = 1, learning_rate = 0.00001):


        h_presig, h_mean, phsample = self.sample_h_given_v(self.data)
        if persistent_chain == None:
            persistent = False
            chain_start = phsample
        else:
            persistent = True
            chain_start = persistent_chain

        [vmeans, vsamples ,presigs, probs, hsamples], updates = theano.scan(self.gibbs_hvh,
                outputs_info = [None, None, None, None, chain_start],
                n_steps = k)
        chain_end = vsamples[-1]

        cost = T.mean(self.free_energy(self.data))-T.mean(self.free_energy(chain_end))

        gparams = T.grad(cost, self.params, consider_constant = [chain_end])

        for param, gparam in zip(self.params, gparams):
            updates[param] = param-T.cast(learning_rate,dtype=theano.config.floatX)*\
                            T.cast(gparam,dtype=theano.config.floatX)
            #updates.append((param, param - T.cast(learning_rate,dtype = theano.config.floatX) *
            #                 T.cast(gparam, dtype=theano.config.floatX)))

        if param==self.b_h:
            updates[param] = param + ((T.cast(self.sparse,dtype=theano.config.floatX)
                - probs.mean(0))*T.cast(learning_rate*0.1,dtype=theano.config.floatX))
        if persistent:
            updates[persistent_chain] = hsamples[-1]
            #updates.append((persistent_chain, hsamples[-1]))
        return (cost, updates)


if __name__ == '__main__':
    #Mnist has 70000 examples, we use 50000 for training
    # set 20000 aside for validation
    train_size = 60000
    train_data, validation_data = ld.load_data_mnist(train_size=train_size)

    train_data['images'] = train_data['images'][:20000,:,:,:]
    validation_data['images'] = validation_data['images'][:5000,:,:,:]
    train_data['labels'] = train_data['labels'][:20000]
    validation_data['labels'] = validation_data['labels'][:5000]

    #fiddle around, not sure which values to use
    training_epochs = 1000
    training_batches = 100
    patch_size = 28
    batch_size = int(train_data['images'].shape[0] / training_batches)
    #batches = ld.make_vector_patches(train_data, training_batches,
    #                                 batch_size, patch_size)
    #validation_images = ld.make_vector_patches(validation_data, 1,
    #                            validation_data['images'].shape[0], patch_size)
    #batches,ys = ld.make_vector_patches(train_data,training_batches,batch_size,patch_size)
    batches,ys = ld.make_vector_batches(train_data,training_batches,batch_size)
    validation_images,validation_ys = ld.make_vector_batches(validation_data,1,validation_data['images'].shape[0])

    #layer sizes
    nvisible = patch_size**2
    nhidden = 50

    index = T.lscalar()
    x = T.matrix('x')

    persistent_chain = theano.shared(value = np.ones((batch_size, nhidden), 
        dtype=theano.config.floatX),
        name = 'persistent_chain')

    #Creates a denoising autoencoder with 500 hidden nodes
    rbm= RBM(nvisible, nhidden, data=x)

    #sEt theano shared variables for the train and validation data
    data_x = theano.shared(value=np.asarray(batches,
                                    dtype=theano.config.floatX), name='data_x')
    #validation_x = theano.shared(value=np.asarray(validation_images[0, :, :],
     #                       dtype=theano.config.floatX), name='validation_x')

    #get cost and update functions for the rbm
    cost, updates = rbm.get_cost_and_updates(persistent_chain = persistent_chain, k = 5, learning_rate = 0.00001)
    #cost, updates = rbm.get_cost_and_updates(k = 5)

    #train_da returns the current cost and updates the rbm parameters,
    #index gives the batch index.
    train_rbm = theano.function([index], cost, updates=updates,
                        givens=[(x, data_x[index])], on_unused_input='ignore', name = 'train_rbm')
    #validation_error just returns the cost on the validation set
#    validation_error = theano.function([], cost,
 #                       givens=[(x, validation_x)], on_unused_input='ignore')

    #loop over training epochs
    print '--->\n....Now Training RBM\n'
    
    for epoch in xrange(training_epochs):
        c = []
        #ve = validation_error()
        #loop over batches
        for batch in xrange(training_batches):
            #collect costs for this batch
            c.append(train_rbm(batch))

        #print mean training cost in this epoch
        #and final validation cost for checking
        #print 'Training epoch %d, cost %lf, validation cost %lf' % (epoch,
        #                                                         np.mean(c), ve)
        if epoch==training_epochs-1 and epoch!=0:
            for ind in xrange(nhidden):
                plt.imshow(np.reshape( rbm.W.get_value()[ :, ind], (28, 28)), interpolation='nearest', cmap=plt.cm.gray)
                plt.colorbar()
                plt.show()
                plt.imshow(np.reshape(rbm.vvar.get_value(),(28,28)), interpolation='nearest', cmap=plt.cm.gray)
                plt.colorbar()
                plt.show()
        print 'Training epoch %d, cost %lf' %(epoch,np.mean(c))
    
    finame = 'output_pickle_rbm'
    fi = open(finame, 'w')
    b = [rbm.W.get_value(), rbm.b_h.get_value(), rbm.b_v.get_value()]
    pic.dump(b, fi)
