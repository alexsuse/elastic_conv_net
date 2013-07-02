#!/usr/bin/env python
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import time
from kmeans import KMeans
from conv_net import OneLayerConvNet, LogisticRegression
import load_data as ld
import cPickle as pic
import os



if __name__ == '__main__':
    #Mnist has 70000 examples, we use 50000 for training,
    # set 20000 aside for validation
    train_size = 50000
    train_data, validation_data = ld.load_data_mnist(train_size=train_size)

    #taking only a subset of the data,
    # as my laptop will burn to the ground otherwise
    #COMMENT THIS AWAY FOR FULL DATASET
    train_data['images'] = train_data['images'][:20000,:,:,:]
    validation_data['images'] = validation_data['images'][:5000,:,:,:]
    train_data['labels'] = train_data['labels'][:20000]
    validation_data['labels'] = validation_data['labels'][:5000]
    #ALL OF IT UP TO HERE

    #Random backup name, change at will, it will get a pickle of the
    # autoencoder weight and bias

    #set values to something useful, batch_size and number of epochs
    # doesn't seem to make much of a difference
    training_epochs = 300
    training_batches = 100

    #patch_size, for 28x28 images, 10x10 patches seemed reasonable
    patch_size = 10

    batch_size = int(train_data['images'].shape[0] / training_batches)
    print 'Training Epochs %d, Batch Size %d, Training Batches %d' % (
                            training_epochs, batch_size, training_batches)
    # n_filters is 30 so I can run it on my laptop,
    # reasonable values should be around 100, 200 at least.
    n_filters = 30

    output_file = ('km_' + str(patch_size) + 'x' + str(patch_size) + '_'
                   + str(n_filters) + '_filters_backup.pikle')

    #conv_net parameters
    filter_shape = (n_filters, 1, patch_size, patch_size)
    image_shape = (batch_size, 1, 28, 28)
    validation_shape = (validation_data['images'].shape[0], 1, 28, 28)

    #theano batch index
    index = T.lscalar()

    # if the autoencoder hasn't been trained and pickled beforehand,
    # train it and back it up now
    if output_file not in os.listdir('.'):
        '''
	KMEANS TRAINING STARTS HERE
        '''
        batches = ld.make_vector_patches(train_data, 1,train_data['images'].shape[0], patch_size)
	print batches.shape
	km = KMeans(batches[0,:,:],n_filters,show_results=True)
        
	W_km = np.reshape(km.prototypes, (n_filters, 1,
                                            patch_size, patch_size))
        b_km = np.zeros((n_filters,))

        fi = open(output_file, 'w')
        pic.dump([W_km, b_km], fi)
        fi.close()
        '''
        KMEANS TRAINING ENDS HERE
        '''

    else:
        # if autoencoder has been trained and backed up in the file named,
        # load it up from there
        fi = open(output_file, 'r')
        [W_km, b_km] = pic.load(fi)
        fi.close()

    #reshape ae matrices for conv.conv2d function
    W_km = np.reshape(W_km, (n_filters, 1, patch_size, patch_size))
    b_km = np.reshape(b_km, (n_filters,))

    #make vector batches, put in proper sizes
    t_x, t_y = ld.make_vector_batches(train_data, training_batches, batch_size)
    v_x, v_y = ld.make_vector_batches(validation_data, 1,
                                      validation_data['images'].shape[0])

    #theano shared variables for log_reg_training
    train_x = theano.shared(value=t_x, name='train_x')
    train_y = theano.shared(value=np.asarray(t_y, dtype='int'), name='train_y')

    #theano shared variables for log_reg_testing
    validation_x = theano.shared(value=v_x[0, :, :], name='validation_x')
    validation_y = theano.shared(value=np.asarray(v_y[0, :], dtype='int'),
                                 name='validation_y')

    #function inputs for cost and validation cost
    inp = T.matrix('inp')
    val_inp = T.matrix('val_inp')
    y = T.ivector('y')
    val_y = T.ivector('val_y')

    #reshaping inputs for conv_net
    data_conv = inp.reshape((batch_size, 1, 28, 28))
    data_val_conv = val_inp.reshape(validation_shape)

    #cbuild conv_net
    conv_net = OneLayerConvNet(data_conv, filter_shape, image_shape,
                               filters_init=W_km, bias_init=b_km)

    #log_reg inputs are conv_net_outputs but flattened
    log_reg_input = conv_net.output.flatten(2)
    log_reg_validation_input = conv_net.get_output(data_val_conv, filter_shape,
                                                validation_shape).flatten(2)

    #dimension of log_reginput
    n_in = n_filters * (28 - patch_size + 1) ** 2
    #10 classes
    n_out = 10

    #creat logistic regression object
    log_reg = LogisticRegression(log_reg_input, n_in, n_out)

    #cost and update functions for log_reg with labels and learning rate
    cost, updates = log_reg.get_cost_and_updates(y, 0.01)

    #val_cost gives the validation cost of log_reg
    val_cost = log_reg.get_cost(log_reg_validation_input, val_y)

    #theano functions for training and testing of log_reg
    train_lr = theano.function([index], cost,
            updates=updates,
            givens=[(y, train_y[index]), (inp, train_x[index])])
    validation_lr = theano.function([], val_cost,
            givens=[(val_y, validation_y), (val_inp, validation_x)])

    #errors on train and test set for lr
    err_train = log_reg.errors(log_reg_input, y)
    err_test = log_reg.errors(log_reg_validation_input, val_y)

    #theano functions for the errors
    errors_train = theano.function([index], err_train,
                        givens=[(y, train_y[index]), (inp, train_x[index])])
    errors_test = theano.function([], err_test,
                    givens=[(val_y, validation_y), (val_inp, validation_x)])

    print '\n\n------------\n'
    print 'Now training Logistic Regression with ConvNet'
    print '\n------------\n'

    for epoch in xrange(training_epochs):
        c = []  # accumulates costs
        train_es = []  # accumulates errors
        for batch in xrange(training_batches):
            c.append(train_lr(batch))
        for batch in xrange(training_batches):
            train_es.append(errors_train(batch))
        #test errors and validation
        test_e = errors_test()
        ve = validation_lr()
        train_e = np.mean(train_es)

        print ('LR training epoch %d, train cost %lf, training error %lf,'
               'validation cost %lf, validation error %lf') % (epoch,
                                            np.mean(c), train_e, ve, test_e)
    print ('---->\n.....Dumping logistic regression parameters'
           'to log_reg_params.pickle')

    fi = open("log_reg_params.pickle", 'w')
    pic.dump([log_reg.W.get_value(), log_reg.b.get_value()], fi)
    fi.close()
