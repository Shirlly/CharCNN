import tensorflow as tf
import tensorgraph as tg
from tensorgraph.node import StartNode, HiddenNode, EndNode
from tensorgraph.layers.linear import Linear
from tensorgraph.layers.activation import RELU, Softmax, Sigmoid
from tensorgraph.layers.merge import Concat, Mean, Sum
from tensorgraph.layers.misc import Embedding, Flatten, ReduceSum
from tensorgraph.layers.conv import Conv2D
from tensorgraph.layers.normalization import BatchNormalization
from tensorgraph.graph import Graph
from sklearn.preprocessing import OneHotEncoder
from layer import WordsCombined, Reshape, Squeeze
import numpy as np
from tensorgraph.data_iterator import SequentialIterator
from math import ceil
import pandas
from data import CharNumberEncoder
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.metrics import accuracy_score
from clean import processText
from data_new import makedata, makeinfodoc, onehot, twenty_newsgroup, infodocs, BASF_TFIDF
from data_new import BASF_char
import os
from utils import group_mse, group_accuracy, total_mse, total_accuracy

def same(in_height, in_width, strides, filters):
    out_height = ceil(float(in_height) / float(strides[0]))
    out_width  = ceil(float(in_width) / float(strides[1]))
    return out_height, out_width

def valid(in_height, in_width, strides, filters):
    out_height = ceil(float(in_height - filters[0] + 1) / float(strides[0]))
    out_width  = ceil(float(in_width - filters[1] + 1) / float(strides[1]))
    return out_height, out_width

def data1(num_train, word_len, sent_len, components):
    #### real dataset #####
    X, y = makedata(word_len, sent_len)
    shlfidx = np.arange(len(X))
    np.random.shuffle(shlfidx)
    X, y = X[shlfidx], y[shlfidx]
    train_X = X[:2000]
    valid_X = X[2000:]

    train_ys = [y[:2000]]
    valid_ys = [y[2000:]]
    return train_X, valid_X, train_ys, valid_ys

def data2(num_train, word_len, sent_len, components):
    ##### infodocs #####
    savdir = 'data/infodocs'
    Xpath = savdir + '/X.npy'
    ypath = savdir + '/y.npy'
    if not os.path.exists(savdir):
        os.makedirs(savdir)
    if not os.path.exists(Xpath):
        print '..creating X, y'
        X, comps = makeinfodoc(word_len, sent_len, max_comp_len=6)
        with open(Xpath, 'wb') as Xout:
            np.save(Xout, X)
        with open(ypath, 'wb') as yout:
            np.save(yout, comps)
        print '..saved done!'
    else:
        print '..loading X, y'
        with open(Xpath, 'rb') as Xin:
            X = np.load(Xin)
        with open(ypath, 'rb') as yin:
            comps = np.load(yin)
    # import pdb; pdb.set_trace()
    shlfidx = np.arange(len(X))
    np.random.shuffle(shlfidx)
    X = X[shlfidx]
    train_X = X[:num_train]
    valid_X = X[num_train:]
    train_ys = []
    valid_ys = []
    for l, n_comp in enumerate(components):
        y = comps[l][shlfidx]
        y = onehot(y, n_comp)
        print l, n_comp
        print 'y shape', y.shape
        train_ys.append(y[:num_train])
        valid_ys.append(y[num_train:])
    return train_X, valid_X, train_ys, valid_ys


def train():
    ### params
    sent_len = 50
    word_len = 20
    ch_embed_dim = 100
    unicode_size = 128
    tfidf_dim = 1000
    tfidf_embed_dim = 1000
    fc_dim = 1000
    batchsize = 32
    train_valid_ratio = [5, 1]
    learning_rate = 0.001
    # components = [len(np.unique(val)) for val in comps]
    # components = [65, 454, 983, 892, 242, 6]
    # components = [42]
    # components = [65]
    # num_train = 10000

    num_train=10000
    components = [65]
    train_X, valid_X, train_ys, valid_ys = infodocs(num_train, word_len, sent_len, components)
    
#    num_train = 560000
#    train_X, valid_X, train_ys, valid_ys, components = BASF_char(num_train,word_len, sent_len)
    
    #num_train=16000
    #components = [20]  
    #train_X, valid_X, train_ys, valid_ys = twenty_newsgroup(num_train, word_len, sent_len, components, use_sean=True)
    #num_train = 20000
    #train_X, valid_X, train_ys, valid_ys, components = BASF_char(num_train,word_len, sent_len)    
    print 'num train', len(train_X)
    print 'num valid', len(valid_X)

    trainset_X = SequentialIterator(train_X, batchsize=batchsize)
    trainset_y = SequentialIterator(*train_ys, batchsize=batchsize)
    validset_X = SequentialIterator(valid_X, batchsize=batchsize)
    validset_y = SequentialIterator(*valid_ys, batchsize=batchsize)
    ### define placeholders

    X_ph = tf.placeholder('int32', [None, sent_len, word_len])
    y_phs = []
    for comp in components:
        y_phs.append(tf.placeholder('float32', [None, comp]))


    ### define the graph model structure
    start = StartNode(input_vars=[X_ph])

    # character CNN
    embed_n = HiddenNode(prev=[start], layers=[Reshape(shape=(-1, word_len)),
                                               Embedding(cat_dim=unicode_size,
                                                         encode_dim=ch_embed_dim,
                                                         zero_pad=True),
                                               Reshape(shape=(-1, ch_embed_dim, word_len, 1))])

    h1, w1 = valid(ch_embed_dim, word_len, strides=(1,1), filters=(ch_embed_dim,4))
    conv1_n = HiddenNode(prev=[embed_n], layers=[Conv2D(input_channels=1, num_filters=10, padding='VALID',
                                                        kernel_size=(ch_embed_dim,4), stride=(1,1)),
                                                 RELU(),
                                                 Flatten(),
                                                 Linear(int(h1*w1*10), 1000),
                                                 RELU(),
                                                 Reshape((-1, sent_len, 1000)),
                                                 ReduceSum(1),
                                                 BatchNormalization(layer_type='fc', dim=1000, short_memory=0.01)
                                                 ])


    # conv2_n = HiddenNode(prev=[embed_n], layers=[Conv2D(input_channels=1, num_filters=1, padding='VALID',
    #                                                   kernel_size=(ch_embed_dim,2), stride=(1,1)),
    #                                              Squeeze()])
    # h2, w2 = valid(ch_embed_dim, word_len, strides=(1,1), filters=(ch_embed_dim,2))

    # conv3_n = HiddenNode(prev=[embed_n], layers=[Conv2D(input_channels=1, num_filters=1, padding='VALID',
    #                                                   kernel_size=(ch_embed_dim,3), stride=(1,1)),
    #                                              Squeeze()])
    # h3, w3 = valid(ch_embed_dim, word_len, strides=(1,1), filters=(ch_embed_dim,3))
    #
    # conv4_n = HiddenNode(prev=[embed_n], layers=[Conv2D(input_channels=1, num_filters=1, padding='VALID',
    #                                                   kernel_size=(ch_embed_dim,4), stride=(1,1)),
    #                                              Squeeze()])
    # h4, w4 = valid(ch_embed_dim, word_len, strides=(1,1), filters=(ch_embed_dim,4))
    # concat_n = HiddenNode(prev=[conv1_n, conv2_n, conv3_n, conv4_n],
    #                       input_merge_mode=Concat(), layers=[RELU()])
    # concat_n = HiddenNode(prev=[conv1_n, conv2_n],
    #                       input_merge_mode=Concat(), layers=[RELU()])
    # fc_n = HiddenNode(prev=[concat_n], layers=[Linear(int(w1+w2), fc_dim), Sigmoid()])
    #
    # # TF-IDF Embedding
    # words_combined_layer = WordsCombined(this_dim=tfidf_dim, mode='sum')
    # words_combined_n = HiddenNode(prev=[fc_n],
    #                               layers=[Linear(prev_dim=fc_dim, this_dim=tfidf_dim), Sigmoid(),
    #                                       Reshape(shape=(-1, sent_len, tfidf_dim)),
    #                                       words_combined_layer,
    #                                       BatchNormalization(dim=tfidf_dim, layer_type='fc', short_memory=0.01)])

    out_n = HiddenNode(prev=[conv1_n],
                       layers=[Linear(prev_dim=1000, this_dim=components[0]), Softmax()])

    # # hierachical softmax
    # prev_dim = components[0]
    # prev_node = HiddenNode(prev=[out_n], layers=[Linear(tfidf_embed_dim, prev_dim), Softmax()])
    # end_nodes = []
    # end_nodes.append(EndNode(prev=[prev_node]))
    # for this_dim in components[1:]:
    #     top_connect = HiddenNode(prev=[out_n], layers=[Linear(tfidf_embed_dim, prev_dim), Sigmoid()])
    #     prev_node = HiddenNode(prev=[prev_node, top_connect], layers=[Linear(prev_dim, this_dim), Softmax()])
    #     end_nodes.append(EndNode(prev=[prev_node]))
    #     prev_dim = this_dim
    end_nodes = [EndNode(prev=[out_n])]

    graph = Graph(start=[start], end=end_nodes)
    # import pdb; pdb.set_trace()

    train_outs_sb = graph.train_fprop()
    test_outs = graph.test_fprop()

    ttl_mse = []
    accus = []
    for y_ph, out in zip(y_phs, train_outs_sb):
        ttl_mse.append(tf.reduce_mean((y_ph-out)**2))
        pos = tf.reduce_sum((y_ph * out))
        accus.append(pos)

    mse = sum(ttl_mse)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(mse)

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)

        max_epoch = 50
        es = tg.EarlyStopper(max_epoch=max_epoch, epoch_look_back=3, percent_decrease=0.1)
        temp_acc = []

        for epoch in range(max_epoch):
            print 'epoch:', epoch
            train_error = 0
            train_accuracy = 0
            ttl_examples = 0
            for X_batch, ys in zip(trainset_X, trainset_y):
                feed_dict = {X_ph:X_batch[0]}
                for y_ph, y_batch in zip(y_phs, ys):
                    feed_dict[y_ph] = y_batch

                sess.run(optimizer, feed_dict=feed_dict)
                train_outs = sess.run(train_outs_sb, feed_dict=feed_dict)
                train_error += total_mse(train_outs, ys)[0]
                train_accuracy += total_accuracy(train_outs, ys)[0]
                ttl_examples += len(X_batch[0])
            print 'train mse', train_error/float(ttl_examples)
            print 'train accuracy', train_accuracy/float(ttl_examples)
                # print 'outputs'
                # ypred_train = sess.run(outs, feed_dict=feed_dict)
                #
                # print 'ypreds'
                # ypred = np.argmax(ypred_train[0],axis=1)
                # print ypred
                # print 'ylabels'
                # ylabel = np.argmax(ys[0],axis=1)
                # print ylabel
                # print 'mse'
                # print np.mean((ypred_train[0] - ys[0])**2)
                # for v in graph.variables:
                #     print v.name,
                #     print 'mean:', np.mean(np.abs(sess.run(v)))
                #     print 'std:', np.std(sess.run(v))
                #     print sess.run(v)
                # print '---------------------------------'
                # import pdb; pdb.set_trace()
                # ypreds = []
                # print 'words_combined_layer in',sess.run(tf.reduce_mean(words_combined_layer.train_in, reduction_indices=0), feed_dict=feed_dict)
                # print 'words_combined_layer out',sess.run(tf.reduce_mean(words_combined_layer.train_out, reduction_indices=0), feed_dict=feed_dict)
                # # for out in outs:
                #     ypreds.append(sess.run(out, feed_dict=feed_dict))
                # accus = []
                # for y_batch, ypred_batch in zip(ys, ypreds):
                    # accu = accuracy_score(y_batch.argmax(axis=1), ypred_batch.argmax(axis=1))
                    # accus.append(accu)
                # print accus

                # import pdb; pdb.set_trace()
                # train_error = sess.run(mse, feed_dict=feed_dict)
                # print 'train error:', train_error
                # for accu in accus:
                #     train_pos = sess.run(, feed_dict=feed_dict)

                # print sess.run(embed._W[0,:])
                #
                # print sess.run(embed.embedding[0,:])
                # print '--------------'
                # import pdb; pdb.set_trace()



            # train_error = sess.run(mse, feed_dict=feed_dict)
            # print 'train error:', train_error

            valid_error = 0
            valid_accuracy = 0
            ttl_examples = 0
            for X_batch, ys in zip(validset_X, validset_y):
                feed_dict = {X_ph:X_batch[0]}
                for y_ph, y_batch in zip(y_phs, ys):
                    feed_dict[y_ph] = y_batch

                valid_outs = sess.run(test_outs, feed_dict=feed_dict)
                valid_error += total_mse(valid_outs, ys)[0]
                valid_accuracy += total_accuracy(valid_outs, ys)[0]
                ttl_examples += len(X_batch[0])

            print 'valid mse', valid_error/float(ttl_examples)
            print 'valid accuracy', valid_accuracy/float(ttl_examples)
            temp_acc.append(valid_accuracy/float(ttl_examples))
            
        
        print 'average accuracy is:\t', sum(temp_acc)/len(temp_acc)
            

            # ypreds = []
            # for out in outs:
            #     ypreds.append(sess.run(out, feed_dict=feed_dict))
            # accus = []
            # for y_batch, ypred_batch in zip(valid_ys, ypreds):
            #     accu = accuracy_score(y_batch.argmax(axis=1), ypred_batch.argmax(axis=1))
            #     accus.append(accu)
            # print 'valid accuracy:', accus
            #
            # if es.continue_learning(valid_error=valid_error):
            #     print 'best valid error so far:', valid_error
            #     print 'best epoch last update:', es.best_epoch_last_update
            #     print 'best valid last update:', es.best_valid_last_update
            #     print 'self.epoch', es.epoch
            #     print 'pass'
            # else:
            #     print 'training done!'
            #     break




if __name__ == '__main__':
    train()
