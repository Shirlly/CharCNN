
import numpy as np
import string
import pandas
import os
from math import ceil
from clean import processText
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from clean import preprocessText
import sys
import csv
dname = os.path.dirname(os.path.realpath(__file__))
sys.path += [os.path.dirname(dname) + '/TicketCategorization']




class CharNumberEncoder(object):

    def __init__(self, data_iterator, word_len=30, sent_len=200):
        '''
        DESCRIPTIONS:
            This class converts text to numbers for the standard unicode vocabulary
            size.
        PARAMS:
            data_iterator (iterator): iterator to iterates the text strings
            word_len (int): maximum length of the word, any word of length less
                than that will be padded with zeros, any word of length more than
                that will be cut at max word length.
            sent_len (int): maximum number of words in a sentence, any sentence
                with less number of words than that will be padded with zeros,
                any sentence with more words than the max number will be cut at
                the max sentence length.
        '''
        self.data_iterator = data_iterator
        self.word_len = word_len
        self.sent_len = sent_len
        self.char_map = {}
        for i, ch in enumerate(string.printable):
            self.char_map[ch] = i+1 # hash character to number, leave 0 for blank space


    def make_char_embed(self):
        '''build array vectors of words and sentence, automatically skip non-ascii
           words.
        '''
        sents = []
        for paragraph in self.data_iterator:
            word_toks = paragraph.split(' ')
            word_vec = []
            for word in word_toks:
                word = word.strip()
                try:
                    word.encode('ascii')
                except:
                    print '..Non ASCII Word', word
                    continue
                if len(word) > 0:
                    word_vec.append(self.spawn_word_vec(word))

            if len(word_vec) > self.sent_len:
                sents.append(word_vec[:self.sent_len])
            else:
                zero_pad = np.zeros((self.sent_len-len(word_vec), self.word_len))
                if len(word_vec) > 0:
                    sents.append(np.vstack([np.asarray(word_vec), zero_pad]))
                else:
                    sents.append(zero_pad)

        return np.asarray(sents)


    def spawn_word_vec(self, word):
        '''Convert a word to number vector with max word length, skip non-ascii
           characters
        '''
        word_vec = []
        for c in word:
            try:
                assert c in self.char_map and c != ' ', '({}) of {} not in char map'.format(c,word)
            except:
                continue
            word_vec.append(self.char_map[c])
        if len(word_vec) > self.word_len:
            return word_vec[:self.word_len]
        else:
            word_vec += [0]*(self.word_len-len(word_vec))
        return word_vec


def makedata(word_len, sent_len):
    '''infodocs on unfiltered data
    '''
    datapath = './route_1_10.txt'
    df = pandas.read_csv(datapath, sep='\t&&\t')
    text = df[df.columns[0]].values
    components = df[df.columns[1]].values
    clean_text = processText(text)
    data = CharNumberEncoder(clean_text, word_len=word_len, sent_len=sent_len)
    X = data.make_char_embed()
    colname = df.columns[1]
    y_df = df[colname].astype('category')
    num_cat = len(y_df.cat.categories)
    y_df.cat.rename_categories(np.arange(num_cat)+1, inplace=True)
    y_df.cat.add_categories([0], inplace=True)
    y_df.fillna(0, inplace=True)
    y_ls = np.asarray(y_df.tolist())[:, np.newaxis]
    # import pdb; pdb.set_trace()
    encoder = OneHotEncoder(n_values=num_cat+1)
    y = encoder.fit_transform(y_ls).toarray()
    return X, y


def makeinfodoc(word_len, sent_len, max_comp_len):
    # datapath = 'data/sample.csv'
    zero_hash = '0x0x0x0x'

    # df = pandas.read_csv(datapath)[10:]

    # text = df[df.columns[0]].values
    # components = df[df.columns[1]].values
    from preparation.prepare import prepare_wo_transform

    incidents = prepare_wo_transform()
    # import pdb; pdb.set_trace()
    text = np.asarray(incidents)[:, 1]
    components = np.asarray(incidents)[:, 3]

    clean_text = processText(text)

    comp_in_str = []
    for comp in components:
        toks = comp.split('-')
        if len(toks) < max_comp_len:
            toks += [zero_hash]*(max_comp_len-len(toks))
        assert len(toks) == max_comp_len, toks
        comp_in_str.append(toks)

    comp_in_str = np.asarray(comp_in_str)
    comp_in_num = []
    for level in np.arange(max_comp_len):
        clss = np.unique(comp_in_str[:,level])
        clss_tbl = {}
        for i, cl in enumerate(clss):
            clss_tbl[cl] = i

        cim = map(lambda x:clss_tbl[x], comp_in_str[:,level])
        cim = np.asarray(cim)[:, np.newaxis]
        comp_in_num.append(cim)

    data = CharNumberEncoder(clean_text, word_len=word_len, sent_len=sent_len)
    X = data.make_char_embed()
    return X, np.asarray(comp_in_num)


def onehot(X, nclass):
    encoder = OneHotEncoder(n_values=nclass)
    return encoder.fit_transform(X).toarray()


def data1(num_train, word_len, sent_len, components):
    #### real dataset #####
    X, y = makedata(word_len, sent_len)
    shlfidx = np.arange(len(X))
    np.random.shuffle(shlfidx)
    X, y = X[shlfidx], y[shlfidx]
    train_X = X[:num_train]
    valid_X = X[num_train:]

    train_ys = [y[:num_train]]
    valid_ys = [y[num_train:]]
    return train_X, valid_X, train_ys, valid_ys

def infodocs(num_train, word_len, sent_len, components):
    ##### infodocs #####
    savdir = 'data/infodocs'
    Xpath = savdir + '/X_{}_{}.npy'.format(word_len, sent_len)
    ypath = savdir + '/y_{}_{}.npy'.format(word_len, sent_len)
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
    shlfidx = np.arange(len(X))
    np.random.shuffle(shlfidx)
    X = X[shlfidx]
    train_X = X[:num_train]
    valid_X = X[num_train:]
    print valid_X
    train_ys = []
    valid_ys = []
    for l, n_comp in enumerate(components):
        y = comps[l][shlfidx]
        print '\n*****************\n',y
        y = onehot(y, n_comp)
        print l, n_comp
        print 'y shape', y.shape
        train_ys.append(y[:num_train])
        valid_ys.append(y[num_train:])
    return train_X, valid_X, train_ys, valid_ys


def twenty_newsgroup(num_train, word_len, sent_len, components=[20], use_sean=True):
    '''20 news group flatten labels
       use_sean: follows sean's preprocessing step
    '''
    from sklearn.datasets import fetch_20newsgroups

    savdir = 'data/twenty_newsgroup'
    Xpath = savdir + '/X_{}_{}.npy'.format(word_len, sent_len)
    ypath = savdir + '/y_{}_{}.npy'.format(word_len, sent_len)

    if not os.path.exists(savdir):
        os.makedirs(savdir)
    if not os.path.exists(Xpath):
        print '..creating X, y'
        dataset = fetch_20newsgroups(subset='all')
        if use_sean:
            from preparation.twenty_newsgroups_parser import preprocess
            clean_text = preprocess(dataset['data'])
        else:
            clean_text = processText(dataset['data'])
        data = CharNumberEncoder(clean_text, word_len=word_len, sent_len=sent_len)
        X = data.make_char_embed()
        comps = [dataset['target']]
        # X, comps = makeinfodoc(word_len, sent_len, max_comp_len=6)
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

    shlfidx = np.arange(len(X))
    np.random.shuffle(shlfidx)
    X = X[shlfidx]
    train_X = X[:num_train]
    valid_X = X[num_train:]
    train_ys = []
    valid_ys = []

    for l, n_comp in enumerate(components):
        y = comps[l][shlfidx]
        y = onehot(y[:,np.newaxis], n_comp)
        print l, n_comp
        print 'y shape', y.shape
        train_ys.append(y[:num_train])
        valid_ys.append(y[num_train:])
    return train_X, valid_X, train_ys, valid_ys




    
def labelIndex(y):
    label = {}
    label_idx = 1
    y_idx = []
    for i in range(len(y)):
        if y[i] not in label:
            label[y[i]] = label_idx
            label_idx += 1
        y_idx.append(label[y[i]])
    return y_idx, label


def BASF_TFIDF(num_train):
    df = pandas.read_csv('basf.csv')    
    X = df.Question
    y = df.Area
    X_text = []
    y_label = []
    if len(X)!=len(y):
        print 'Content and label are not of same size...'
    else:
        for i in range(len(X)):
            if type(X[i]) is str and type(y[i]) is str:
                if X[i].strip() and y[i].strip():
                    X_text.append(X[i])
                    y_label.append(y[i])
    X_text = preprocessText(X_text)
    tv = TfidfVectorizer(decode_error = 'ignore', min_df = 3, max_df = 0.8)
    X_tfidf = tv.fit_transform(X_text).toarray()
    y_idx, label = labelIndex(y_label)
    y_idx = np.asarray(y_idx)
    nclass = len(label)
    shlfidx = np.arange(len(X_text))
    np.random.shuffle(shlfidx)
    X_idx = X_tfidf[shlfidx]
    train_X = X_idx[:num_train]
    valid_X = X_idx[num_train:]
    train_ys = []
    valid_ys = []
    components = [nclass+1]
    for l, n_comp in enumerate(components):
        y = y_idx[shlfidx]
        y = onehot(y[:,np.newaxis], n_comp)
        print l, n_comp
        print 'y shape', y.shape
        train_ys.append(y[:num_train])
        valid_ys.append(y[num_train:])

    return train_X, valid_X, train_ys, valid_ys, nclass


if __name__ == '__main__':
    twenty_newsgroup(100,10,10)
    BASF_TFIDF()
    # makedata(10, 20)
    # import pdb; pdb.set_trace()
    # import sys
    # sys.path += ['/Users/i329486/Projects/TicketCategorization']
    # X, comps = makeinfodoc(10,20,max_comp_len=6)
    # datapath = './route_1_10_clean.txt'
    # df = pandas.read_csv(datapath, sep='\t&&\t')
    # data = CharNumberEncoder(df[df.columns[0]].values)
    # data.process()
    # import pdb; pdb.set_trace()
    # print
