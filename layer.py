
import tensorflow as tf
from tensorgraph.layers.template import Template

class WordsCombined(Template):

    def __init__(self, this_dim, mode='sum'):
        assert mode in ['sum', 'max', 'mean']
        self.mode = mode
        self.this_dim = this_dim


    def _train_fprop(self, X):
        '''
        X is of dimension (batch_size, sentence length, this_dim)
        '''
        self.train_in = X
        if self.mode == 'sum':
            self.train_out = tf.reduce_sum(X, reduction_indices=1)
        elif self.mode == 'max':
            self.train_out = f.reduce_max(X, reduction_indices=1)
        elif self.mode == 'mean':
            self.train_out = tf.reduce_mean(X, reduction_indices=1)
        return self.train_out


class Reshape(Template):

    def __init__(self, shape):
        self.shape = shape

    def _train_fprop(self, X):
        return tf.reshape(X, self.shape)


class Squeeze(Template):

    def __init__(self, squeeze_dims=None):
        '''
        PARAM:
            squeeze_dims: An optional list of ints. Defaults to []. If specified,
            only squeezes the dimensions listed. The dimension index starts at 0.
            It is an error to squeeze a dimension that is not 1. Refer to tensorflow
            for details.
        '''
        self.squeeze_dims = squeeze_dims

    def _train_fprop(self, state_below):
        return tf.squeeze(state_below, self.squeeze_dims)
