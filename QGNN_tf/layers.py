from inits import *
import tensorflow as tf

'''Quaternion Graph Neural Networks @ Dai Quoc Nguyen 2020'''

# Make Hamilton matrix
def make_quaternion_mul(kernel):
    """" The constructed 'hamilton' W is a modified version of the quaternion representation,
        thus doing tf.matmul(Input,W) is equivalent to W * Inputs. """
    r, i, j, k = tf.split(kernel, 4, axis=1)
    r2 = tf.concat([r, -i, -j, -k], axis=0)  # 0, 1, 2, 3
    i2 = tf.concat([i, r, -k, j], axis=0)  # 1, 0, 3, 2
    j2 = tf.concat([j, k, r, -i], axis=0)  # 2, 3, 0, 1
    k2 = tf.concat([k, -j, i, r], axis=0)  # 3, 2, 1, 0
    hamilton = tf.concat([r2, i2, j2, k2], axis=1)
    return hamilton

# Quaternion feedforward
def quaternion_ffn(x, dim, name='', activation=None, reuse=None):
    """ Implements quaternion feed-forward layer x is [bsz x features] tensor """
    input_dim = x.get_shape().as_list()[1] // 4
    with tf.compat.v1.variable_scope('Q{}'.format(name), reuse=reuse) as scope:
        kernel = glorot([input_dim, dim], name='quaternion_weights_'+name)
        hamilton = make_quaternion_mul(kernel)
        output = tf.matmul(x, hamilton)
        if (activation):
            output = activation(output)
        return output

'''Class QGNN layer'''
class QGNNLayer(Layer):
    """Quaternion Graph NN layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, quaternion_ff=True, act=tf.nn.relu, **kwargs):
        super(QGNNLayer, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.adj = placeholders['adj']
        self.sparse_inputs = sparse_inputs
        self.quaternion_ff = quaternion_ff

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.compat.v1.variable_scope(self.name + '_vars'):
            if self.quaternion_ff:
                # input_dim // 4 for quaternion
                self.vars['weights'] = glorot([input_dim//4, output_dim], name='weights')
            else:
                self.vars['weights'] = glorot([input_dim, output_dim], name='weights')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # compute Wu; u in {N(v) union v}
        if self.quaternion_ff:
            hamilton = make_quaternion_mul(self.vars['weights'])
        else:
            hamilton = self.vars['weights']
        pre_sup = dot(x, hamilton, sparse=self.sparse_inputs)
        # sigma Xu ~ adj matrix * pre_sup
        outputs = dot(self.adj, pre_sup, sparse=True)

        return self.act(outputs)

