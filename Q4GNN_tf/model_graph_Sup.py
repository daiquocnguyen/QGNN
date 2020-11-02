import tensorflow as tf
from inits import *
from layers import *

class Q4GNN(object):
    def __init__(self, feature_dim_size, hidden_size, num_GNN_layers, num_classes):
        # Placeholders for input, output
        self.Adj_block = tf.compat.v1.sparse_placeholder(tf.float32, [None, None], name="Adj_block")
        self.X_concat = tf.compat.v1.sparse_placeholder(tf.float32, [None, feature_dim_size], name="X_concat")
        self.graph_pool = tf.compat.v1.sparse_placeholder(tf.float32, [None, None], name="graph_pool")
        self.one_hot_labels = tf.compat.v1.placeholder(tf.float32, [None, num_classes], name="one_hot_labels")
        self.num_features_nonzero = tf.compat.v1.placeholder(tf.int32, name="num_features_nonzero")
        self.dropout = tf.compat.v1.placeholder(tf.float32, name="dropout")

        self.placeholders = {
            'adj': self.Adj_block,
            'dropout': self.dropout,
            'num_features_nonzero': self.num_features_nonzero
        }

        self.input = self.X_concat   # set hidden_size = feature_dim_size if not tuning sizes of hidden stacked layers
        in_hidden_size = feature_dim_size
        #Construct k GNN layers
        self.scores = 0
        for idx_layer in range(num_GNN_layers):
            sparse_inputs = False
            if idx_layer == 0:
                sparse_inputs = True
            quaternion_gnn = QuaternionGraphNN1(input_dim=in_hidden_size,
                                                  output_dim=hidden_size,
                                                  placeholders=self.placeholders,
                                                  act=tf.nn.relu,
                                                  dropout=True,
                                                  sparse_inputs=sparse_inputs)
            in_hidden_size = hidden_size
            # run quaternion --> output --> input for next layer
            self.input = quaternion_gnn(self.input)
            # graph (sum) pooling
            self.graph_embeddings = dot(self.graph_pool, self.input, sparse=True)
            self.graph_embeddings = tf.nn.dropout(self.graph_embeddings, 1-self.dropout)

            # Similar to concatenate graph representations from all GNN layers
            with tf.compat.v1.variable_scope("layer_%d" % idx_layer):
                W = glorot([hidden_size, num_classes], name="W_layer_%d" % idx_layer)
                b = tf.Variable(tf.zeros([num_classes]))
                self.scores += tf.compat.v1.nn.xw_plus_b(self.graph_embeddings, W, b)

        # Final predictions
        self.predictions = tf.argmax(self.scores, 1, name="predictions")
        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=label_smoothing(self.one_hot_labels))
            self.total_loss = tf.reduce_mean(losses)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.one_hot_labels, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_predictions, "float"), name="accuracy")

        self.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=500)
        tf.compat.v1.logging.info('Seting up the main structure')

def label_smoothing(inputs, epsilon=0.1):
    V = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / V)