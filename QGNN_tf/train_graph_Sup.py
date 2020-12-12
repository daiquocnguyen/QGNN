#! /usr/bin/env python

import tensorflow as tf
import numpy as np
np.random.seed(123)
tf.compat.v1.set_random_seed(123)

import os
from model_graph_Sup import Q4GNN
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.sparse import coo_matrix
from utils_graph_cls import *

# Parameters
# ==================================================

parser = ArgumentParser("Q4GNN", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

parser.add_argument("--run_folder", default="../", help="")
parser.add_argument("--dataset", default="PTC", help="Name of the dataset.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate")
parser.add_argument("--batch_size", default=4, type=int, help="Batch Size")
parser.add_argument("--num_epochs", default=100, type=int, help="Number of training epochs")
parser.add_argument("--saveStep", default=1, type=int, help="")
parser.add_argument("--allow_soft_placement", default=True, type=bool, help="Allow device soft device placement")
parser.add_argument("--log_device_placement", default=False, type=bool, help="Log placement of ops on devices")
parser.add_argument("--model_name", default='MUTAG', help="")
parser.add_argument("--dropout", default=0.5, type=float, help="Dropout")
parser.add_argument("--num_GNN_layers", default=2, type=int, help="Number of stacked layers")
parser.add_argument("--hidden_size", default=256, type=int, help="Hidden_size//4 = number of quaternion units within each hidden layer.")
parser.add_argument('--fold_idx', type=int, default=8, help='the index of fold in 10-fold validation. 0-9.')
args = parser.parse_args()

print(args)

# Load data
print("Loading data...")

degree_as_tag = False
if args.dataset == 'COLLAB' or args.dataset == 'IMDBBINARY' or args.dataset == 'IMDBMULTI' or\
    args.dataset == 'REDDITBINARY' or args.dataset == 'REDDITMULTI5K' or args.dataset == 'REDDITMULTI12K':
    degree_as_tag = True

graphs, num_classes = load_graph_data(args.dataset, degree_as_tag)
train_graphs, test_graphs = separate_data(graphs, args.fold_idx)
feature_dim_size = graphs[0].node_features.shape[1]

def get_Adj_matrix(batch_graph):
    edge_mat_list = []
    start_idx = [0]
    for i, graph in enumerate(batch_graph):
        start_idx.append(start_idx[i] + len(graph.g))
        edge_mat_list.append(graph.edge_mat + start_idx[i])

    Adj_block_idx = np.concatenate(edge_mat_list, 1)
    Adj_block_elem = np.ones(Adj_block_idx.shape[1])

    #self-loop
    num_node = start_idx[-1]
    self_loop_edge = np.array([range(num_node), range(num_node)])
    elem = np.ones(num_node)
    Adj_block_idx = np.concatenate([Adj_block_idx, self_loop_edge], 1)
    Adj_block_elem = np.concatenate([Adj_block_elem, elem], 0)

    Adj_block = coo_matrix((Adj_block_elem, Adj_block_idx), shape=(num_node, num_node))

    return Adj_block

def get_graphpool(batch_graph):
    start_idx = [0]
    # compute the padded neighbor list
    for i, graph in enumerate(batch_graph):
        start_idx.append(start_idx[i] + len(graph.g))

    idx = []
    elem = []
    for i, graph in enumerate(batch_graph):
        elem.extend([1] * len(graph.g))
        idx.extend([[i, j] for j in range(start_idx[i], start_idx[i + 1], 1)])

    elem = np.array(elem)
    idx = np.array(idx)

    graph_pool = coo_matrix((elem, (idx[:, 0], idx[:, 1])), shape=(len(batch_graph), start_idx[-1]))
    return graph_pool

def get_batch_data(batch_graph):
    # features
    X_concat = np.concatenate([graph.node_features for graph in batch_graph], 0)
    # A + Ai + Aj + Ak
    X_concat = np.tile(X_concat, 4)  # feature_dim_size*4
    X_concat = coo_matrix(X_concat)
    X_concat = sparse_to_tuple(X_concat)
    # adj
    Adj_block = get_Adj_matrix(batch_graph)
    Adj_block = sparse_to_tuple(Adj_block)
    # graph pool
    graph_pool = get_graphpool(batch_graph)
    graph_pool = sparse_to_tuple(graph_pool)

    graph_labels = np.array([graph.label for graph in batch_graph])
    one_hot_labels = np.zeros((graph_labels.size, num_classes))
    one_hot_labels[np.arange(graph_labels.size), graph_labels] = 1
    num_features_nonzero = X_concat[1].shape
    return Adj_block, X_concat, graph_pool, one_hot_labels, num_features_nonzero

class Batch_Loader(object):
    def __call__(self):
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]
        batch_graph = [train_graphs[idx] for idx in selected_idx]
        Adj_block, X_concat, graph_pool, one_hot_labels, num_features_nonzero = get_batch_data(batch_graph)
        return Adj_block, X_concat, graph_pool, one_hot_labels, num_features_nonzero
batch_nodes = Batch_Loader()

# Adj_block, X_concat, graph_pool, one_hot_labels, num_features_nonzero = batch_nodes()
# print(Adj_block)
# print(graph_pool)
# print(X_concat)
# print(one_hot_labels)
# print(num_features_nonzero)

print("Loading data... finished!")
# Training
# ==================================================
with tf.Graph().as_default():
    session_conf = tf.compat.v1.ConfigProto(allow_soft_placement=args.allow_soft_placement, log_device_placement=args.log_device_placement)
    session_conf.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=session_conf)
    with sess.as_default():
        global_step = tf.Variable(0, name="global_step", trainable=False)
        q4gnn = Q4GNN(feature_dim_size=feature_dim_size*4,  # A + Ai + Aj + Ak
                      hidden_size=args.hidden_size,
                      num_GNN_layers=args.num_GNN_layers,
                      num_classes=num_classes
                      )

        # Define Training procedure
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.learning_rate)
        grads_and_vars = optimizer.compute_gradients(q4gnn.total_loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        out_dir = os.path.abspath(os.path.join(args.run_folder, "../runs_Q4GNN_Sup", args.model_name))
        print("Writing to {}\n".format(out_dir))

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Initialize all variables
        sess.run(tf.compat.v1.global_variables_initializer())
        graph = tf.compat.v1.get_default_graph()

        def train_step(Adj_block, X_concat, graph_pool, one_hot_labels, num_features_nonzero):
            feed_dict = {
                q4gnn.Adj_block: Adj_block,
                q4gnn.X_concat: X_concat,
                q4gnn.graph_pool: graph_pool,
                q4gnn.one_hot_labels: one_hot_labels,
                q4gnn.num_features_nonzero: num_features_nonzero,
                q4gnn.dropout: args.dropout
            }
            _, step, loss = sess.run([train_op, global_step, q4gnn.total_loss], feed_dict)
            return loss

        def eval_step(Adj_block, X_concat, graph_pool, one_hot_labels, num_features_nonzero):
            feed_dict = {
                q4gnn.Adj_block: Adj_block,
                q4gnn.X_concat: X_concat,
                q4gnn.graph_pool: graph_pool,
                q4gnn.one_hot_labels: one_hot_labels,
                q4gnn.num_features_nonzero: num_features_nonzero,
                q4gnn.dropout: 0.0
            }
            _, acc = sess.run([global_step, q4gnn.accuracy], feed_dict)
            return acc

        write_acc = open(checkpoint_prefix + '_acc.txt', 'w')
        num_batches_per_epoch = int((len(train_graphs) - 1) / args.batch_size) + 1
        for epoch in range(1, args.num_epochs+1):
            loss = 0
            for _ in range(num_batches_per_epoch):
                Adj_block, X_concat, graph_pool, one_hot_labels, num_features_nonzero = batch_nodes()
                loss += train_step(Adj_block, X_concat, graph_pool, one_hot_labels, num_features_nonzero)
                # current_step = tf.compat.v1.train.global_step(sess, global_step)
            print(loss)

            acc_output = []
            #evaluating
            idx = np.arange(len(test_graphs))
            for i in range(0, len(test_graphs), args.batch_size):
                sampled_idx = idx[i:i+args.batch_size]
                if len(sampled_idx) == 0:
                    continue
                batch_test_graphs = [test_graphs[j] for j in sampled_idx]
                test_Adj_block, test_X_concat, test_graph_pool, test_one_hot_labels, test_num_features_nonzero = get_batch_data(batch_test_graphs)
                acc_output.append(eval_step(test_Adj_block, test_X_concat, test_graph_pool, test_one_hot_labels, test_num_features_nonzero))

            fold_test_acc = sum(acc_output) / float(len(test_graphs))*100.0

            print('epoch ' + str(epoch) + ' fold ' + str(args.fold_idx) + ' acc ' + str(fold_test_acc) + '%')

            write_acc.write('epoch ' + str(epoch) + ' fold ' + str(args.fold_idx) + ' acc ' + str(fold_test_acc) + '%\n')

        write_acc.close()

