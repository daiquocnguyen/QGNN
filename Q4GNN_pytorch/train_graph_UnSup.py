#! /usr/bin/env python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(123)

import numpy as np
np.random.seed(123)
import time

from model_graph_UnSup import *
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.sparse import coo_matrix
from utils_graph_cls import *
from sklearn.linear_model import LogisticRegression
import statistics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

# Parameters
# ==================================================
parser = ArgumentParser("QGNN", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--run_folder", default="../", help="")
parser.add_argument("--dataset", default="PTC", help="Name of the dataset.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate")
parser.add_argument("--batch_size", default=4, type=int, help="Batch Size")
parser.add_argument("--num_epochs", default=100, type=int, help="Number of training epochs")
parser.add_argument("--model_name", default='PTC', help="")
parser.add_argument("--dropout", default=0.5, type=float, help="Dropout")
parser.add_argument("--num_GNN_layers", default=2, type=int, help="Number of hidden layers")
parser.add_argument("--hidden_size", default=256, type=int, help="Hidden_size//4 = number of quaternion units within each hidden layer")
parser.add_argument('--sampled_num', default=512, type=int, help='')
args = parser.parse_args()

print(args)

# Load data
print("Loading data...")

degree_as_tag = False
if args.dataset == 'COLLAB' or args.dataset == 'IMDBBINARY' or args.dataset == 'IMDBMULTI' or\
    args.dataset == 'REDDITBINARY' or args.dataset == 'REDDITMULTI5K' or args.dataset == 'REDDITMULTI12K':
    degree_as_tag = True

graphs, num_classes = load_graph_data(args.dataset, degree_as_tag)
feature_dim_size = graphs[0].node_features.shape[1]
graph_labels = np.array([graph.label for graph in graphs])

def get_Adj_matrix(batch_graph):
    edge_mat_list = []
    start_idx = [0]
    for i, graph in enumerate(batch_graph):
        start_idx.append(start_idx[i] + len(graph.g))
        edge_mat_list.append(graph.edge_mat + start_idx[i])

    Adj_block_idx = np.concatenate(edge_mat_list, 1)
    Adj_block_elem = np.ones(Adj_block_idx.shape[1])

    # self-loop
    num_node = start_idx[-1]
    self_loop_edge = np.array([range(num_node), range(num_node)])
    elem = np.ones(num_node)
    Adj_block_idx = np.concatenate([Adj_block_idx, self_loop_edge], 1)
    Adj_block_elem = np.concatenate([Adj_block_elem, elem], 0)

    Adj_block_idx = torch.from_numpy(Adj_block_idx)
    Adj_block_elem = torch.from_numpy(Adj_block_elem)

    Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1], start_idx[-1]]))

    return Adj_block.to(device)

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

    elem = torch.FloatTensor(elem)
    idx = torch.LongTensor(idx).transpose(0, 1)
    graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))

    return graph_pool.to(device)
#
graph_pool = get_graphpool(graphs)
graph_indices = graph_pool._indices()[0]
vocab_size = graph_pool.size()[1]

def get_idx_nodes(selected_graph_idx):
    idx_nodes = [torch.where(graph_indices==i)[0] for i in selected_graph_idx]
    idx_nodes = torch.cat(idx_nodes)
    return idx_nodes.to(device)

def get_batch_data(selected_idx):
    batch_graph = [graphs[idx] for idx in selected_idx]
    # features
    X_concat = np.concatenate([graph.node_features for graph in batch_graph], 0)
    # A + Ai + Aj + Ak
    X_concat = np.tile(X_concat, 4)  # feature_dim_size*4
    X_concat = torch.from_numpy(X_concat).to(device)
    # adj
    Adj_block = get_Adj_matrix(batch_graph)
    #
    idx_nodes = get_idx_nodes(selected_idx)

    return Adj_block, X_concat, idx_nodes

class Batch_Loader(object):
    def __call__(self):
        selected_idx = np.random.permutation(len(graphs))[:args.batch_size]
        Adj_block, X_concat, idx_nodes = get_batch_data(selected_idx)
        return Adj_block, X_concat, idx_nodes
batch_nodes = Batch_Loader()

print("Loading data... finished!")
#===================================
model = UnSupQGNN(feature_dim_size=feature_dim_size*4,  # A + Ai + Aj + Ak
                hidden_size=args.hidden_size,
                num_GNN_layers=args.num_GNN_layers,
                vocab_size=graph_pool.shape[1],
                sampled_num=args.sampled_num,
                dropout=args.dropout,
                device=device).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
num_batches_per_epoch = int((len(graphs) - 1) / args.batch_size) + 1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_batches_per_epoch, gamma=0.1)

def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    for _ in range(num_batches_per_epoch):
        Adj_block, X_concat, idx_nodes = batch_nodes()
        optimizer.zero_grad()
        logits = model(Adj_block, X_concat, idx_nodes)
        loss = torch.sum(logits)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()

    return total_loss

def evaluate():
    model.eval() # Turn on the evaluation mode
    with torch.no_grad():
        # evaluating
        node_embeddings = model.sampled_softmax.weight
        graph_embeddings = torch.spmm(graph_pool, node_embeddings).data.cpu().numpy()
        acc_10folds = []
        for fold_idx in range(10):
            train_idx, test_idx = separate_data_idx(graphs, fold_idx)
            train_graph_embeddings = graph_embeddings[train_idx]
            test_graph_embeddings = graph_embeddings[test_idx]
            train_labels = graph_labels[train_idx]
            test_labels = graph_labels[test_idx]

            cls = LogisticRegression(solver="liblinear", tol=0.001)
            cls.fit(train_graph_embeddings, train_labels)
            ACC = cls.score(test_graph_embeddings, test_labels)
            acc_10folds.append(ACC)
            print('epoch ', epoch, ' fold ', fold_idx, ' acc ', ACC)

        mean_10folds = statistics.mean(acc_10folds)
        std_10folds = statistics.stdev(acc_10folds)
        # print('epoch ', epoch, ' mean: ', str(mean_10folds), ' std: ', str(std_10folds))

    return mean_10folds, std_10folds

"""main process"""
import os
out_dir = os.path.abspath(os.path.join(args.run_folder, "../runs_pytorch_Q4GNN_UnSup", args.model_name))
print("Writing to {}\n".format(out_dir))
# Checkpoint directory
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
write_acc = open(checkpoint_prefix + '_acc.txt', 'w')

cost_loss = []
for epoch in range(1, args.num_epochs + 1):
    epoch_start_time = time.time()
    train_loss = train()
    cost_loss.append(train_loss)
    mean_10folds, std_10folds = evaluate()
    print('| epoch {:3d} | time: {:5.2f}s | loss {:5.2f} | mean {:5.2f} | std {:5.2f} | '.format(
                epoch, (time.time() - epoch_start_time), train_loss, mean_10folds*100, std_10folds*100))

    if epoch > 5 and cost_loss[-1] > np.mean(cost_loss[-6:-1]):
        scheduler.step()

    write_acc.write('epoch ' + str(epoch) + ' mean: ' + str(mean_10folds*100) + ' std: ' + str(std_10folds*100) + '\n')

write_acc.close()