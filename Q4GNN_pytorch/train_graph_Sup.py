#! /usr/bin/env python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(123)

import numpy as np
np.random.seed(123)
import time

from model_graph_Sup import *
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.sparse import coo_matrix
from utils_graph_cls import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

# Parameters
# ==================================================

parser = ArgumentParser("Q4GNN", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

parser.add_argument("--run_folder", default="../", help="")
parser.add_argument("--dataset", default="MUTAG", help="Name of the dataset.")
parser.add_argument("--learning_rate", default=0.0005, type=float, help="Learning rate")
parser.add_argument("--batch_size", default=4, type=int, help="Batch Size")
parser.add_argument("--num_epochs", default=100, type=int, help="Number of training epochs")
parser.add_argument("--saveStep", default=1, type=int, help="")
parser.add_argument("--model_name", default='MUTAG', help="")
parser.add_argument("--dropout", default=0.5, type=float, help="Dropout")
parser.add_argument("--num_GNN_layers", default=2, type=int, help="Number of hidden layers")
parser.add_argument("--hidden_size", default=16, type=int, help="Hidden_size//4 = number of quaternion units within each hidden layer.")
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

def get_batch_data(batch_graph):
    # features
    X_concat = np.concatenate([graph.node_features for graph in batch_graph], 0)
    # A + Ai + Aj + Ak
    X_concat = np.tile(X_concat, 4)  # feature_dim_size*4
    X_concat = torch.from_numpy(X_concat).to(device)
    # adj
    Adj_block = get_Adj_matrix(batch_graph)
    # graph pool
    graph_pool = get_graphpool(batch_graph)

    graph_labels = np.array([graph.label for graph in batch_graph])
    graph_labels = torch.from_numpy(graph_labels).to(device)

    return Adj_block, X_concat, graph_pool, graph_labels

class Batch_Loader(object):
    def __call__(self):
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]
        batch_graph = [train_graphs[idx] for idx in selected_idx]
        Adj_block, X_concat, graph_pool, graph_labels = get_batch_data(batch_graph)
        return Adj_block, X_concat, graph_pool, graph_labels
batch_nodes = Batch_Loader()

Adj_block, X_concat, graph_pool, graph_labels = batch_nodes()

print("Loading data... finished!")

model = SupQ4GNN(feature_dim_size=feature_dim_size*4,  # A + Ai + Aj + Ak
                hidden_size=args.hidden_size, dropout=args.dropout,
                num_GNN_layers=args.num_GNN_layers,
                num_classes=num_classes).to(device)

def cross_entropy(pred, soft_targets): # use nn.CrossEntropyLoss if not using soft labels in Line 159
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
num_batches_per_epoch = int((len(train_graphs) - 1) / args.batch_size) + 1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=num_batches_per_epoch, gamma=0.1)

def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    for _ in range(num_batches_per_epoch):
        Adj_block, X_concat, graph_pool, graph_labels = batch_nodes()
        graph_labels = label_smoothing(graph_labels, num_classes)
        optimizer.zero_grad()
        prediction_scores = model(Adj_block, X_concat, graph_pool)
        # loss = criterion(prediction_scores, graph_labels)
        loss = cross_entropy(prediction_scores, graph_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # prevent the exploding gradient problem
        optimizer.step()
        total_loss += loss.item()

    return total_loss

def evaluate():
    model.eval() # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        # evaluating
        prediction_output = []
        idx = np.arange(len(test_graphs))
        for i in range(0, len(test_graphs), args.batch_size):
            sampled_idx = idx[i:i + args.batch_size]
            if len(sampled_idx) == 0:
                continue
            batch_test_graphs = [test_graphs[j] for j in sampled_idx]
            Adj_block, X_concat, graph_pool, graph_label = get_batch_data(batch_test_graphs)
            prediction_scores = model(Adj_block, X_concat, graph_pool).detach()
            prediction_output.append(prediction_scores)
    prediction_output = torch.cat(prediction_output, 0)
    predictions = prediction_output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = predictions.eq(labels.view_as(predictions)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    return acc_test

"""main process"""
import os
out_dir = os.path.abspath(os.path.join(args.run_folder, "../runs_pytorch_Q4GNN_Sup", args.model_name))
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
    acc_test = evaluate()
    print('| epoch {:3d} | time: {:5.2f}s | loss {:5.2f} | test acc {:5.2f} | '.format(
                epoch, (time.time() - epoch_start_time), train_loss, acc_test*100))

    if epoch > 5 and cost_loss[-1] > np.mean(cost_loss[-6:-1]):
        scheduler.step()

    write_acc.write('epoch ' + str(epoch) + ' fold ' + str(args.fold_idx) + ' acc ' + str(acc_test*100) + '%\n')

write_acc.close()
