from __future__ import division
from __future__ import print_function

import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

np.random.seed(123)
torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)

from utils_node_cls import *
from q4gnn import *

# Parameters
# ==================================================
parser = ArgumentParser("Q4GNN", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--dataset", default="cora", help="Name of the dataset.")
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning_rate', type=float, default=0.05, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden_size', type=int, default=16, help='Hidden_size//4 = number of quaternion units within each hidden layer.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--fold', type=int, default=2, help='The fold index. 0-9.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
args = parser.parse_args()

# Load data
adj, features, labels, y_train, y_val, y_test, train_mask, val_mask, test_mask = \
    load_data_new_split(args.dataset, '../splits/' + args.dataset + '_split_0.6_0.2_'+ str(args.fold) + '.npz')
labels = torch.from_numpy(labels).to(device)
labels = torch.where(labels==1)[1]
idx_train = torch.where(torch.from_numpy(train_mask)==True)
idx_val = torch.where(torch.from_numpy(val_mask)==True)
idx_test = torch.where(torch.from_numpy(test_mask)==True)

"""Convert a scipy sparse matrix to a torch sparse tensor."""
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to(device)

""" quaternion preprocess for feature vectors """
def quaternion_preprocess_features(features):
    """Row-normalize feature matrix"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    features = features.todense()
    features = np.tile(features, 4) # A + Ai + Aj + Ak
    return torch.from_numpy(features).to(device)

# Some preprocessing
features = quaternion_preprocess_features(features)
adj = normalize_adj(adj + sp.eye(adj.shape[0])).tocoo()
adj = sparse_mx_to_torch_sparse_tensor(adj)

# Accuracy
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

'''Quaternion graph neural network! 2-layer Q4GNN!'''
class Q4GNN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(Q4GNN, self).__init__()
        self.q4gnn1 = Q4GNNLayer(nfeat, nhid, dropout=dropout)
        self.q4gnn2 = Q4GNNLayer(nhid, nclass, dropout=dropout, quaternion_ff=False, act=False)

    def forward(self, x, adj):
        x = self.q4gnn1(x, adj)
        x = self.q4gnn2(x, adj)
        return F.log_softmax(x, dim=1)

# Model and optimizer
model = Q4GNN(nfeat=features.size(1), nhid=args.hidden_size, nclass=y_train.shape[1], dropout=args.dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

"""Adapted from https://github.com/tkipf/pygcn/blob/master/pygcn/train.py"""
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately, deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()


