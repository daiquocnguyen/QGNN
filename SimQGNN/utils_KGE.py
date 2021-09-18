from load_data import Data
import numpy as np
import time
import torch
from collections import defaultdict
import argparse
import scipy.sparse as sp
from collections import Counter
import itertools
from scipy import sparse

torch.manual_seed(1337)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)
np.random.seed(1337)

def normalize_sparse(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# The Adj matrix
def get_adj_matrix(data, entity_idxs):
    row_indxs = []
    col_indxs = []
    dat_values = []
    for hrt in data:
        row_indxs.append(entity_idxs[hrt[0]])
        col_indxs.append(entity_idxs[hrt[2]])
        dat_values.append(1)
        row_indxs.append(entity_idxs[hrt[2]])
        col_indxs.append(entity_idxs[hrt[0]])
        dat_values.append(1)
    edge_mat = sparse.csr_matrix((dat_values, (row_indxs, col_indxs)))
    edge_mat[edge_mat > 1] = 1 # to deal with duplicate indexes
    # adding self-loop
    adj = edge_mat + sparse.eye(edge_mat.shape[0], format="csr")
    # print(adj)
    adj = normalize_sparse(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj
