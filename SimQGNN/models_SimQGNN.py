import numpy as np
import torch
from torch.nn.init import xavier_normal_
from torch import empty, matmul, tensor
import torch
from torch.cuda import empty_cache
from torch.nn import Parameter, Module
from torch.nn.functional import normalize
from tqdm.autonotebook import tqdm
import torch.nn.functional as F
import math
import numpy as np


"""Run GNNs directly on the single undirected graph of entities"""
class SimQGNN(torch.nn.Module):
    def __init__(self, encoder, decoder, emb_dim, hid_dim, adj, n_entities, n_relations, num_layers=1):
        super(SimQGNN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.adj = adj
        self.num_layers = num_layers
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.entity_embeddings = torch.nn.Embedding(n_entities, emb_dim)
        self.relation_embeddings = torch.nn.Embedding(n_relations, hid_dim)
        torch.nn.init.xavier_normal_(self.entity_embeddings.weight.data)
        torch.nn.init.xavier_normal_(self.relation_embeddings.weight.data)
        self.lst_gnn = torch.nn.ModuleList()

        if self.encoder.lower() == "gcn":
            print("SimRGCN: using GCN encoder")
            gnn_mode = GraphConvolution
        else: # self.encoder.lower() == "qgnn":
            print("SimQGNN: using QGNN encoder")
            gnn_mode = QGNN_layer

        for _layer in range(self.num_layers):
            if _layer == 0:
                self.lst_gnn.append(gnn_mode(emb_dim, hid_dim, act=torch.tanh))
            else:
                self.lst_gnn.append(gnn_mode(hid_dim, hid_dim, act=torch.tanh))

        self.bn1 = torch.nn.BatchNorm1d(hid_dim)
        self.hidden_dropout2 = torch.nn.Dropout()
        self.loss = torch.nn.BCELoss()

    def score(self, e1_idx, r_idx, X):
        h = X[e1_idx]
        r = self.relation_embeddings(r_idx)
        # DistMult decoder
        hr = h * r
        hr = self.bn1(hr)
        hr = self.hidden_dropout2(hr)
        hrt = torch.mm(hr, X.t())  # following the 1-N scoring strategy in ConvE for faster computation; thus, predicting h given (?, r, t) is equal to predicting h given (t, r-1, ?).
        return hrt

    def forward(self, e1_idx, r_idx, lst_ents):
        X = self.entity_embeddings(lst_ents)
        # consider the last hidden layer
        for _layer in range(self.num_layers):
            X = self.lst_gnn[_layer](X, self.adj)
        hrt = self.score(e1_idx, r_idx, X)
        pred = torch.sigmoid(hrt)
        return pred


''' A re-implementation of DistMult, following the 1-N scoring strategy '''
class DistMult(torch.nn.Module):
   def __init__(self, emb_dim, n_entities, n_relations):
        super(DistMult, self).__init__()
        self.entity_embeddings = torch.nn.Embedding(n_entities, emb_dim)
        self.relation_embeddings = torch.nn.Embedding(n_relations, emb_dim)
        torch.nn.init.xavier_normal_(self.entity_embeddings.weight.data)
        torch.nn.init.xavier_normal_(self.relation_embeddings.weight.data)
        self.bn1 = torch.nn.BatchNorm1d(emb_dim)
        self.hidden_dropout2 = torch.nn.Dropout()
        self.loss = torch.nn.BCELoss()

   def forward(self, e1_idx, r_idx,  lst_ents):
        X = self.entity_embeddings(lst_ents)
        h = self.entity_embeddings(e1_idx)
        r = self.relation_embeddings(r_idx)
        hr = h * r
        hr = self.bn1(hr)
        hr = self.hidden_dropout2(hr)
        hrt = torch.mm(hr, X.t()) # following the 1-N scoring strategy in ConvE
        pred = torch.sigmoid(hrt)
        return pred

# Quaternion operations
def normalization(quaternion, split_dim=1):  # vectorized quaternion bs x 4dim
    size = quaternion.size(split_dim) // 4
    quaternion = quaternion.reshape(-1, 4, size)  # bs x 4 x dim
    quaternion = quaternion / torch.sqrt(torch.sum(quaternion ** 2, 1, True))  # quaternion / norm
    quaternion = quaternion.reshape(-1, 4 * size)
    return quaternion

def make_wise_quaternion(quaternion):  # for vector * vector quaternion multiplication
    if len(quaternion.size()) == 1:
        quaternion = quaternion.unsqueeze(0)
    size = quaternion.size(1) // 4
    r, i, j, k = torch.split(quaternion, size, dim=1)
    r2 = torch.cat([r, -i, -j, -k], dim=1)  # 0, 1, 2, 3 --> bs x 4dim
    i2 = torch.cat([i, r, -k, j], dim=1)  # 1, 0, 3, 2
    j2 = torch.cat([j, k, r, -i], dim=1)  # 2, 3, 0, 1
    k2 = torch.cat([k, -j, i, r], dim=1)  # 3, 2, 1, 0
    return r2, i2, j2, k2

def get_quaternion_wise_mul(quaternion):
    size = quaternion.size(1) // 4
    quaternion = quaternion.view(-1, 4, size)
    quaternion = torch.sum(quaternion, 1)
    return quaternion

def vec_vec_quaternion_multiplication(q, p):  # vector * vector
    q_r, q_i, q_j, q_k = make_wise_quaternion(q)  # bs x 4dim

    qp_r = get_quaternion_wise_mul(q_r * p)  # qrpr−qipi−qjpj−qkpk
    qp_i = get_quaternion_wise_mul(q_i * p)  # qipr+qrpi−qkpj+qjpk
    qp_j = get_quaternion_wise_mul(q_j * p)  # qjpr+qkpi+qrpj−qipk
    qp_k = get_quaternion_wise_mul(q_k * p)  # qkpr−qjpi+qipj+qrpk

    return torch.cat([qp_r, qp_i, qp_j, qp_k], dim=1)

def regularization(quaternion):  # vectorized quaternion bs x 4dim
    size = quaternion.size(1) // 4
    r, i, j, k = torch.split(quaternion, size, dim=1)
    return torch.mean(r ** 2) + torch.mean(i ** 2) + torch.mean(j ** 2) + torch.mean(k ** 2)

def make_quaternion_mul(kernel):
    """" The constructed 'hamilton' W is a modified version of the quaternion representation,
    thus doing tf.matmul(Input,W) is equivalent to W * Inputs. """
    dim = kernel.size(1)//4
    r, i, j, k = torch.split(kernel, [dim, dim, dim, dim], dim=1)
    r2 = torch.cat([r, -i, -j, -k], dim=0)  # 0, 1, 2, 3
    i2 = torch.cat([i, r, -k, j], dim=0)  # 1, 0, 3, 2
    j2 = torch.cat([j, k, r, -i], dim=0)  # 2, 3, 0, 1
    k2 = torch.cat([k, -j, i, r], dim=0)  # 3, 2, 1, 0
    hamilton = torch.cat([r2, i2, j2, k2], dim=1)
    assert kernel.size(1) == hamilton.size(1)
    return hamilton

'''Quaternion graph neural networks! QGNN layer for other downstream tasks!'''
class QGNN_layer(Module):
    def __init__(self, in_features, out_features, act=torch.tanh):
        super(QGNN_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = act
        self.weight = Parameter(torch.FloatTensor(self.in_features//4, self.out_features))
        self.reset_parameters()
        self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        hamilton = make_quaternion_mul(self.weight)
        support = torch.mm(input, hamilton)  # Hamilton product, quaternion multiplication!
        output = torch.spmm(adj, support)
        output = self.bn(output)  # using act torch.tanh with BatchNorm can produce competitive results
        return self.act(output)

""" GCN layer, similar to https://arxiv.org/abs/1609.02907 """
class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features, act=torch.tanh, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.act = act
        self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        output = self.bn(output)
        return self.act(output)
