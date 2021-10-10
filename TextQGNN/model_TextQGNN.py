import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import TransformerEncoder, TransformerEncoderLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""@Dai Quoc Nguyen"""
'''Make a Hamilton matrix for quaternion linear transformations'''
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

"""Gated Quaternion GNNs"""
class GatedQGNN(nn.Module):
    def __init__(self, feature_dim_size, hidden_size, num_steps, num_classes, dropout, act=torch.relu):
        super(GatedQGNN, self).__init__()
        self.num_steps = num_steps
        self.dropout_encode = nn.Dropout(dropout)
        self.soft_att = nn.Linear(hidden_size, 1)
        self.act = act
        self.prediction = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

        self.emb_encode = Parameter(torch.FloatTensor(feature_dim_size//4, hidden_size))
        self.z0 = Parameter(torch.FloatTensor(hidden_size//4, hidden_size))
        self.z1 = Parameter(torch.FloatTensor(hidden_size//4, hidden_size))
        self.r0 = Parameter(torch.FloatTensor(hidden_size//4, hidden_size))
        self.r1 = Parameter(torch.FloatTensor(hidden_size//4, hidden_size))
        self.h0 = Parameter(torch.FloatTensor(hidden_size//4, hidden_size))
        self.h1 = Parameter(torch.FloatTensor(hidden_size//4, hidden_size))
        self.ln = Parameter(torch.FloatTensor(hidden_size//4, hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.emb_encode.size(0) + self.emb_encode.size(1)))
        self.emb_encode.data.uniform_(-stdv, stdv)
        stdv = math.sqrt(6.0 / (self.z0.size(0) + self.z0.size(1)))
        self.z0.data.uniform_(-stdv, stdv)
        self.z1.data.uniform_(-stdv, stdv)
        self.r0.data.uniform_(-stdv, stdv)
        self.r1.data.uniform_(-stdv, stdv)
        self.h0.data.uniform_(-stdv, stdv)
        self.h1.data.uniform_(-stdv, stdv)
        self.ln.data.uniform_(-stdv, stdv)

    def gatedGNN(self, x, adj):
        a = torch.matmul(adj, x)
        # update gate
        z0 = torch.matmul(a, make_quaternion_mul(self.z0))
        z1 = torch.matmul(x, make_quaternion_mul(self.z1))
        z = torch.sigmoid(z0 + z1) # missing bias
        # reset gate
        r0 = torch.matmul(a, make_quaternion_mul(self.r0))
        r1 = torch.matmul(x, make_quaternion_mul(self.r1))
        r = torch.sigmoid(r0 + r1)
        # update embeddings
        h0 = torch.matmul(a, make_quaternion_mul(self.h0))
        h1 = torch.matmul(r*x, make_quaternion_mul(self.h1))
        h = self.act(h0+h1)

        return h * z + x * (1 - z)

    def forward(self, inputs, adj, mask):
        x = inputs
        x = self.dropout_encode(x)
        x = torch.matmul(x, make_quaternion_mul(self.emb_encode))
        x = x * mask
        for idx_layer in range(self.num_steps):
            x = self.gatedGNN(x, adj) * mask
        # soft attention
        soft_att = torch.sigmoid(self.soft_att(x))
        x = self.act(torch.matmul(x, make_quaternion_mul(self.ln)))
        x = soft_att * x * mask
        # sum and max pooling
        graph_embeddings = torch.sum(x, 1) * torch.amax(x, 1)
        graph_embeddings = self.dropout(graph_embeddings)
        prediction_scores = self.prediction(graph_embeddings)

        return prediction_scores


"""https://arxiv.org/abs/1511.05493"""
class GatedGNN(nn.Module):
    def __init__(self, feature_dim_size, hidden_size, num_steps, num_classes, dropout, act=torch.relu):
        super(GatedGNN, self).__init__()
        self.num_steps = num_steps
        self.act = act
        self.emb_encode = nn.Linear(feature_dim_size, hidden_size)
        self.dropout_encode = nn.Dropout(dropout)
        self.z0 = nn.Linear(hidden_size, hidden_size)
        self.z1 = nn.Linear(hidden_size, hidden_size)
        self.r0 = nn.Linear(hidden_size, hidden_size)
        self.r1 = nn.Linear(hidden_size, hidden_size)
        self.h0 = nn.Linear(hidden_size, hidden_size)
        self.h1 = nn.Linear(hidden_size, hidden_size)
        self.soft_att = nn.Linear(hidden_size, 1)
        self.ln = nn.Linear(hidden_size, hidden_size)
        self.prediction = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def gatedGNN(self, x, adj):
        a = torch.matmul(adj, x)
        # update gate
        z0 = self.z0(a)
        z1 = self.z1(x)
        z = torch.sigmoid(z0 + z1)
        # reset gate
        r = torch.sigmoid(self.r0(a) + self.r1(x))
        # update embeddings
        h = self.act(self.h0(a) + self.h1(r * x))

        return h * z + x * (1 - z)

    def forward(self, inputs, adj, mask):
        x = inputs
        x = self.dropout_encode(x)
        x = self.emb_encode(x)
        x = x * mask
        for idx_layer in range(self.num_steps):
            x = self.gatedGNN(x, adj) * mask
        # soft attention
        soft_att = torch.sigmoid(self.soft_att(x))
        x = self.act(self.ln(x))
        x = soft_att * x * mask
        # sum and max pooling
        graph_embeddings = torch.sum(x, 1) * torch.amax(x, 1)
        graph_embeddings = self.dropout(graph_embeddings)
        prediction_scores = self.prediction(graph_embeddings)

        return prediction_scores
