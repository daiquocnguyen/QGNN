import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

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
    def __init__(self, feature_dim_size, hidden_size, num_classes, dropout, num_steps=1, act=nn.functional.tanh):
        super(GatedQGNN, self).__init__()
        self.num_steps = num_steps
        self.act = act
        self.dropout_encode = nn.Dropout(dropout)
        self.emb_encode = Parameter(torch.FloatTensor(feature_dim_size//4, hidden_size))
        self.z0 = Parameter(torch.FloatTensor(hidden_size//4, hidden_size))
        self.z1 = Parameter(torch.FloatTensor(hidden_size//4, hidden_size))
        self.r0 = Parameter(torch.FloatTensor(hidden_size//4, hidden_size))
        self.r1 = Parameter(torch.FloatTensor(hidden_size//4, hidden_size))
        self.h0 = Parameter(torch.FloatTensor(hidden_size//4, hidden_size))
        self.h1 = Parameter(torch.FloatTensor(hidden_size//4, hidden_size))
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

    def forward(self, inputs, adj):
        x = inputs
        x = self.dropout_encode(x)
        x = torch.matmul(x, make_quaternion_mul(self.emb_encode))
        for idx_layer in range(self.num_steps):
            x = self.gatedGNN(x, adj)
        return x
    
'''Quaternion graph neural networks! QGNN layer for other downstream tasks!'''
class QGNNLayer_v2(Module):
    def __init__(self, in_features, out_features, act=torch.tanh):
        super(QGNNLayer_v2, self).__init__()
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

'''Quaternion graph neural networks! QGNN layer for node and graph classification tasks!'''
class QGNNLayer(Module):
    def __init__(self, in_features, out_features, dropout, quaternion_ff=True, act=F.relu):
        super(QGNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quaternion_ff = quaternion_ff 
        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.bn = torch.nn.BatchNorm1d(out_features)
        #
        if self.quaternion_ff:
            self.weight = Parameter(torch.FloatTensor(self.in_features//4, self.out_features))
        else:
            self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, double_type_used_in_graph=False):

        x = self.dropout(input) # Current Pytorch 1.5.0 doesn't support Dropout for sparse matrix

        if self.quaternion_ff:
            hamilton = make_quaternion_mul(self.weight)
            if double_type_used_in_graph:  # to deal with scalar type between node and graph classification tasks
                hamilton = hamilton.double()

            support = torch.mm(x, hamilton)  # Hamilton product, quaternion multiplication!
        else:
            support = torch.mm(x, self.weight)

        if double_type_used_in_graph: #to deal with scalar type between node and graph classification tasks, caused by pre-defined feature inputs
            support = support.double()

        output = torch.spmm(adj, support)

        # output = self.bn(output) # should tune whether using BatchNorm or Dropout

        return self.act(output)

'''Dual quaternion multiplication'''
def dual_quaternion_mul(A, B, input):
    '''(A, B) * (C, D) = (A * C, A * D + B * C)'''
    dim = input.size(1) // 2
    C, D = torch.split(input, [dim, dim], dim=1)
    A_hamilton = make_quaternion_mul(A)
    B_hamilton = make_quaternion_mul(B)
    AC = torch.mm(C, A_hamilton)
    AD = torch.mm(D, A_hamilton)
    BC = torch.mm(C, B_hamilton)
    AD_plus_BC = AD + BC
    return torch.cat([AC, AD_plus_BC], dim=1)

''' Dual Quaternion Graph Neural Networks! https://arxiv.org/abs/2104.07396 '''
class DQGNN_layer(Module):
    def __init__(self, in_features, out_features, act=torch.tanh):
        super(DQGNN_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = act
        #
        self.A = Parameter(torch.FloatTensor(self.in_features // 8, self.out_features // 2)) # (A, B) = A + eB, e^2 = 0
        self.B = Parameter(torch.FloatTensor(self.in_features // 8, self.out_features // 2))

        self.reset_parameters()
        self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.A.size(0) + self.A.size(1)))
        self.A.data.uniform_(-stdv, stdv)
        self.B.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = dual_quaternion_mul(self.A, self.B, input)
        output = torch.spmm(adj, support)
        output = self.bn(output)
        return self.act(output)

''' Simplifying Quaternion Graph Neural Networks! following SGC https://arxiv.org/abs/1902.07153'''
class SQGNN_layer(Module):
    def __init__(self, in_features, out_features, step_k=1):
        super(SQGNN_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.step_k = step_k
        self.weight = Parameter(torch.FloatTensor(self.in_features // 4, self.out_features))
        self.reset_parameters()
        #self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        hamilton = make_quaternion_mul(self.weight)
        new_input = torch.spmm(adj, input)
        if self.step_k > 1:
            for _ in range(self.step_k-1):
                new_input = torch.spmm(adj, new_input)
        output = torch.mm(new_input, hamilton)  # Hamilton product, quaternion multiplication!
        #output = self.bn(output)
        return output
