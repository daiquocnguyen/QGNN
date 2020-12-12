import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sampled_softmax import  *
from q4gnn import *

class UnSupQGNN(nn.Module):
    def __init__(self, feature_dim_size, hidden_size, num_GNN_layers, sampled_num, vocab_size, dropout, device):
        super(UnSupQGNN, self).__init__()
        self.feature_dim_size = feature_dim_size
        self.hidden_size = hidden_size
        self.num_GNN_layers = num_GNN_layers
        self.sampled_num = sampled_num
        self.vocab_size = vocab_size
        self.device = device
        #
        self.q4gnnlayers = torch.nn.ModuleList()
        for layer in range(self.num_GNN_layers):
            if layer == 0:
                self.q4gnnlayers.append(Q4GNNLayer(self.feature_dim_size, self.hidden_size, dropout=dropout))
            else:
                self.q4gnnlayers.append(Q4GNNLayer(self.hidden_size, self.hidden_size, dropout=dropout))
        #
        self.dropouts = nn.Dropout(dropout)
        self.sampled_softmax = SampledSoftmax(self.vocab_size, self.sampled_num, self.hidden_size*self.num_GNN_layers, self.device)

    def forward(self, Adj_block, X_concat, idx_nodes):
        output_vectors = []  # should test output_vectors = [X_concat] --> self.feature_dim_size+self.hidden_size*self.num_GNN_layers
        input = X_concat
        for layer in range(self.num_GNN_layers):
            #
            input = self.q4gnnlayers[layer](input.double(), Adj_block, True)
            output_vectors.append(input)

        output_vectors = torch.cat(output_vectors, dim=1)
        output_vectors = self.dropouts(output_vectors)

        logits = self.sampled_softmax(output_vectors, idx_nodes)

        return logits

