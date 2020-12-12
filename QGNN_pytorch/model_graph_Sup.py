import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from q4gnn import *

class SupQGNN(nn.Module):
    def __init__(self, feature_dim_size, hidden_size, num_GNN_layers, num_classes, dropout):
        super(SupQGNN, self).__init__()
        self.feature_dim_size = feature_dim_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_GNN_layers = num_GNN_layers
        #
        self.q4gnnlayers = torch.nn.ModuleList()
        for layer in range(self.num_GNN_layers):
            if layer == 0:
                self.q4gnnlayers.append(QGNNLayer(self.feature_dim_size, self.hidden_size, dropout=dropout))
            else:
                self.q4gnnlayers.append(QGNNLayer(self.hidden_size, self.hidden_size, dropout=dropout))
        #
        self.predictions = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        # self.predictions.append(nn.Linear(feature_dim_size, num_classes)) # For including feature vectors to predict graph labels???
        for _ in range(self.num_GNN_layers):
            self.predictions.append(nn.Linear(self.hidden_size, self.num_classes))
            self.dropouts.append(nn.Dropout(dropout))

    def forward(self, Adj_block, X_concat, graph_pool):
        prediction_scores = 0
        input = X_concat
        for layer in range(self.num_GNN_layers):
            input = self.q4gnnlayers[layer](input.double(), Adj_block, True)
            #sum pooling
            graph_embeddings = torch.spmm(graph_pool, input.float())
            graph_embeddings = self.dropouts[layer](graph_embeddings)
            # Produce the final scores
            prediction_scores += self.predictions[layer](graph_embeddings)

        return prediction_scores

def label_smoothing(true_labels: torch.Tensor, classes: int, smoothing=0.1):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist