"""
Train a GNN for Graph classification.

Luiz Felipe da Silveira Coelho - luizfelipe.coelho@smt.ufrj.br
apr 18, 2021
"""

import pandas as pd
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader
from dgl.data import DGLDataset
from dgl.nn import GraphConv

import matplotlib.pyplot as plt
import networkx as nx


class SyntheticDataset(DGLDataset):
    """
    Synthetic dataset generated from my edge list and properties list.
    """

    def __init__(self):
        super().__init__(name='synthetic')

    def process(self):
        """Process .csv files."""
        edges = pd.read_csv('./data_set/edge_list_coerrection.csv')
        properties = pd.read_csv('./data_set/properties.csv')
        self.graphs = []
        self.labels = []

        # Create a graph for each graph ID from the edges table.
        # First process the properties table into two dictionaries with
        # graph IDs as keys.
        # The label and number of nodes are values.
        label_dict = {}
        num_nodes_dict = {}
        for _, row in properties.iterrows():
            label_dict[row['graph_id']] = row['label']
            num_nodes_dict[row['graph_id']] = row['num_nodes']

        # For the edges, first group the table by graph IDs.
        edges_group = edges.groupby('graph_id')

        # For each graph ID...
        for graph_id in edges_group.groups:
            # Find the edges as well as the number of nodes and its
            # label.
            edges_of_id = edges_group.get_group(graph_id)
            src = edges_of_id['src'].to_numpy()
            dst = edges_of_id['dst'].to_numpy()
            num_nodes = num_nodes_dict[graph_id]
            label = label_dict[graph_id]

            # Create a graph and add it to the list of graphs and
            # labels.
            graph = dgl.graph((src, dst), num_nodes=num_nodes)
            self.graphs.append(graph)
            self.labels.append(label)

        # Convert the label list to tenso for saving.
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


class GCN(nn.Module):
    """Graph Convolutional Network model."""

    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)
        self.classify = nn.Linear(h_feats, num_classes)

    def forward(self, graph):
        """Compute graph convolution."""
        # Use the node degree as the initial node feature. For undirected
        # graphs, the in-degree is the same as the out-degree.
        h = graph.in_degrees().view(-1, 1).float()
        # Perform a graph convolution and activation function
        h = F.relu(self.conv1(graph, h))
        h = F.relu(self.conv2(graph, h))
        graph.ndata['h'] = h
        # Calculate graph representation by averaging all the node
        # representations.
        hg = dgl.mean_nodes(graph, 'h')
        return self.classify(hg)


if __name__ == '__main__':
    # Load the synthetic dataset:
    DATASET = SyntheticDataset()

    # Define data loader:
    NUM_EXAMPLES = len(DATASET)
    NUM_TRAIN = int(NUM_EXAMPLES * .8)

    TRAIN_SAMPLER = SubsetRandomSampler(torch.arange(NUM_TRAIN))
    TEST_SAMPLER = SubsetRandomSampler(torch.arange(NUM_TRAIN, NUM_EXAMPLES))

    TRAIN_DATALOADER = GraphDataLoader(DATASET, sampler=TRAIN_SAMPLER,
                                       batch_size=5, drop_last=False)
    TEST_DATALOADER = GraphDataLoader(DATASET, sampler=TEST_SAMPLER,
                                      batch_size=5, drop_last=False)
    MODEL = GCN(1, 256, 2)
    LOSS_FUNC = nn.CrossEntropyLoss()
    OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=0.01)
    MODEL.train()

    EPOCH_LOSSES = []
    for epoch in range(20):
        EPOCH_LOSS = 0
        for it, (bg, label) in enumerate(TRAIN_DATALOADER):
            pred = MODEL(bg)
            loss = LOSS_FUNC(pred, label)
            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()
            EPOCH_LOSS += loss.detach().item()
        EPOCH_LOSS /= (it+1)
        print('Epoch {}, loss {:.4f}'.format(epoch, EPOCH_LOSS))
        EPOCH_LOSSES.append(EPOCH_LOSS)

    NUM_CORRECT = 0
    NUM_TESTS = 0
    for batched_graph, labels in TEST_DATALOADER:
        pred = MODEL(batched_graph, batched_graph.ndata['attr'].float())
        NUM_CORRECT += (pred.argmax(1) == labels).sum().item()
        NUM_TESTS += len(labels)

    print('Test accuracy: {:.4f}'.format(NUM_CORRECT/NUM_TESTS))
