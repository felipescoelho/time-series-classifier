"""
Train a GNN for Graph classification.

Luiz Felipe da Silveira Coelho - luizfelipe.coelho@smt.ufrj.br
apr 18, 2021
"""

import pandas as pd
import matplotlib.pyplot as plt
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader
from dgl.data import DGLDataset
from dgl.nn import GraphConv
from math import sqrt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]
})


class SyntheticDataset(DGLDataset):
    """
    Synthetic dataset generated from my edge list and properties list.
    """

    def __init__(self):
        super().__init__(name='synthetic')

    def process(self):
        """Process .csv files."""
        edges = pd.read_csv('./data_set/edge_list.csv')
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


class SyntheticTestset(DGLDataset):
    """
    Synthetic dataset generated from my edge list and properties list.
    """

    def __init__(self):
        super().__init__(name='synthetic')

    def process(self):
        """Process .csv files."""
        edges = pd.read_csv('./data_set/edge_list_test.csv')
        properties = pd.read_csv('./data_set/properties_test.csv')
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

    def __init__(self,
                 in_feats,
                 hidden_dim,
                 num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, num_classes)

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
    IMAGE_FOLDER = './images/'
    # Load the synthetic dataset:
    TRAINSET = SyntheticDataset()
    TESTSET = SyntheticTestset()

    # Define data loader:
    NUM_TRAIN = len(TRAINSET)
    NUM_TEST = len(TESTSET)

    TRAIN_SAMPLER = SubsetRandomSampler(torch.arange(NUM_TRAIN))
    TEST_SAMPLER = SubsetRandomSampler(torch.arange(NUM_TEST))

    TRAIN_DATALOADER = GraphDataLoader(TRAINSET, sampler=TRAIN_SAMPLER,
                                       batch_size=13, drop_last=False)
    TEST_DATALOADER = GraphDataLoader(TESTSET, sampler=TEST_SAMPLER,
                                      batch_size=11, drop_last=False)

    MODEL = GCN(1, 56, 2)
    LOSS_FUNC = nn.CrossEntropyLoss()
    OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=0.001)
    MODEL.train()

    EPOCH_LOSSES = []
    for epoch in range(1000):
        EPOCH_LOSS = 0
        for it, (bg, label) in enumerate(TRAIN_DATALOADER):
            # import pdb
            # pdb.set_trace()

            pred = MODEL(bg)
            loss = LOSS_FUNC(pred, label)
            OPTIMIZER.zero_grad()
            loss.backward()
            OPTIMIZER.step()
            EPOCH_LOSS += loss.detach().item()
        EPOCH_LOSS /= (it+1)
        print('Epoch {}, loss {:.4f}'.format(epoch, EPOCH_LOSS))
        EPOCH_LOSSES.append(EPOCH_LOSS)

    # Plot entropy
    GOLDEN_RATIO = (1+sqrt(5))/2
    WIDTH = 8.9
    HEIGHT = WIDTH/GOLDEN_RATIO

    FIG = plt.figure(figsize=(WIDTH, HEIGHT))
    AX = FIG.add_subplot(111)
    AX.plot(EPOCH_LOSSES, linewidth=2.5, color='#3783ff')
    AX.tick_params(axis='both', labelsize=20)
    AX.set_xlabel('Épocas', size=20)
    AX.set_ylabel('Perda logaritmica', size=20)
    AX.grid()
    for axis in ['top', 'bottom', 'left', 'right']:
        AX.spines[axis].set_linewidth(2)
    FIG.tight_layout()
    FIG.savefig(IMAGE_FOLDER+'avg_entropy.svg', format='svg',
                bbox_inches='tight')

    NUM_CORRECT = 0
    NUM_TESTS = 0
    for batched_graph, labels in TEST_DATALOADER:
        pred = MODEL(batched_graph)
        NUM_CORRECT += (pred.argmax(1) == labels).sum().item()
        NUM_TESTS += len(labels)

    print('Test accuracy: {:.4f}'.format(NUM_CORRECT/NUM_TESTS))
