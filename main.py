"""
Main script

Luiz Felipe da Silveira COelho - luizfelipe.coelho@smt.ufrj.br
apr 14, 2021
"""

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


def series2graph(timeseries):
    """Transforms a time series into a graph."""
    node_no = timeseries.shape[-1]  # Number of nodes in a graph
    graph = nx.Graph()
    graph.add_nodes_from(range(1, node_no+1), color='blue')
    for i in range(1, node_no+1):
        for j in range(i+1, node_no+1):
            if j > i+1:
                for k in range(i+1, j):
                    right_side = timeseries[j] + (timeseries[i]-timeseries[j])*((j-k)/(j-i))
                    visible = timeseries[k] < right_side
                    if not visible:
                        break
                if visible:
                    graph.add_edge(i, j, color='red')
    return graph


DATA_FOLDER = './raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/'
TRAIN_FILE = 'FordA_TRAIN.tsv'
TEST_FILE = 'FordA_TEST.tsv'

DF = pd.read_csv(DATA_FOLDER+TRAIN_FILE, sep='\t', header=None)
LABELS_TRAIN = DF[1]
TIMESERIES_TRAIN = DF.iloc[0:, 1:]

G1 = series2graph(TIMESERIES_TRAIN.iloc[0, 0:])


plt.figure()
plt.subplot(111)
nx.draw(G1, with_labels=True)
plt.show()
