"""
Create a .csv file containing the necessary information to create a
"Synthetic Dataset" to work with Deep Graph Library.

Specifications:
Must create 2 files:
edge_list.csv: 3-cloumn .csv file, where the first column indicates the
               Graph, the second the source node and the third the
               destionation.

properties.csv: 3-column .csv file, where the first column indicates
                the Graph, the second the labels and the third
                the number of nodes


Luiz Felipe da Silveira COelho - luizfelipe.coelho@smt.ufrj.br
apr 14, 2021
"""

import os
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


if __name__ == '__main__':

    PATH_FOLDER = './data_set'
    DATA_FOLDER = './raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/'
    TRAIN_FILE = 'FordA_TRAIN.tsv'
    TEST_FILE = 'FordA_TEST.tsv'
    EDGE_LIST_PATH = PATH_FOLDER+'/'+'edge_list.csv'
    PROPERTIES_PATH = PATH_FOLDER+'/'+'properties.csv'

    if not os.path.isdir(PATH_FOLDER):
        try:
            os.mkdir(PATH_FOLDER)
        except OSError:
            print('Failed to create the directory '+PATH_FOLDER+'.')
        else:
            print('Successfully created the directory '+PATH_FOLDER+'.')

    DF = pd.read_csv(DATA_FOLDER+TRAIN_FILE, sep='\t', header=None)
    LABELS_TRAIN = DF[0]
    TIMESERIES_TRAIN = DF.iloc[0:, 1:]

    if os.path.isfile(EDGE_LIST_PATH):
        CLEAR_FILE = open(EDGE_LIST_PATH, 'w')
        CLEAR_FILE.close()

    if os.path.isfile(PROPERTIES_PATH):
        CLEAR_FILE = open(PROPERTIES_PATH, 'w')
        CLEAR_FILE.close()

    for g_idx in range(LABELS_TRAIN.shape[0]):
        G = series2graph(TIMESERIES_TRAIN.iloc[0, 0:])

        with open(EDGE_LIST_PATH, 'a') as file:
            for node in G.nodes():
                for edge in G.edges([node]):
                    file.write(str(g_idx)+', '+str(edge)[1:-1]+'\n')

        with open(PROPERTIES_PATH, 'a') as file:
            file.write(str(g_idx)+', '+str(LABELS_TRAIN[g_idx-1])+', 500\n')

    plt.figure()
    plt.subplot(111)
    nx.draw(G, with_labels=True)
    plt.show()
