"""
Creates a .csv file containing the necessary information to create a
"Synthetic Dataset" to work with Deep Graph Library.

Specifications:
Must create 2 files:
edge_list.csv: 3-cloumn .csv file, where the first column indicates the
               Graph, the second the source node and the third the
               destionation.
               . header=['graph_id', 'src', 'dst']

properties.csv: 3-column .csv file, where the first column indicates
                the Graph, the second the labels and the third
                the number of nodes
                . header=['graph_id', 'label', 'num_nodes']

Luiz Felipe da Silveira COelho - luizfelipe.coelho@smt.ufrj.br
may 2, 2021
"""

import os
import csv
import pandas as pd
import networkx as nx
from tqdm import tqdm


def series2graph(timeseries):
    """Transforms a time series into a graph."""
    node_no = timeseries.shape[-1]  # Number of nodes in a graph
    graph = nx.Graph()
    graph.add_nodes_from(range(1, node_no+1))
    for i in range(1, node_no+1):
        for j in range(i+1, node_no+1):
            if j == i+1:
                graph.add_edge(i, j)
            else:
                for k in range(i+1, j):
                    right_side = timeseries[j] + (timeseries[i]-timeseries[j])*((j-k)/(j-i))
                    visible = timeseries[k] < right_side
                    if not visible:
                        break
                if visible:
                    graph.add_edge(i, j)
    return graph


if __name__ == '__main__':

    PATH_FOLDER = './data_set'
    DATA_FOLDER = './raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/'
    TRAIN_FILE = 'FordA_TRAIN.tsv'
    TEST_FILE = 'FordA_TEST.tsv'
    EDGE_LIST_PATH = PATH_FOLDER+'/'+'edge_list_test.csv'
    PROPERTIES_PATH = PATH_FOLDER+'/'+'properties_test.csv'
    EDGE_LIST_HEADER = ('graph_id', 'src', 'dst')
    PROPERTIES_HEADER = ('graph_id', 'label', 'num_nodes')

    if not os.path.isdir(PATH_FOLDER):
        try:
            os.mkdir(PATH_FOLDER)
        except OSError:
            print('Failed to create the directory '+PATH_FOLDER+'.')
        else:
            print('Successfully created the directory '+PATH_FOLDER+'.')

    DF = pd.read_csv(DATA_FOLDER+TEST_FILE, sep='\t', header=None)
    LABELS_TEST = DF[0].replace(-1, 0)
    TIMESERIES_TEST = DF.iloc[0:, 1:]

    with open(EDGE_LIST_PATH, 'w', newline='') as csvfile:
        HEADER_WRITER = csv.writer(csvfile, delimiter=',')
        HEADER_WRITER.writerow(EDGE_LIST_HEADER)

    with open(PROPERTIES_PATH, 'w', newline='') as csvfile:
        HEADER_WRITER = csv.writer(csvfile, delimiter=',')
        HEADER_WRITER.writerow(PROPERTIES_HEADER)

    for g_idx in tqdm(range(1, LABELS_TEST.shape[0]+1)):
        G = series2graph(TIMESERIES_TEST.iloc[g_idx-1, 0:])

        with open(EDGE_LIST_PATH, 'a') as csvfile:
            for node in G.nodes():
                for edge in G.edges([node]):
                    src, dst = edge
                    LINE_WRITER = csv.writer(csvfile, delimiter=',')
                    LINE_WRITER.writerow([g_idx-1, src-1, dst-1])

        with open(PROPERTIES_PATH, 'a') as csvfile:
            LINE_WRITER = csv.writer(csvfile, delimiter=',')
            LINE_WRITER.writerow([g_idx-1, LABELS_TEST[g_idx-1], 500])


# EoF
