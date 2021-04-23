"""Image Generator"""

from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from convert2csv import series2graph

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

if __name__ == '__main__':
    DATA_FOLDER = './raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/'
    TRAIN_FILE = 'FordA_TRAIN.tsv'
    IMAGE_FOLDER = './images/'

    DF = pd.read_csv(DATA_FOLDER+TRAIN_FILE, sep='\t', header=None)
    LABELS_TRAIN = DF[0].replace(-1, 0)
    TIMESERIES_TRAIN = DF.iloc[0:, 1:]

    G1 = series2graph(TIMESERIES_TRAIN.iloc[0, 0:])
    # nodes1 = nx.draw_networkx_nodes(G1, pos=nx.spring_layout(G1),
    #                                 node_size=250, node_color='#e04f39')
    G2 = series2graph(TIMESERIES_TRAIN.iloc[1, 0:])
    # nodes2 = nx.draw_networkx_nodes(G2, pos=nx.spring_layout(G2),
    #                                 node_size=250, node_color='#e04f39')

    # Plot time series examples:
    GOLDEN_RATIO = (1+sqrt(5))/2
    WIDTH = 8.9
    HEIGHT = WIDTH/GOLDEN_RATIO

    FIG1 = plt.figure(figsize=(WIDTH, HEIGHT))
    AX = FIG1.add_subplot(111)
    AX.plot(TIMESERIES_TRAIN.iloc[0, 0:], linewidth=2.5, color='#3783ff',
            label='Rótulo: $' + str(LABELS_TRAIN[0]) + '$')
    AX.plot(TIMESERIES_TRAIN.iloc[1, 0:], linewidth=2.5, color='#fa0101',
            label='Rótulo: $' + str(LABELS_TRAIN[1]) + '$')
    AX.grid()
    AX.legend(prop={'size': 20})
    AX.tick_params(axis='both', labelsize=20)
    AX.set_xlabel('Amostras, $n$', size=20)
    AX.set_ylabel('Amplitude, $x(n)$', size=20)
    for axis in ['top', 'bottom', 'left', 'right']:
        AX.spines[axis].set_linewidth(2)
    FIG1.tight_layout()
    FIG1.savefig(IMAGE_FOLDER+'series_examples.svg', format='svg',
                 bbox_inches='tight')

    # Plot graph example
    FIG2 = plt.figure(figsize=(WIDTH, HEIGHT))
    AX1 = FIG2.add_subplot(121)
    nx.draw(G1, pos=nx.kamada_kawai_layout(G1), node_size=50,
            node_color='#3783ff')
    AX1.set_title('(a)', size=20)
    AX2 = FIG2.add_subplot(122)
    nx.draw(G2, pos=nx.kamada_kawai_layout(G2), node_size=50,
            node_color='#fa0101')
    AX2.set_title('(b)', size=20)
    FIG2.tight_layout()
    FIG2.savefig(IMAGE_FOLDER+'graph_examples.svg', format='svg',
                 bbox_inches='tight')


# EoF
