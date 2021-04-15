"""Dataset downloader and visualizer."""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})


def readucr(filename):
    """Read and download files."""
    data = np.loadtxt(filename, delimiter="\t")
    labels = data[:, 0]
    time_series = data[:, 1:]
    return time_series, labels.astype(int)


ROOT_URL = "http://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

TIMESERIES_TRAIN, LABELS_TRAIN = readucr(ROOT_URL + "FordA_TRAIN.tsv")
TIMESERIES_TEST, LABELS_TEST = readucr(ROOT_URL + "FordA_TEST.tsv")

# Both possible classes
CLASSES = np.unique(np.concatenate((LABELS_TRAIN, LABELS_TEST), axis=0))


# Plot one example from each class
FIG = plt.figure()
AX = FIG.add_subplot(111)
for c in CLASSES:
    X_TRAIN = TIMESERIES_TRAIN[LABELS_TRAIN == c]
    AX.plot(X_TRAIN[0], label='Class: ' + str(c))
AX.legend(loc='best')
AX.grid()
AX.set_title('Exemplo de dados')
AX.set_xlabel('Amostras, $t$')
AX.set_ylabel('Amplitude, $x(t)$')
plt.show()
