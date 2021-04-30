"""
Correct list. The list I've made has nodes starting from 1, and for the
gdl package it must start at 0.
"""

# import os
import csv
# from tqdm import tqdm

FOLDER = './data_set/'
FILE1 = 'edge_list.csv'
FILE2 = 'edge_list_correction.csv'
EDGE_LIST_HEADER = ('graph_id', 'src', 'dst')

with open(FOLDER+FILE2, 'w', newline='') as csvfile:
    HEADER_WRITER = csv.writer(csvfile, delimiter=',')
    HEADER_WRITER.writerow(EDGE_LIST_HEADER)

with open(FOLDER+FILE1, 'r') as file:
    LINE = file.readline()
    LINE = file.readline()
    while LINE:
        AUX = LINE.split()
        G_ID, SRC, DST = AUX[0].split(',')
        with open(FOLDER+FILE2, 'a') as correct_file:
            LINE_WRITER = csv.writer(correct_file, delimiter=',')
            LINE_WRITER.writerow([G_ID, int(SRC)-1, int(DST)-1])
        LINE = file.readline()
