import os

import numpy as np
import wave
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import pickle
import networkx as nx


def build_forest(split_idx):
    forest = nx.DiGraph()

    for electrode_data in tqdm(split_idx, 'electrode'):
        for i in range(len(electrode_data) - 1):
            current_id = electrode_data[i]
            next_id = electrode_data[i + 1]

            if forest.has_node(current_id):
                forest.add_node(next_id)
                forest.add_edge(current_id, next_id)
            else:
                forest.add_node(current_id)
                forest.add_node(next_id)
                forest.add_edge(current_id, next_id)

    return forest


def count_trees(forest):
    undirected_forest = forest.to_undirected(as_view=True)
    num_trees = nx.number_connected_components(undirected_forest)
    return num_trees


with open('ms_idx.pkl', 'rb') as f:
    split_idx = pickle.load(f)

# Build the forest
forest = build_forest(split_idx)
ntrees = count_trees(forest)

# TODO binarize trees

# Print the edges of the forest to visualize the structure
print('trees:', ntrees, '; edges:', len(list(forest.edges)))
pos = nx.spring_layout(forest)
plt.figure(figsize=(16, 16))
nx.draw(forest, pos, with_labels=False, node_size=5, node_color='skyblue', font_size=8, edge_color='gray')
plt.title("Forest of Trees from split_idx")
plt.savefig('forest.png')
plt.show()

# plt.imshow(xs_idx[:, 10000:15000])
# plt.show()

# for x in xs_idx[:20]:
#     plt.hist(x, bins=100, alpha=.2)
# plt.axis('off')
# plt.show()
#
# plt.hist(np.concatenate(xs_idx), bins=100);plt.show()
print()

