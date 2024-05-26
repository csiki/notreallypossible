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


with open('xs_idx.pkl', 'rb') as f:
    idx_data = pickle.load(f)
uniq_vals = idx_data['uniq_vals']
xs_idx = idx_data['xs_idx']

minl = min([x.shape[0] for x in xs_idx])
minl = (minl // 20) * 20
xs_idx = np.stack([x[:minl] for x in xs_idx])

splits = xs_idx.reshape((xs_idx.shape[0], -1, 20))
uniq_splits = set(tuple(x) for x in tqdm(splits.reshape(-1, 20), 'uniq'))
split_map = {u: ui for ui, u in enumerate(uniq_splits)}
splits_tupld = map(tuple, splits.reshape(-1, 20))
split_idx = np.array([split_map[x] for x in tqdm(splits_tupld, 'map')]).reshape(xs_idx.shape[0], -1)


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

