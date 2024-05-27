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


with open('xs_idx.pkl', 'rb') as f:
    idx_data = pickle.load(f)
uniq_vals = idx_data['uniq_vals']
xs_idx = idx_data['xs_idx']

# 20 samples = 1 ms
minl = min([x.shape[0] for x in xs_idx])
minl = (minl // 20) * 20
xs_idx = np.stack([x[:minl] for x in xs_idx])

ms = xs_idx.reshape((xs_idx.shape[0], -1, 20))
uniq_splits = set(tuple(x) for x in tqdm(ms.reshape(-1, 20), 'uniq'))
ms_map = {u: ui for ui, u in enumerate(uniq_splits)}
ms_tupld = map(tuple, ms.reshape(-1, 20))
ms_idx = np.array([ms_map[x] for x in tqdm(ms_tupld, 'map')]).reshape(xs_idx.shape[0], -1)

with open('ms_idx.pkl', 'wb') as f:
    pickle.dump(ms_idx, f)

# nanogpt bins
sos_token = np.uint16(1023)
eos_token = np.uint16(1024)
n_valid = int(.1 * len(xs_idx))
np.random.shuffle(ms_idx)

train_data = ms_idx[n_valid:]
valid_data = ms_idx[:n_valid]

os.makedirs('nanoGPT/data/elec2', exist_ok=True)
np.concatenate(ms_idx).astype(np.uint16).tofile('nanoGPT/data/elec2/train.bin')  # intended don't freak out nerd
np.concatenate(valid_data).astype(np.uint16).tofile('nanoGPT/data/elec2/val.bin')
