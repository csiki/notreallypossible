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
from fun import *


# llm solution
# rsq solution
# rsq + llm solution
# tree solution
# massive parallel compute find patterns solution
# it's a binary tree to the next ms and a little better, but you can update at each sample
#   make a tree, then split/rearrange it into binary
#   nodes can be longer than 200b
# just do a nanogpt


wav_files = sorted(glob('data/*.wav'))
x, sr = read_wav(wav_files[0])
xs, srs = zip(*[read_wav(w) for w in wav_files])
n_uniq_vals = np.array([np.unique(x).shape[0] for x in xs])

if not os.path.isfile('xs_idx.pkl'):
    uniq_vals = np.unique(np.concatenate(xs))
    value_to_index = {val: idx for idx, val in enumerate(uniq_vals)}
    xs_idx = [np.vectorize(value_to_index.get)(x) for x in tqdm(xs, 'map')]
    with open('xs_idx.pkl', 'wb') as f:
        pickle.dump({'xs_idx': xs_idx, 'xs_map': value_to_index}, f)

else:
    with open('xs_idx.pkl', 'rb') as f:
        idx_data = pickle.load(f)
    uniq_vals = idx_data['uniq_vals']
    xs_idx = idx_data['xs_idx']

# nanogpt bins
sos_token = np.uint16(1023)  # xs max is 1022
eos_token = np.uint16(1024)
n_valid = int(.1 * len(xs_idx))
random.shuffle(xs_idx)

xs_idx = [np.concatenate([[sos_token], idx, [eos_token]]) for idx in xs_idx]

train_data = xs_idx[n_valid:]
valid_data = xs_idx[:n_valid]

os.makedirs('nanoGPT/data/elec', exist_ok=True)
np.concatenate(xs_idx).astype(np.uint16).tofile('nanoGPT/data/elec/train.bin')  # no worries
np.concatenate(valid_data).astype(np.uint16).tofile('nanoGPT/data/elec/val.bin')


for idx in xs_idx[:10]:
    plt.plot(idx)
plt.show()
