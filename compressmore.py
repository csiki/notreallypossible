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
shifts = range(6)  # [0] TODO

final_uniq_ms = set()
for si, shift in enumerate(shifts):
    shifted_xs_idx = [x[shift:x.shape[0] // 20 * 20 - (20 - shift)] for x in xs_idx]  # TODO from here, combine like here: https://chatgpt.com/c/373ba750-30b8-46e7-b2a7-7382386cbbaa
    ms = np.concatenate(shifted_xs_idx).reshape(-1, 20)
    uniq_ms = set(tuple(x) for x in tqdm(ms, f'uniq {si}/{len(shifts)}'))
    final_uniq_ms.update(uniq_ms)

ms_map = {u: ui for ui, u in enumerate(final_uniq_ms)}
ncodes = len(ms_map)
npossible_codes = sum((x.shape[0] for x in xs_idx)) // 20
print(f'{ncodes} / {npossible_codes} = {ncodes / npossible_codes}')

ms_tupld = [map(tuple, x.reshape(-1, 20)) for x in xs_idx]
ms_idx = [np.array([ms_map[xi] for xi in x], dtype=np.uint32)
          for x in tqdm(ms_tupld, 'map')]

ms_uniq_idx = np.array(list(ms_map.keys()))
np.save('ms_map.npy', ms_uniq_idx.astype(np.uint16))

with open('ms_map.pkl', 'wb') as f:
    pickle.dump(list(ms_map.keys()), f)

with open('ms_idx.pkl', 'wb') as f:
    pickle.dump({'ms_idx': ms_idx, 'ms_map': ms_map}, f)
print('saved ms_idx', flush=True)

# nanogpt bins
sos_token = np.uint16(ms_idx.max() + 1)
eos_token = np.uint16(ms_idx.max() + 2)
n_valid = int(.1 * len(ms_idx))

ms_idx = [np.concatenate([[sos_token], idx, [eos_token]]) for idx in tqdm(ms_idx, 'cat')]
random.shuffle(ms_idx)

train_data = ms_idx[n_valid:]
valid_data = ms_idx[:n_valid]

os.makedirs('nanoGPT/data/elec2', exist_ok=True)
np.concatenate(ms_idx).astype(np.uint32).tofile('nanoGPT/data/elec2/train.bin')  # intended don't freak out nerd
np.concatenate(valid_data).astype(np.uint32).tofile('nanoGPT/data/elec2/val.bin')  # TODO uint32 not 16 !
