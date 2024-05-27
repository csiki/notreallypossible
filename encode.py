import os
import sys

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


# TODO little slow

with open('xs_idx.pkl', 'rb') as f:
    xs_map = pickle.load(f)['xs_map']

# with open('ms_idx.pkl', 'rb') as f:
#     ms_map = pickle.load(f)['ms_map']
ms_map = np.load('ms_map.npy')
ms_map = {tuple(m): mi for mi, m in enumerate(ms_map)}

in_path = sys.argv[1]
out_path = sys.argv[2]

wav = read_wav(in_path)[0]
max_ms = max(list(ms_map.values()))
start_residual_code = max_ms + 1
stop_code = 1023  # needs to be 10bit, but not used by xs
nbits = np.ceil(np.log2(start_residual_code)).astype(int)

# map with xs then ms, leave the len % 20 residual encoded bitbybit after a stop_codde
xswav = np.vectorize(xs_map.get)(wav)
ms_cutoff = xswav.shape[0] // 20 * 20

mswav = xswav[:ms_cutoff]
post_mswav = xswav[ms_cutoff:]

ms_tupld = map(tuple, mswav.reshape(-1, 20))
mswav = np.array([ms_map[x] for x in ms_tupld])

# to bitstring
bitstring = ''
for x in mswav:
    # Format each value as a 22-bit binary string and concatenate
    bitstring += f'{x:0{nbits}b}'

bitstring += f'{start_residual_code:0{nbits}b}'
for x in post_mswav:
    bitstring += f'{x:0{10}b}'  # 10bit enc of the remainder
bitstring += f'{stop_code:0{10}b}'

# ensure the bitstring length is a multiple of 8
while len(bitstring) % 8 != 0:
    bitstring += '0'

byte_array = bytearray(int(bitstring[i:i+8], 2) for i in range(0, len(bitstring), 8))
byte_array = bytes(byte_array)

with open(out_path, 'wb') as file:
    file.write(byte_array)
