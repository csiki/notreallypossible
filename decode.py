import sys
import numpy as np
import pickle
from fun import *


in_path = sys.argv[1]
out_path = sys.argv[2]
sr = 19531

with open('xs_idx.pkl', 'rb') as f:
    xs_map = pickle.load(f)['xs_map']

# with open('ms_idx.pkl', 'rb') as f:
#     ms_map = pickle.load(f)['ms_map']
ms_map = np.load('ms_map.npy')
ms_map = {tuple(m): mi for mi, m in enumerate(ms_map)}

# invert maps
inv_xs_map = {v: k for k, v in xs_map.items()}
inv_ms_map = {v: k for k, v in ms_map.items()}

max_ms = max(list(ms_map.values()))
start_residual_code = max_ms + 1
stop_code = 1023  # needs to be 10bit, but not used by xs
nbits = np.ceil(np.log2(start_residual_code)).astype(int)

# Read the binary file
with open(in_path, 'rb') as file:
    byte_array = bytearray(file.read())

# Convert bytes to bitstring
bitstring = ''.join(f'{byte:08b}' for byte in byte_array)

# Decode the bitstring
index = 0
decoded_values = []

# Read 22-bit values until start_residual_code is found
while index < len(bitstring) - nbits:
    value = int(bitstring[index:index + nbits], 2)
    if value == start_residual_code:
        index += nbits
        break
    decoded_values.append(list(inv_ms_map[value]))
    index += nbits

# Read 10-bit values until stop_code is found
while index < len(bitstring) - 10:
    value = int(bitstring[index:index + 10], 2)
    if value == stop_code:
        break
    decoded_values.append([value])
    index += 10

# Convert list to numpy array of type uint16
decoded_values = [inv_xs_map[v] for v in sum(decoded_values, [])]
decoded_values = np.array(decoded_values, dtype=np.uint16)

save_wav(decoded_values, out_path, sr)
