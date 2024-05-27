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


file1 = 'data/0052503c-2849-4f41-ab51-db382103690c.wav'  # sys.argv[1]
file2 = 'data/0052503c-2849-4f41-ab51-db382103690c.wav.copy'  # sys.argv[2]

wav1 = read_wav(file1)[0]
wav2 = read_wav(file2)[0]

print(np.all(wav1 == wav2))

# plt.plot(wav1)
# plt.plot(wav2)
# plt.show()
print('a')
