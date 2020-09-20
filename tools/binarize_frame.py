import os
import sys
import copy
from tqdm import tqdm
import torch.utils.data
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import glob
import pandas as pd

load_path = "/data/inputs_full/"
save_path = "/data/inputs_binarized/"
count = 0
for comp in glob.glob(load_path+"*/"):
    for midi in glob.glob(comp+"*/"):
        if not os.path.exists(save_path + midi.replace(load_path, "")):
            os.makedirs(save_path + midi.replace(load_path, ""))
        for file in glob.glob(midi+"*.npy"):
            piece = np.load(file)
            bin_piece = np.copy(piece)
            bin_piece = np.clip(bin_piece, 0,1)
            save_loc = save_path + file.replace(load_path, "")
            np.save(save_loc, bin_piece)
            count+=1
            print("{}: {}".format(count, save_loc))



