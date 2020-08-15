from music21 import converter, corpus, instrument, midi, note, tempo
from music21 import chord, pitch, environment, stream, analysis, duration
import glob
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import csv
import pandas as pd

input_dir = "/data/inputs_full/composer0/"
midi_list = os.listdir(input_dir)
for midi in midi_list:
    for data in glob.glob(input_dir + midi + "/*.npy"):
        mat = np.load(data)
        mat_notes = mat[1]  # note channel
        nzero = mat_notes.nonzero()
        x = nzero[0]
        y = nzero[1]
        # df = pd.DataFrame(mat[1])
        # df_T = df.transpose()

        # draw plot
        plt.ylim(0, 128)
        plt.title(data)
        plt.xlabel("/0.05 sec")
        plt.ylabel("pitch")
        plt.scatter(x=x, y=y, c="red", s=2)
        plt.show()

        # print(df_T)
