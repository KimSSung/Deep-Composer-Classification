from music21 import converter, corpus, instrument, midi, note, tempo
from music21 import chord, pitch, environment, stream, analysis, duration
import glob
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import csv
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


temp_dict = dict()
for k in range(13):  # 13 composers
    temp_dict.update(
        {k: [0 for i in range(61)]}
    )  # list index = midi (value = segments)

# total_dict = {"composer": list(), "midi": list(), "segment": list()}
input_dir = "/data/inputs/*/"
dir_list = glob.glob(input_dir)
for dir in dir_list:
    midi_list = glob.glob(dir + "*/")  # midi
    for midi in midi_list:
        seg_list = glob.glob(midi + "*.npy")
        composer = int(
            dir.replace("/data/inputs/", "").replace("/", "").replace("composer", "")
        )
        mid = int(midi.replace(dir, "").replace("/", "").replace("midi", ""))
        # total_dict["composer"].append(composer)
        # total_dict["midi"].append(mid)
        # total_dict["segment"].append(int(len(seg_list)))

        # print(temp_dict[composer])
        temp_dict[composer][mid] = int(len(seg_list) / 10)


# plot
temp_df = pd.DataFrame(temp_dict)
temp_df = temp_df.transpose()
# print(temp_df)
# temp_df.plot(kind="bar")
plt.table(temp_df)

# total_df = pd.DataFrame(total_dict)
# total_df.sort_values(by=["composer", "midi"], axis=0, inplace=True)
# total_df.plot(kind="hist")
# print(total_df)


# plt.ylim(0, 30)
# plt.title("segment distribution")
plt.show()
