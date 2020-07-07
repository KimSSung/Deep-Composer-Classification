# MIDIDataset


from torch.utils.data import DataLoader, Dataset


import torch
import numpy as np

from os.path import *
from os import listdir, path


class MIDIDataset(Dataset):
    def __init__(self, path, start, end, genre_list, option):  # start
        self.genres = genre_list  # ex) ['Classical', 'Rock', 'Country']
        self.path = path  # ex) '/data/music820_drum/'
        self.x_path = []
        self.y = []

        # num = end - start  # 5 ~ 10 -> num = 5 (5, 6, 7, 8, 9)
        if option == "folder":
            for genre in self.genres:  # iterate genres
                count = 0
                genre_dir = self.path + genre + "/"
                for f in listdir(genre_dir):  # for each file in genre
                    if count >= end:
                        break
                    if count >= start:
                        self.y.append(self.genres.index(genre))
                        self.x_path.append(genre_dir + f)  # each file path
                    count += 1

        elif option == "flat":
            for f in listdir(self.path):  # for each file in genre
                genre = f.split("_")[1]
                self.y.append(self.genres.index(genre))
                self.x_path.append(self.path + f)  # each file path

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):  # supports retrieving each file
        # Actually fetch the data
        X = np.load(self.x_path[idx], allow_pickle=True)
        Y = self.y[idx]
        F = self.x_path[idx].replace(self.path, "")
        # print(F)
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.long),
            F,
        )


# genres = ['GameMusic']
# MyDataset = MIDIDataset('/data/midi820_single/', genres, 0, 300)
# MyDataLoader = DataLoader(MyDataset, batch_size=5, shuffle=True)
# for data in MyDataLoader:
#  # print(data[0])
#  print(data[1])
#  print(data[2])
