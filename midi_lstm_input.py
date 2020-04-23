import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from os.path import *
from os import *
import pandas as pd


class GenreFeatureData:

    genres = ['Classical', 'Jazz', 'Pop', 'Country', 'Rock']
    min_shape = None

    train_X = train_Y = None
    dev_X = dev_Y = None
    test_X = test_Y = None

    def __init__(self):
        self.min_shape = 370

        self.input_total = []
        self.output_total = []


    def load_input_data(self):

        for genre in self.genres:

            input_dir = "/data/midi370_input/" + genre + "_input.npy"
            load_saved = np.load(input_dir, allow_pickle=True)
            self.input_total.append(load_saved)
            if(load_saved.shape[0] < min_shape):
                min_shape = load_saved.shape[0] # num of data in genre
            output_temp = [self.genres.index(genre)]*load_saved.shape[0]
            self.output_total.append(output_temp)

        input_total = np.asarray(self.input_total)
        # print(input_total.shape)
        # print(input_total)
        input_total = input_total[:,:min_shape] #error cases -> smaller than 370
        output_total = np.asarray(self.output_total)
        print(input_total.shape)
        print(output_total.shape)



    def preprocess_data(self):

        # self.train_X =
        # self.train_Y =
        # self.dev_X =
        # self.dev_Y =
        # self.test_X =
        # self.test_Y =
