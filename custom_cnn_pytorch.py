import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sklearn
import torchaudio
import torch
import matplotlib.pyplot as plt
from torchaudio import transforms
import numpy as np

# For reproducibility purposes
np.random.seed(42)

# path
PATH = '../../../../data/'
gtzan_dir = PATH + 'wav/genres/'
song_samples = 660000
genres = {'classical': 0, 'disco': 1, 'jazz': 2, 'pop': 3, 'country': 4,
          'hiphop': 5, 'metal': 6, 'reggae': 7}
num_genres = 8 # 8genres of wav



###############################TORCH VERSION#####################################

#1. read .wav file names into array
def read_data(src_dir, genres):
    arr_fn = []
    arr_genres = []

    for x,_ in genres.items():
        folder = src_dir + x #each genre folder
        for _, _, files in os.walk(folder):
            for file in files: #each .wav file
                file_name = folder + "/" + file

                #save file name & matching genre
                arr_fn.append(file_name)
                arr_genres.append(genres[x])

    return arr_fn, arr_genres

fn, gnrs = read_data(gtzan_dir, genres)
print(fn, gnrs)

#2. load .wav -> (split into windows) -> tranform to mel
def prepare_data(fn, genres):

    for filename, genre in zip(fn, genres): #iterate as couple

        #1. load .wav file
        waveform, sample_rate = torchaudio.load(filename) #return waveform(torch.Tensor)
        waveform = waveform[:song_samples] #cut to 660000 size

        # #2. split into multiple songs using window(0.05) , overlap(0.5)
        # split_data(waveform, genre)


        #3.
        n_fft = 1024
        hop_length = 256
        n_mels = 128
        mel_specgram = transforms.MelSpectrogram(n_fft=n_fft,
        sample_rate=sample_rate, hop_length=hop_length, n_mels=n_mels)(waveform)
        print(mel_specgram)
        resized_mel = mel_specgram[:,:,np.newaxis] #add a dimension -> 3D

    return resized_mel

print(prepare_data(fn, gnrs))

# def split_data(x, y, window=0.05, overlap=0.5):
#     split_x = []
#     split_y = []
#
#     xshape = x.shape[0]
#     chunk = int(xshape * window)
#     offset = int(chunk * (1. - overlap))
#
#     spsong = [x[i:i + chunk] for i in range(0, xshape - chunk + offset, offset)]
#     for s in spsong:
#         if s.shape[0] != chunk:
#             continue
#
#         split_x.append(s)
#         split_y.append(y)
#     return split_x, split_y

#####################################################################################



#torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
class GenreClassModel(nn.Module):

    def __init__(self):
        super(GenreClassModel, self).__init__()
        self.pool = nn.MaxPool2d(stride=(2,2))
        self.dropout = nn.Dropout(0.25)
        self.dropout1 = nn.Dropout(0.5)
        self.dense = nn.Linear(512)
        self.pred = nn.Linear(num_genres)

        self.conv1 = nn.Conv2d(1, 16, (3,3), stride=(1,1)) #in / out(filters) / filter_size / stride
        self.conv2 = nn.Conv2d(16, 32, (3,3), stride=(1,1))
        self.conv3 = nn.Conv2d(32, 64, (3,3), stride=(1,1))
        self.conv4 = nn.Conv2d(64, 128, (3,3), stride=(1,1))
        self.conv5 = nn.Conv2d(128, 256, (3,3), stride=(1,1))


    def forward(self, x):
        # cnn2d -> relu -> maxpool2d -> dropout
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv5(x)))
        x = self.dropout(x)

        # global pooling and MLP
        x = torch.flatten(x)
        x = self.dropout1(x)
        x = F.relu(self.dense(x))
        x = self.dropout(x)
        x = F.softmax(x)

        return x




