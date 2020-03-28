import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import sklearn
import torchaudio
import torch
import matplotlib.pyplot as plt
from torchaudio import transforms
from torch import utils
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# For reproducibility purposes
torch.manual_seed(123)

# path
PATH = '../../../../data/'
gtzan_dir = PATH + 'wav/genres/8genres/'
song_samples = 660000
genres = {'classical': 0, 'disco': 1, 'jazz': 2, 'pop': 3, 'country': 4,
          'hiphop': 5, 'metal': 6, 'reggae': 7}
num_genres = 8 #8genres of wav

#for GPU use
device = torch.device('cuda:0')


###############################TORCH VERSION#####################################

#1. read .wav file names into array
def read_data(src_dir, genres):
    arr_fn = []
    arr_genres = []

    for x,_ in genres.items():
        folder = src_dir + x #each genre folder
        for path, subdir, files in os.walk(folder):
            for file in files: #each .wav file
                file_name = folder + "/" + file

                #save file name & matching genre
                arr_fn.append(file_name)
                arr_genres.append(genres[x])

    # data_set = D.TensorDataset(arr_fn, arr_genres)
    # train, test = torch.utils.data.random_split(data_set, [int(len(data_set)*0.7), int(len(data_set)*0.3)])

    return arr_fn, arr_genres


#2. load .wav -> (split into windows) -> tranform to mel
def prepare_data(fn, genres):
    allChunks_x = []
    allChunks_y = []
    for filename, genre in zip(fn, genres): #iterate as couple

        #1. load .wav file
        waveform, sample_rate = torchaudio.load(filename) #return waveform(torch.Tensor)
        waveform = waveform[:, :song_samples] #cut to 660000 size <== torch.Size([1, 660000)]
        # waveform = waveform.squeeze()

        # #2. split SINGLE SONG into MULTIPLE SONGS using window(0.05) , overlap(0.5)
        window = 0.05
        # overlap = 0.5
        wvsize = song_samples
        chunk = int(wvsize * window)
        # offset = int(chunk * (1. - overlap))
        offset = int(chunk)
        spsong = [waveform[:, i:i + chunk] for i in range(0, wvsize - chunk + offset, offset)]

        for each in spsong:
            if each.shape[1] != chunk:
                continue
            allChunks_x.append(each)
            allChunks_y.append(genre)

    return allChunks_x, allChunks_y


#3. do MELSPECTROGRAM conversion
def getMel(all_chunks, all_y):
    n_fft = 1024
    hop_length = 512
    n_mels = 128
    win_length = 1024
    sample_rate = 16000

    mel_x = []
    for each_chunk in tqdm(all_chunks):
        mel_specgram = transforms.MelSpectrogram(n_fft=n_fft,sample_rate=sample_rate, hop_length=hop_length, n_mels=n_mels, win_length=win_length)(each_chunk)
        # mel_specgram = torch.log10(1 + 10*mel_specgram) ??
        mel_x.append(mel_specgram)

    return mel_x



######################################RUN CODE######################################

#get filename
fn, genres = read_data(gtzan_dir, genres)

#shuffle
zipped = list(zip(fn, genres))
np.random.shuffle(zipped)
fn, genres = zip(*zipped)

#train : valid = 8 : 2
train_len = int(len(fn)*8/10)
fn_trn, fn_val, genres_trn, genres_val = fn[:train_len], fn[train_len:], genres[:train_len], genres[train_len:]

tx, ty = prepare_data(fn_trn, genres_trn)
vx, vy = prepare_data(fn_val, genres_val)
# ch_x, ch_y = prepare_data(fn, genres)
tdata_x = getMel(tx, ty)
vdata_x = getMel(vx, vy)
# data_x = getMel(ch_x, ch_y)
# genres = ch_y #expand to chunk length

#expand to chunk length
# tdata_y = ty
# vdata_y = vy

# #expand to chunk length
genres_trn = ty
genres_val = vy

#shuffle (one more time after truncation)
tdata_xy = list(zip(tdata_x, genres_trn))
vdata_xy = list(zip(vdata_x, genres_val))
np.random.shuffle(tdata_xy)
np.random.shuffle(vdata_xy)
tdata_x, tdata_y = zip(*tdata_xy)
vdata_x, vdata_y = zip(*vdata_xy)

#work needed for tensorDataset: tuple -> list -> tensorfy
train_x, val_x, train_y, val_y = list(tdata_x), list(vdata_x), list(tdata_y), list(vdata_y)
train_x, val_x = torch.stack(train_x), torch.stack(val_x)
train_y, val_y = torch.tensor(train_y), torch.tensor(val_y)

# # shuffle
# data_xy = list(zip(data_x, genres))
# np.random.shuffle(data_xy)
# data_x, data_y = zip(*data_xy)
#
# # train : valid = 8 : 2
# train_len = int(len(data_x) * 8 / 10)
# train_x, val_x, train_y, val_y = data_x[:train_len], data_x[train_len:], data_y[:train_len], data_y[train_len:]
#
# # work needed for tensorDataset
# train_x, val_x, train_y, val_y = list(train_x), list(val_x), list(train_y), list(val_y)
# train_x, val_x = torch.stack(train_x), torch.stack(val_x)
# train_y, val_y = torch.tensor(train_y), torch.tensor(val_y)

# tensorDataset
t = TensorDataset(train_x, train_y)
v = TensorDataset(val_x, val_y)

# create batch
batch_size = 32
train_loader = DataLoader(t, batch_size=batch_size, shuffle=True) # 5(data for each batch) x 128(batches) = 640 data
val_loader = DataLoader(v, batch_size=batch_size, shuffle=True) # 5(data for each batch) x 32(batches) = 160 data



################################## NEURAL NETWORK ######################################

class GenreClassModel(nn.Module):

    def __init__(self):
        super(GenreClassModel, self).__init__()

        self.pool = nn.MaxPool2d(stride=(2,2), kernel_size=(2,2))
        self.dropout025 = nn.Dropout(0.25)
        self.dropout05 = nn.Dropout(0.5)
        self.dense = nn.Linear(in_features=2048, out_features=512)
        self.pred = nn.Linear(in_features=512, out_features=num_genres)

        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1) #in / out(filters) / filter_size / stride
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)

        self.batch1 = nn.BatchNorm2d(16)
        self.batch2 = nn.BatchNorm2d(32)
        self.batch3 = nn.BatchNorm2d(64)
        self.batch4 = nn.BatchNorm2d(128)
        self.batch5 = nn.BatchNorm2d(256)


    def forward(self, x):
        # cnn2d -> relu -> maxpool2d -> dropout
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout025(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout025(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout025(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout025(x)
        x = self.pool(F.relu(self.conv5(x)))
        x = self.dropout025(x)
        # x => torch.Size([5, 256, 2, 30)] => torch.Size([5, 256, 4, 32)]


        # global pooling and MLP
        x = x.view(x.size(0), -1) # = flatten in keras
        x = self.dropout05(x)
        x = F.relu(self.dense(x))
        x = self.dropout025(x)
        predictions = self.pred(x) #supposed to be [batch_size, classes]
        # print(predictions)

        return predictions



################################## LET'S TRAIN! ######################################

model = GenreClassModel()

#hyper params
criterion = nn.CrossEntropyLoss()
learning_rate = 0.0005
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
# optimizer = optim.SGD(model.parameters(),lr=learning_rate)
num_epochs = 200
num_batches = len(train_loader)

#use GPU
model = model.to(device)
criterion = criterion.to(device)

trn_loss_list = []
val_loss_list = []
trn_acc_list = []
val_acc_list= []
for epoch in tqdm(range(num_epochs)):
    print('\n')
    trn_loss = 0.0
    trn_correct = 0
    trn_total = 0
    for i, trainset in enumerate(train_loader):
        #train mode
        model.train()
        #unpack
        train_in, train_out = trainset
        #use GPU
        train_in = train_in.to(device)
        train_out = train_out.to(device)
        #grad init
        optimizer.zero_grad()
        #forward pass
        train_pred = model(train_in)
        #calculate acc
        _, label_pred = torch.max(train_pred.data, 1)
        trn_total += train_out.size(0)
        trn_correct += (label_pred == train_out).sum().item()
        #calculate loss
        t_loss = criterion(train_pred, train_out)
        #back prop
        t_loss.backward()
        #weight update
        optimizer.step()

        trn_loss += t_loss.item()


        #####################################################
        # # TO BE ERASED
        # if (i + 1) % 30 == 0:
        #     print('predictions: {}'.format(label_pred))
        #     print('truth: {}'.format(train_out))
        #     print((label_pred == train_out).sum().item())
        ####################################################

        mini_batch = 100
        # VALIDATION of every 100 mini-batches
        if (i + 1) % mini_batch == 0:
            # validate mode
            model.eval()
            with torch.no_grad():  # important!!! for validation
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                for j, valset in enumerate(val_loader):
                    val_in, val_out = valset
                    # to GPU
                    val_in = val_in.to(device)
                    val_out = val_out.to(device)

                    # forward
                    val_pred = model(val_in)
                    v_loss = criterion(val_pred, val_out)
                    val_loss += v_loss

                    # accuracy
                    _, val_label_pred = torch.max(val_pred.data, 1)
                    val_total += val_out.size(0)
                    val_correct += (val_label_pred == val_out).sum().item()

            print("epoch: {}/{} | step: {}/{} | trn loss: {:.4f} | val loss: {:.4f} | trn acc: {:.2f}% | val acc: {:.2f}%"
                .format(epoch + 1, num_epochs,
                        i + 1, num_batches,
                        trn_loss / mini_batch,
                        val_loss / len(val_loader),
                        100*(trn_correct / trn_total),
                        100*(val_correct / val_total)))

            trn_loss_list.append(trn_loss / mini_batch)
            val_loss_list.append(val_loss / len(val_loader))
            trn_acc_list.append(100*(trn_correct / trn_total))
            val_acc_list.append(100*(val_correct / val_total))


            # reinit to 0
            trn_loss = 0.0
            trn_total = 0
            trn_correct = 0




# Summarize history for accuracy
plt.plot(trn_acc_list)
plt.plot(val_acc_list)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('mini-batch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(trn_loss_list)
plt.plot(val_loss_list)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('mini-batch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()