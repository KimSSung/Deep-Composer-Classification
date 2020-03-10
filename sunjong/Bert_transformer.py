import torch
import BMI
from transformers import BertConfig
# import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
import os
import random
import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import pandas as pd

from mido import MidiFile


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_800(genre_list):
	rlist = [] #list for return
	dir_genre = './newintinput/800'
	for genre in genre_list:
		new_path = dir_genre + '/' + genre + ".pickle"
		df = pd.read_pickle(new_path)
		tlist = df.values.tolist() #2d list
		rlist = rlist + tlist

	return rlist


def get_attention(genre_list):
	rlist = [] #list for return
	dir_genre = './newintinput/attention'
	for genre in genre_list:
		new_path = dir_genre + '/' + genre + ".pickle"
		df = pd.read_pickle(new_path)
		tlist = df.values.tolist() #2d list
		rlist = rlist + tlist

	return rlist


def get_token(genre_list):
	rlist = []
	dir_genre = './newintinput/token'
	for genre in genre_list:
		new_path = dir_genre + '/' + genre + ".pickle"
		df = pd.read_pickle(new_path)
		tlist = df.values.tolist()  # 2d list
		rlist = rlist + tlist

	return rlist


###################################################################


# Converting midi into Bert's input -> !! list !!

genres = ['Classical', 'Jazz', 'Pop', 'Country','Rock']

input_ids = get_800(genres)
token_type_ids = get_token(genres)
attention_mask = get_attention(genres)
labels = [0]*4000
labels[800:1599] = [1] * 800
labels[1600:2399] = [2] * 800
labels[2400:3199] = [3] * 800
labels[3200:3999] = [4] * 800


data = list(zip(input_ids, token_type_ids, attention_mask, labels)) #zip data structure
random.shuffle(data)

input_ids, token_type_ids, attention_mask, labels = zip(*data)  # tuples

input_ids, token_type_ids, attention_mask, labels = np.array(input_ids), np.array(token_type_ids), np.array(attention_mask), np.array(labels)  # tuple -> ndarray
input_ids, token_type_ids, attention_mask, labels = torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), torch.tensor(labels)  # tensor

train_len = len(input_ids) * 8 / 10  # train : valid = 8 : 2
train_input_ids, train_token_type_ids, train_attention_mask, train_labels = input_ids[:train_len], token_type_ids[:train_len], attention_mask[:train_len], labels[:train_len]
val_input_ids, val_token_type_ids, val_attention_mask, val_labels = input_ids[train_len:], token_type_ids[train_len:], attention_mask[train_len:], labels[train_len:]

train_input = TensorDataset(train_input_ids, train_token_type_ids, train_attention_mask, train_labels)
val_input = TensorDataset(val_input_ids, val_token_type_ids, val_attention_mask, val_labels)
# batch
train_loader = DataLoader(train_input, batch_size = 10,shuffle = True)
val_loader = DataLoader(val_input, batch_size = 10,shuffle = True)

# vocab_size = # of total notes
# hidden_size = hidden_dim = num_attention_heads * attention layer(QKV) dim
# num_hidden_Layers = # of encoders
# intermediate_size = dim of feed forward
# max_position_embeddings = # total notes ()
# args: [vocab_size, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072,
# 		hidden_act="gelu", hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=16, initializer_range=0.02):
# https://github.com/google-research/bert/blob/master/modeling.py

# instrument_num = 128
num_labels = 5
config = BertConfig(vocab_size=369, hidden_size=369,
                    num_hidden_layers=12, num_attention_heads=12, intermediate_size=1024)

model = BMI(config, num_labels)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.config, lr=0.005, betas=(0.5, 0.999))

model.cuda()
criterion = criterion.cuda()
# input to cuda??

# train
num_epochs = 4
num_batches = len(train_loader)
best_val_loss = 10000.0
for epoch in range(num_epochs):

    model.train()
    tot_train_loss = 0.0
    for i, trainset in enumerate(train_loader):
        train_x, train_y = trainset
        # train_x, train_y = Variable(train_x), Variable(train_y)

        # use GPU
        train_x = train_x.cuda()
        train_y = train_y.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward(backprop) + optimize(weight update)
        output = model(input_ids=input_ids, input_mask=attention_mask, token_type_ids=token_type_ids)
        loss = criterion(output, train_y)
        loss.backward()
        optimizer.step()

        # loss for one epoch all batches
        tot_train_loss += loss.item()

        if (i + 1) % 100 == 0:  # validation of every 100 mini-batches

            with torch.no_grad():  # important!!! for validation
                tot_val_loss = 0.0
                for j, val_set in enumerate(val_loader):
                    val_x, val_y = val_set
                    # val_x, val_y = tf.Variable(val_x), tf.Variable(val_y)

                    val_x = val_x.cuda()
                    val_y = val_y.cuda()

                    val_output = model(input_ids=input_ids, input_mask=attention_mask, token_type_ids=token_type_ids)
                    val_loss = criterion(val_output, val_y)
                    tot_val_loss += val_loss.item()

                # save the model if validation accuracy improved
                if tot_val_loss < best_val_loss:
                    state = model.state_dict()
                    filename = "./models/model_valloss_" + str(val_loss)
                    print("=> Saving a new best")
                    torch.save(state, filename)  # save checkpoint
                    best_val_loss = tot_val_loss

            # print average train loss of 100 batch &
            print("epoch: {}/{} | step: {}/{} | train loss: {:.4f} | val loss: {:.4f}".format(
                epoch + 1, num_epochs, i + 1, num_batches, tot_train_loss / 100, tot_val_loss / len(val_loader)))

            tot_train_loss = 0.0
