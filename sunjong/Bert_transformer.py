import torch
import BMI
from transformers import BertConfig, optimization
from pytorch_pretrained_bert import BertConfig, optimization
import torch.nn.functional as F
import torch.optim as optim
import torch
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

from tqdm import tqdm
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
		df = df.drop(df.columns[500:1000], axis=1)
		tlist = df.values.tolist() #2d list
		rlist = rlist + tlist

	return rlist


def get_attention(genre_list):
	rlist = [] #list for return
	dir_genre = './newintinput/attention'
	for genre in genre_list:
		new_path = dir_genre + '/' + genre + ".pickle"
		df = pd.read_pickle(new_path)
		df = df.drop(df.columns[500:1000], axis=1)
		tlist = df.values.tolist() #2d list
		rlist = rlist + tlist

	return rlist


# def get_token(genre_list):
# 	rlist = []
# 	dir_genre = './newintinput/token'
# 	for genre in genre_list:
# 		new_path = dir_genre + '/' + genre + ".pickle"
# 		df = pd.read_pickle(new_path)
# 		df = df.drop(df.columns[500:1000], axis=1)
# 		tlist = df.values.tolist()  # 2d list
# 		rlist = rlist + tlist
#
# 	return rlist



###################################################################


# Converting midi into Bert's input -> !! list !!

genres = ['Classical', 'Jazz', 'Pop', 'Country','Rock']

input_ids = get_800(genres)
# token_type_ids = get_token(genres)
token_type_ids = [[0]*500]*4000
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
input_ids, token_type_ids, attention_mask, labels = torch.tensor(input_ids, dtype=torch.long), torch.tensor(token_type_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long), torch.tensor(labels, dtype=torch.long)  # tensor

train_len = int(len(input_ids) * 8 / 10)  # train : valid = 8 : 2
train_input_ids, train_token_type_ids, train_attention_mask, train_labels = input_ids[:train_len], token_type_ids[:train_len], attention_mask[:train_len], labels[:train_len]
val_input_ids, val_token_type_ids, val_attention_mask, val_labels = input_ids[train_len:], token_type_ids[train_len:], attention_mask[train_len:], labels[train_len:]



train_input = TensorDataset(train_input_ids, train_token_type_ids, train_attention_mask, train_labels)
val_input = TensorDataset(val_input_ids, val_token_type_ids, val_attention_mask, val_labels)
# batch
train_loader = DataLoader(train_input, batch_size = 5,shuffle = True)
val_loader = DataLoader(val_input, batch_size = 5,shuffle = True)

# vocab_size = # of total notes
# hidden_size = hidden_dim = num_attention_heads * attention layer(QKV) dim
# num_hidden_Layers = # of encoders
# intermediate_size = dim of feed forward
# max_position_embeddings = # total notes ()
# args: [vocab_size, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072,
# 		hidden_act="gelu", hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=16, initializer_range=0.02):
# https://github.com/google-research/bert/blob/master/modeling.py


num_labels = 5
config = BertConfig(vocab_size_or_config_json_file=369, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=1024)

model = BMI.BertForMidiClassification(config, num_labels)
criterion = nn.CrossEntropyLoss()
optimizer = optimization.BertAdam(model.parameters(), lr=0.005, t_total=3200)
# optimizer = BertAdam(model.parameters(), lr=lr, schedule='warmup_linear', warmup=warmup_proportion, num_training_steps=num_training_steps)
# optimizer = Adam(model.config, lr=0.005, betas=(0.5, 0.999))

#use GPU
model.cuda()
criterion = criterion.cuda()

# train
num_epochs = 4
num_batches = len(train_loader)
best_val_loss = 10000.0

result = {
    # 'epoch': [],
    'accuracy': [],
    'val_accuracy': [],
    'loss': [],
    'val_loss': []
}
# for i in range(num_epochs):
#     result['epoch'].append(i)


for epoch in tqdm(range(num_epochs)):

    model.train()
    tot_train_loss = 0.0
    tot_train_acc = 0.0
    total_correct = 0

    for i, trainset in enumerate(train_loader): #5(inputs each batch) x 640(batches)
        train_x_id, train_x_tok, train_x_att, train_y = trainset #unpack 4 items

        # train_x, train_y = Variable(train_x), Variable(train_y)

        # use GPU
        if torch.cuda.is_available():
            train_x_id = train_x_id.cuda()
            train_x_tok = train_x_tok.cuda()
            train_x_att = train_x_att.cuda()
            train_y = train_y.cuda()


        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward(backprop) + optimize(weight update)
        output = model(input_ids=train_x_id, token_type_ids=train_x_tok, attention_mask=train_x_att, labels=train_y)

        #ACCURACY
        pred = torch.argmax(F.softmax(output, dim=1)) #convert to prob that's sums upto 1 -> pick max value
        correct = pred.eq(train_y) #if label is correct
        total_correct += correct.sum().item()

        #LOSS
        loss = criterion(output, train_y)
        loss.backward()
        optimizer.step()

        #will output for each 100 mini batch
        tot_train_loss += loss.item()
        tot_train_acc += total_correct/len(train_y)


        # VALIDADTION of every 100 mini-batches
        if (i + 1) % 100 == 0:

            with torch.no_grad():  # important!!! for validation

                tot_val_loss = 0.0
                tot_val_acc = 0.0
                tot_val_correct = 0

                for j, val_set in enumerate(val_loader):
                    val_x_id, val_x_att, val_x_tok, val_y = val_set
                    # val_x, val_y = tf.Variable(val_x), tf.Variable(val_y)

                    val_x_id = val_x_id.cuda()
                    val_x_att = val_x_att.cuda()
                    val_x_tok = val_x_tok.cuda()
                    val_y = val_y.cuda()

                    val_output = model(input_ids=val_x_id, token_type_ids=val_x_tok, attention_mask=val_x_att, labels=val_y)

                    #ACCURACY
                    val_pred = torch.argmax(F.softmax(val_output, dim=1))  # convert to prob that's sums upto 1 -> pick max value
                    val_correct = val_pred.eq(val_y)  # if label is correct
                    tot_val_correct += val_correct.sum().item()

                    #LOSS
                    val_loss = criterion(val_output, val_y)
                    tot_val_loss += val_loss.item()

                # save the model if validation accuracy improved
                if tot_val_loss < best_val_loss:
                    state = model.state_dict()
                    filename = "./bert_models/model_valloss_" + str(val_loss)
                    print("=> Saving a new best")
                    torch.save(state, filename)  # save checkpoint
                    best_val_loss = tot_val_loss

            # print average train loss of 100 batch &
            print("epoch: {}/{} | step: {}/{} | train loss: {:.4f} | val loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}".format(
                epoch + 1, num_epochs,
                i + 1, num_batches,
                tot_train_loss / 100, #avg of 100 batches (500 inputs)
                tot_val_loss / len(val_loader),
                tot_train_acc / 100, #avg of 100 batches (500 inputs)
                tot_val_acc / len(val_loader)
                ))


            #append to dict => FOR GRAPH
            result['loss'].append(format(tot_train_loss/100,"10.2f"))
            result['val_loss'].append(format(tot_val_loss/len(val_loader),"10.2f"))
            result['accuracy'].append(format(tot_train_acc/100,"10.2f"))
            result['val_accuracy'].append(format(tot_train_acc/len(val_loader),"10.2f"))

            tot_train_loss = 0.0
            tot_train_acc = 0.0



# visualizing
# print(history.history.keys())

# Summarize history for accuracy
plt.plot(result['accuracy'])
plt.plot(result['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(result['loss'])
plt.plot(result['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
