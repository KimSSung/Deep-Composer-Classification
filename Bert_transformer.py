import torch
import BMI
from transformers import BertConfig
# import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
import os
import random

from mido import MidiFile

def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return  param_group['lr']

# make input list
# count_file = 0
# genres = ['Classical', 'Jazz', 'Pop', 'Country', 'Rock']
# files = []
# for genre in genres:
# 	dir_genre = './midi820/' + genre
# 	for f in listdir(dir_genre):
# 		if isfile(join(dir_genre, f)) :
# 			new_path = dir_genre + '/' + f
# 			files.append(new_path)

# labels = np.array([])
# for file in files:
# 	try:
# 		count_file += 1
# 		mid = MidiFile(file)
	
# 	except:
# 		print('error opening ',count_file,',th file | file name: \"',files[count_file-1],'\"')
	
# 	else:

# 		midi_notes = []
# 		for i, track in enumerate(mid.tracks):
# 			# print('Track {}: {}'.format(i, track.name))
# 			for msg in track:
# 				if (not msg.is_meta) and (msg.type == 'note_off' or msg.type == 'note_on'):
# 					midi_notes.append(msg.bytes()[1])

# 		# labels
# 		length = len(midi_notes)
# 		this_label = np.full((length), genre_num)
# 		labels = np.hstack([labels, this_label])	

# Converting midi into Bert's input -> !! list !!
input_ids = 
token_type_ids = 
attention_mask = 
labels = 

data = list(zip(input_ids, token_type_ids, attention_mask, labels))
random.shuffle(data)

input_ids, token_type_ids, attention_mask, labels = zip(*data) # tuples

input_ids, token_type_ids, attention_mask, labels =
			np.array(input_ids), np.array(token_type_ids), np.array(attention_mask), np.array(labels) # tuple -> ndarray

input_ids, token_type_ids, attention_mask, labels =
			torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), torch.tensor(labels) # tensor

train_len = len(input_ids) * 8 / 10 # train : valid = 8 : 2

train_input_ids, train_token_type_ids, train_attention_mask, train_labels =
			input_ids[:train_len], token_type_ids[:train_len], attention_mask[:train_len], labels[:train_len]

valid_input_ids, valid_token_type_ids, valid_attention_mask, valid_labels =
			input_ids[train_len:], token_type_ids[train_len:], attention_mask[train_len:], labels[train_len:]

# batch
train_loader = 
valid_loader = 

# vocan_size = # of total notes
# hidden_size = hidden_dim = num_attention_heads * attention layer(QKV) dim
# num_hidden_Layers = # of encoders
# intermediate_size = dim of feed forward
# max_position_embeddings = # total notes ()
# args: [vocab_size, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072,
# 		hidden_act="gelu", hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=16, initializer_range=0.02):
# https://github.com/google-research/bert/blob/master/modeling.py
instrument_num = 128
num_labels = 5
config = BertConfig(vocab_size=(388 * instrument_num)+3, hidden_size=(388 * instrument_num)+3,
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
best_valid_loss = 10000.0
for epoch in range(num_epochs):

	model.train()
	tot_train_loss = 0.0
	for i, trainset in enumerate(train_loader):
		train_x, train_y = trainset
		train_x, train_y = Variable(train_x), Variable(train_y)

		# use GPU
		train_x = train_x.cuda()
		train_y = train_y.cuda()

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward(backprop) + optimize(weight update)
		output = model(input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)
		loss = criterion(output, train_y)
		loss.backward()
		optimizer.step()

		# loss for one epoch all batches
		tot_train_loss += loss.item()

		if (i+1) % 100 == 0: # validation of every 100 mini-batches

			with torch.no_grad(): # important!!! for validation
				tot_val_loss = 0.0
				for j, valset in enumerate(valid_loader):
					val_x, val_y = valset
					val_x, val_y = Variable(val_x), Variable(val_y)

					val_x = val_x.cuda()
					val_y = val_y.cuda()

					val_output = model(input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)
					val_loss = criterion(val_output, val_y)
					tot_val_loss += val_loss.item()


				# save the model if validation accuracy improved
				if tot_val_loss < best_valid_loss:
					state = model.state_dict()
					filename = "./models/model_valloss_" + str(valid_loss)
					print ("=> Saving a new best")
					torch.save(state, filename)  # save checkpoint
					best_valid_loss = tot_val_loss

			# print average train loss of 100 batch & 
			print("epoch: {}/{} | step: {}/{} | train loss: {:.4f} | val loss: {:.4f}".format(
				epoch+1, num_epochs, i+1, num_batches, tot_train_loss / 100, tot_val_loss / len(val_loader)))

			tot_train_loss = 0.0