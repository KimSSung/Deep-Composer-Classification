import py_midicsv
from os.path import *
import os
from tqdm import tqdm
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split

# path
PATH = './sunjong/midi820/'
# Max length of notes
MAX_LENGTH = 510
# genres
genres = ['Classical', 'Pop', 'Jazz', 'Country', 'Rock']
genre_num = 0 # label
# Load the MIDI file and parse it into CSV format
files = []
for genre in genres:
	dir_genre = PATH + genre
	for f in os.listdir(dir_genre):
		if isfile(join(dir_genre, f)) :
			new_path = dir_genre + '/' + f
			files.append((new_path, genre_num))

	genre_num += 1

# print(files)

notes = []
note_list = []
note_on_indices_list = []
index = 0
labels = []
attentions = []
token_types = []

for file, label in tqdm(files):
	try:
		csv_string = py_midicsv.midi_to_csv(file)
	except:
		print("Error file:", file)
	else:
		index = 0
		# CLS = 128, SEP = 129, PAD=130
		note = [128] # note of note_on
		note_on_indices = [] # index of note_on in csv_string
		num = 0
		attention = [1]
		token_type = [0]
		for string in csv_string:
			if len(note) == MAX_LENGTH and len(attention) == MAX_LENGTH and len(token_type) == MAX_LENGTH : break
			split = string.split(', ')

			# format: ( Track, Time, Note_on_c, Channel, Note, Velocity )
			if "Note_on_c" in string:
				# note_on_indices.append(index)
				note.append(int(split[4]))
				attention.append(1)
				token_type.append(0)
				# print(num)
				num += 1

			index += 1

		note.append(129)
		attention.append(1)
		token_type.append(0)
		while len(note) != (MAX_LENGTH + 1):
			note.append(130)
			attention.append(0)
			token_type.append(0)

		notes.append(note)
		attentions.append(attention)
		# note_on_indices_list.append(note_on_indices)
		labels.append(label)
		token_types.append(token_type)

# print(notes)
with open('note_on_800.pkl', 'wb') as f:
 	pickle.dump(notes, f)

print(attentions)
with open('attention_800.pkl', 'wb') as f:
 	pickle.dump(attentions, f)

with open('labels.pkl', 'wb') as f:
	pickle.dump(labels, f)

with open('token_type.pkl', 'wb') as f:
	pickle.dump(token_types, f)

a = np.asarray(notes)
print(a.shape)
print(len(notes))
print(len(attentions))
print(len(labels))
print(len(token_types))

# note_list = torch.FloatTensor(note_list)
# labels = torch.FloatTensor(labels)
# train_X, test_X, train_Y, train_Y = train_test_split(note_list, labels, test_size=0.2, shuffle=False)

# # only attack note_on ; change note_off after note_on changed
# # nn.Embedding(num_embeddings = len(vocab) = 128, emvbedding_dim = hyperparam = 20, padding_idx = 0(default))
# # Reference: https://wikidocs.net/64779
# # embedding_layer = nn.Embedding(128, 20, 0)

# # print(embedding_layer.weight)

# class Model(nn.Module):

# 	def __init__(self, note_num=128, emvbedding_dim, tot_note, num_genres):

# 		self.embeddings = nn.Embedding(note_num, embedding_dim) # [batch_size, tot_note_num, 20]
# 		# Conv1d : all num in each 20 dim vector get same weight ([first 20dim] * 1, [second 20dim]*2,...)
# 		# Conv2d : ex. [tot_num, 20] -> filter [5,5]
# 		self.linear1 = nn.Linear(tot_note * embedding_dim, 128)
# 		self.linear2 = nn.Linear(128, num_genres)
# 		# self.conv2 = nn.Conv1d(20, 16, 3)
# 		# self.pool1= nn.AvgPool1d(2)
# 		# self.conv3 = nn.Conv1d(16, 16, 3)
# 		# self.pool2= nn.AvgPool1d(2)
# 		# self.gru = nn.GRU(16, 16, 2, dropout=0.01)
# 		# self.out = nn.Linear(16, output_size)

# 	def forward(self, inputs):
# 		embeds = self.embeddings(inputs).view((1, -1)) # make [1, other] dim
# 		out = F.relu(self.linear1(embeds))
# 		out = self.linear2(out)
# 		return F.softmax(out)

# losses = []
# model = Model(128, 20, len(note), 5)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001)

# for epoch in range(10):
# 	total_loss = 0
# 	for 
