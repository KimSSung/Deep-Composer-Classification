'''
import pandas as pd
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

# read pickle -> make input & output
genres = ['Classical', 'Jazz', 'Pop', 'Country', 'Rock']
genre_num = 0

# empty dataframe
column_list = ['file name', 'file size', 'song length', 'note length avg', 'max pitch',
				'min pitch', 'max pitch diff', 'min pitch diff', 'pitch avg', 'instruments', 'major', 'key']
input_tensor = pd.DataFrame(columns=column_list)
labels = np.array([])

for genre in genres:
	load_df = pd.read_pickle('./pickles/D_'+ genre + '.pickle')
	# remove './midi820/genre/' from filename
	load_df['file name'] = load_df['file name'].str.replace(pat='./midi820/' + genre + '/', repl='', regex=False)
	# remove 'note length avg == 0'
	idx = load_df[load_df['note length avg'] == 0].index
	load_df = load_df.drop(idx)

	print(len(load_df))

	# labels
	df_length = len(load_df)
	this_label = np.full((df_length), genre_num)
	labels = np.hstack([labels, this_label])

	# print(len(labels))

	input_tensor = pd.concat([input_tensor, load_df], axis = 0)

	# print(len(input_tensor))

	genre_num += 1 # label

output_tensor = np_utils.to_categorical(labels)

print(input_tensor.shape) # (4052, 12)
print(output_tensor.shape) # (4052, 5)

# # train, valid split after shuffle
# X_train, X_validation, Y_train, Y_validation = train_test_split(input_tensor, output_tensor, test_size = 0.3, random_state = None)

# print(X_train)
# print(Y_train)

print(input_tensor.isnull().any())

'''
'''
import random
import torch
import numpy as np

a = [[0,1,2], [3,4,5], [6,7,8], [9,10,11]]
b = [[1,1,1], [2,2,2], [3,3,3], [4,4,4]]
c = [[4,5,6], [7,8,9], [10,11,12], [13,14,15]]
d = [100, 101, 102, 103]

e = list(zip(a,b,c,d))

random.shuffle(e)

t = e[:3]
v = e[3:]

a1,b1,c1,d1 = zip(*t)
a2,b2,c2,d2 = zip(*v)

print(a1)
print(b1)
print(c1)
print(d1)

print(a2)
print(b2)
print(c2)
print(d2)

print(type(a1)) # tuple
print(type(a2))

a1, a2 = np.array(a1), np.array(a2)

tensor1, tensor2 = torch.tensor(a1), torch.tensor(a2)
print(tensor1)
print(tensor2)

print(tensor1[:2])
print(tensor2[0])
'''
'''
trainset = data[:train_len]
validset = data[train_len:]

train_input_ids, train_token_type_ids, train_attention_mask, train_labels = zip(*trainset) # tuples
valid_input_ids, valid_token_type_ids, valid_attention_mask, valid_labels = zip(*validset)

train_input_ids, train_token_type_ids, train_attention_mask, train_labels =
			np.array(train_input_ids), np.array(train_token_type_ids), np.array(train_attention_mask), np.array(train_labels)

valid_input_ids, valid_token_type_ids, valid_attention_mask, valid_labels =
			np.array(valid_input_ids), np.array(valid_token_type_ids), np.array(valid_attention_mask), np.array(valid_labels)

train_input_ids, train_token_type_ids, train_attention_mask, train_labels =
			torch.tensor(train_input_ids), torch.tensor(train_token_type_ids), torch.tensor(train_attention_mask), torch.tensor(train_labels)

valid_input_ids, valid_token_type_ids, valid_attention_mask, valid_labels =
			torch.tensor(valid_input_ids), torch.tensor(valid_token_type_ids), torch.tensor(valid_attention_mask), torch.tensor(valid_labels)


'''
from os.path import *
from os import *
from tqdm import tqdm # show status bar of for
import librosa
import numpy as np


wav_file = './wav820/Classical/Winter wind etude.wav'
wav_file2 = './wav820/Rock/Yes - Cinema.wav'

'''
files = []
genres = ['Classical', 'Jazz', 'Pop', 'Rock', 'Country']
genre_num = 0
for genre in genres:
	dir_genre = WAV_PATH + genre
	for f in listdir(dir_genre):
		if isfile(join(dir_genre, f)) :
			new_path = dir_genre + '/' + f
			files.append((new_path, genre_num))

	genre_num += 1

for file in tqdm(files):
	wave, sr = librosa.load(file[0], sr=SAMPLING_RATE, mono=True, duration=30.0)
	# wave = wave[::3] # audio downsampling
	datasetXy(file[1], wave)

X = np.array(X)
y = np.array(y)
'''

frame_length = 0.025
frame_stride = 0.010

def Mel_S(wav_file):
	# mel-spectrogram
	y, sr = librosa.load(wav_file, mono=True, duration=20.0)

	# wav_length = len(y)/sr
	input_nfft = int(round(sr*frame_length))
	input_stride = int(round(sr*frame_stride))

	S = librosa.feature.melspectrogram(y=y, sr=16000)
	print(S.shape)
	print("Wav length: {}, Mel_S shape:{}".format(len(y)/sr,np.shape(S)))

	P = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
	print(P.shape)

Mel_S(wav_file)
Mel_S(wav_file2)