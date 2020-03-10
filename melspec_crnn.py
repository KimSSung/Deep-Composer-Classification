import librosa
import librosa.display
# import IPython.display as ipd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from os.path import *
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm # show status bar of for

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Bidirectional, LSTM, Dropout, Activation, GRU
from keras.layers import Conv2D, concatenate, MaxPooling2D, Flatten, Embedding, Lambda
from keras.callbacks import ModelCheckpoint, EarlyStopping #, LambdaCallback
from keras import backend as K
from keras.utils import to_categorical
from keras import optimizers, regularizers


WAV_PATH = './wav820/'
SAMPLING_RATE = 16000
MFCC_NUM = 40 # n_mels for spectogram
MFCC_MAX_LEN = 1000


MODEL_SAVE_FOLDER_PATH = './melfmodel/CRNN_RMS_0.0005'

if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
  os.mkdir(MODEL_SAVE_FOLDER_PATH)

model_path = MODEL_SAVE_FOLDER_PATH + 'melfreq-' + '{epoch:02d}-{val_loss:.4f}.hdf5'

# Save the model after every epoch
cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss',
								verbose=1, save_best_only=True)

# Stop training when performance goes down
# cb_early_stopping = EarlyStopping(monitor='val_loss', patience=10)

def wav2mfcc(wave, max_len=MFCC_MAX_LEN):
	# mfcc = librosa.feature.mfcc(wave, n_mfcc=MFCC_NUM, sr=SAMPLING_RATE)
	mfcc = librosa.feature.mfcc(wave, n_mfcc=MFCC_NUM, sr=SAMPLING_RATE)
	# print(mfcc.shape)

	# if max length exceeds mfcc lengths then pad the remaining ones
	if (max_len > mfcc.shape[1]):
		pad_width = max_len - mfcc.shape[1]
		# mode=constant : init by 0
		# ((0,0), (0,pad_width)) -> 0 row 0 , 0 column pad_width : only add to column
		mfcc = np.pad(mfcc, pad_width = ((0,0), (0,pad_width)), mode='constant')

	# else cutoff the remaining parts
	else:
		mfcc = mfcc[:,:max_len]

	return mfcc

def wav2melspec(wave, max_len=MFCC_MAX_LEN):
	
	melspec = librosa.feature.melspectrogram(y=wave, sr=SAMPLING_RATE, n_mels=MFCC_NUM)

	# if max length exceeds mfcc lengths then pad the remaining ones
	if (max_len > melspec.shape[1]):
		pad_width = max_len - melspec.shape[1]
		# mode=constant : init by 0
		# ((0,0), (0,pad_width)) -> 0 row 0 , 0 column pad_width : only add to column
		melspec = np.pad(melspec, pad_width = ((0,0), (0,pad_width)), mode='constant')

	# else cutoff the remaining parts
	else:
		melspec = melspec[:,:max_len]

	return melspec


# Make Dataset
X, y = [], []
def datasetXy(label, wave):
	y.append(label)
	melspec = wav2melspec(wave) # (20, 2000)
	melspec = melspec.tolist()
	X.append(melspec) # append list to list
	# print(melspec.shape)


files = []
genres = ['Classical', 'Jazz', 'Pop', 'Rock', 'Country']
genre_num = 0
for genre in genres:
	dir_genre = WAV_PATH + genre
	for f in os.listdir(dir_genre):
		if isfile(join(dir_genre, f)) :
			new_path = dir_genre + '/' + f
			files.append((new_path, genre_num))

	genre_num += 1

# mode
mode = 'load'

if mode == 'save':
	for file in tqdm(files):
		wave, sr = librosa.load(file[0], sr=SAMPLING_RATE, mono=True, duration=30.0)
		# wave = wave[::3] # audio downsampling
		datasetXy(file[1], wave)

	X = np.stack(X, axis=0) # vertical stack
	y = np.array(y)
	print("X shape is:", X.shape)
	print("y shape is:", y.shape)
	with open('num40_len1000_melspec_X.pkl', 'wb') as f:
		pickle.dump(X, f)
	with open ('num40_len1000_melspec_y.pkl', 'wb') as t:
		pickle.dump(y, t)
	print("save success...")


elif mode == 'load':
	with open('num40_len1000_melspec_X.pkl', 'rb') as f:
		X = pickle.load(f)
		print(X.shape)
	with open ('num40_len1000_melspec_y.pkl', 'rb') as t:
		y = pickle.load(t)
	print("load success...")


y_hot = to_categorical(y) # not train label 5, but train [0,0,0,0,1] (prob of label 5 to 1)

X_train, X_test, y_train, y_test = train_test_split(X, y_hot, test_size=0.2, random_state=True, shuffle=True)
# print(X_train.shape)

# Feature dimension & other options
feature_dim_1 = MFCC_NUM # 128
feature_dim_2 = MFCC_MAX_LEN # 
channel = 1 # each pixel has only db . ex) if each has RGB, channel = 3
# epochs = 100
# batch_size = 80
verbose = 1
num_classes = 5

# Reshaping dataset to perform 2D conv (tot data, dim1, dim2, channel)
X_train = X_train.reshape(X_train.shape[0], feature_dim_1, feature_dim_2, channel)
X_test = X_test.reshape(X_test.shape[0], feature_dim_1, feature_dim_2, channel)

y_train_hot = y_train
y_test_hot = y_test

# -----------------------------------------------------------------------------
# model
nb_filters1=16 
nb_filters2=32 
nb_filters3=64
nb_filters4=64
nb_filters5=64
ksize = (3,1)
pool_size_1= (2,2) 
pool_size_2= (4,4)
pool_size_3 = (4,2)

dropout_prob = 0.20
dense_size1 = 128
lstm_count = 64
num_units = 120

BATCH_SIZE = 64
EPOCH_COUNT = 20
L2_regularization = 0.001

def CRNN_model(model_input):
	# ------------Memo----------------
	# Parallel CNN-RNN Model
	# Use 'Functional API', not 'Sequential model'
	# Functional API is useful for complex model (ex. CRNN)
	# https://keras.io/ko/getting-started/functional-api-guide/
	# Each return 'Tensor', and layer instance are callable by tensor

	print('Building model...')
	
	layer = model_input
	
	### Convolutional blocks

	conv_1 = Conv2D(filters = nb_filters1, kernel_size = ksize, strides=1,
					  padding= 'valid', activation='relu', name='conv_1')(layer) # layer, conv_1 = Tensor
	pool_1 = MaxPooling2D(pool_size_1)(conv_1)

	conv_2 = Conv2D(filters = nb_filters2, kernel_size = ksize, strides=1,
					  padding= 'valid', activation='relu', name='conv_2')(pool_1)
	pool_2 = MaxPooling2D(pool_size_1)(conv_2)

	conv_3 = Conv2D(filters = nb_filters3, kernel_size = ksize, strides=1,
					  padding= 'valid', activation='relu', name='conv_3')(pool_2)
	pool_3 = MaxPooling2D(pool_size_1)(conv_3)
		
	# conv_4 = Conv2D(filters = nb_filters4, kernel_size = ksize, strides=1,
	# 				  padding= 'valid', activation='relu', name='conv_4')(pool_3)
	# pool_4 = MaxPooling2D(pool_size_2)(conv_4)
	
	
	# conv_5 = Conv2D(filters = nb_filters5, kernel_size = ksize, strides=1,
	# 				  padding= 'valid', activation='relu', name='conv_5')(pool_4)
	# pool_5 = MaxPooling2D(pool_size_2)(conv_5)

	flatten1 = Flatten()(pool_3)


	### Recurrent Block

	# Pooling layer
	pool_lstm1 = MaxPooling2D(pool_size_3, name = 'pool_lstm')(layer)
	
	# Embedding layer

	squeezed = Lambda(lambda x: K.squeeze(x, axis= -1))(pool_lstm1)
#     flatten2 = K.squeeze(pool_lstm1, axis = -1)
#     dense1 = Dense(dense_size1)(flatten)
	
	# Bidirectional GRU
	lstm = Bidirectional(GRU(lstm_count))(squeezed)  #default merge mode is concat
	
	# Concat Output
	concat = concatenate([flatten1, lstm], axis=-1, name ='concat')
	
	## Softmax Output
	output = Dense(num_classes, activation = 'softmax', name='preds')(concat)
	
	model_output = output

	model = Model(model_input, model_output) # keras model

	print(model.summary())

	return model

input_shape = (feature_dim_1, feature_dim_2, channel) # (n_frames, n_frequency, 1)
model_input = Input(input_shape, name='input') # Input layer return tensor. Input(shape = input_shape)
model = CRNN_model(model_input)

# optimizer = optimizers.Adam(lr=0.001) # beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False
optimizer = optimizers.RMSprop(lr=0.0005)  # Optimizer

# metrics: List of metrics to be evaluated by the model during training and testing
model.compile(loss=keras.losses.categorical_crossentropy,
				optimizer=optimizer, metrics=['accuracy']) 
# train & test
history = model.fit(X_train, y_train_hot, batch_size=BATCH_SIZE, epochs=EPOCH_COUNT,
			verbose=1, validation_data=(X_test, y_test_hot), callbacks=[cb_checkpoint]) # callbacks added except cb_early_stopping


# visualizing
print(history.history.keys())

# Summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



# -----------------------------test----------------------------------------
# wave, sr = librosa.load(TEST_WAV_PATH, mono=True, sr=None)
# mfcc = wav2mfcc(wave)
# X_test = mfcc.reshape(1, feature_dim_1, feature_dim_2, channel)
# preds = model.predict(X_test)[0]