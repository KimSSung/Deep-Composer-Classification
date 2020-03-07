import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt
from os.path import *
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm # show status bar of for

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras import optimizers

WAV_PATH = './wav820/'
SAMPLING_RATE = 44100
MFCC_NUM = 20
MFCC_MAX_LEN = 2000


MODEL_SAVE_FOLDER_PATH = './melfmodel/'

if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
  os.mkdir(MODEL_SAVE_FOLDER_PATH)

model_path = MODEL_SAVE_FOLDER_PATH + 'melfreq-' + '{epoch:02d}-{val_loss:.4f}.hdf5'

# Save the model after every epoch
cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss',
                                verbose=1, save_best_only=True)

# Stop training when performance goes down
# cb_early_stopping = EarlyStopping(monitor='val_loss', patience=10)


def wav2mfcc(wave, max_len=MFCC_MAX_LEN):
	mfcc = librosa.feature.mfcc(wave, n_mfcc=MFCC_NUM, sr=SAMPLING_RATE)

	# if max length exceeds mfcc lengths then pad the remaining ones
	if (max_len > mfcc.shape[1]):
		pad_width = max_len - mfcc.shape[1]
		# mode=constant : init by 0
		# ((0,0), (0,pad_width)) -> 0 row 0 , 0 column pad_width : only add to column
		mfcc = np.pad(mfcc, pad_width((0,0), (0,pad_width)), mode='constant')

	# else cutoff the remaining parts
	else:
		mfcc = mfcc[:,:max_len]

	return mfcc

# Make Dataset
X, y = [], []
def datasetXy(label, wave):
	y.append(label)
	mfcc = wav2mfcc(wave) # (20, 2000)
	X.append(mfcc)

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
y_hot = to_categorical(y) # not train label 5, but train [0,0,0,0,1] (prob of label 5 to 1)

X_train, X_test, y_train, y_test = train_test_split(X, y_hot, test_size=0.2, random_state=True, shuffle=True)

# Feature dimension & other options
feature_dim_1 = MFCC_NUM # 20
feature_dim_2 = MFCC_MAX_LEN # 2000
channel = 1 # each pixel has only db . ex) if each has RGB, channel = 3
epochs = 100
batch_size = 100
verbose = 1
num_classes = 5

# Reshaping dataset to perform 2D conv (batch_size, dim1, dim2, channel)
# X_train.shape[0] -> maybe batch_size
X_train = X_train.reshape(X_train.shape[0], feature_dim_1, feature_dim_2, channel)
X_test = X_test.reshape(X_train.shape[0], feature_dim_1, feature_dim_2, channel)

y_train_hot = y_train
y_test_hot = y_test

def Model():
	# with uncomment -> AlexNet
	# some paper said, simple model is better because of overfitting

	model = Sequential()
	model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(feature_dim_1, feature_dim_2, channel)))
	#model.add(MaxPooling2D(pool_size=(2, 2)))

	#model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	
	#model.add(Conv2D(120, kernel_size=(2, 2), activation='relu'))
	model.add(Conv2D(48, kernel_size=(2, 2), activation='relu')) # if Alexnet, 48 -> 120
	model.add(Conv2D(120, kernel_size=(2, 2), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.4))
	#model.add(Dense(64, activation='relu'))
	#model.add(Dropout(0.4))
	model.add(Dense(num_classes, activation='softmax'))
	
	return model

model = Model()

# optimizer = optimizers.SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True)
# optimizer = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
optimizer = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False) # default decay = 0.0

# metrics: List of metrics to be evaluated by the model during training and testing
model.compile(loss=keras.losses.categorical_crossentropy,
				optimizer=optimizer, metrics=['accuracy']) 
# train & test
history = model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs,
			verbose=verbose, validation_data=(X_test, y_test_hot), callbacks=[cb_checkpoint]) # callbacks added except cb_early_stopping


y_vloss = history.history['val_loss'] #val loss
y_loss = history.history['loss'] #train loss
y_vacc = history.history['val_acc'] #val accuracy
y_acc = history.history['acc'] #train accuracy


# visualizing
x_len = np.arange(len(y_loss))


fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")
loss_ax.plot(x_len, y_vloss, marker='.', c='red', label="Validation-set Loss")
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')


acc_ax.plot(x_len, y_acc, marker='.', c='green', label="Train-set Accuracy")
acc_ax.plot(x_len, y_vacc, marker='.', c='yellow', label="Validation-set Accuracy")
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper right')

epo = 'epoch'
epo = epo + '[' + str(epoch) + ']'
plt.xlabel(epo)

opt = 'opt: '
opt = opt + optimizer
act = 'act: '
for each in activations:
	act = act + '-' + each
opt = opt + '\n' + act
plt.title(opt, loc='left')
total_acc = 'val_acc: {:.4f}'.format(model.evaluate(X_validation, Y_validation, verbose=1)[1])
total_acc = total_acc + '\ntrain_acc: {:.4f}'.format(model.evaluate(X_train, Y_train, verbose=1)[1])
plt.title(total_acc, loc='right')
plt.grid()
plt.show()

# plt.legend(loc='upper right')
# plt.grid()

# plt.ylabel('loss')
# plt.show()


plt.legend(loc='upper right')


# -----------------------------test----------------------------------------
# wave, sr = librosa.load(TEST_WAV_PATH, mono=True, sr=None)
# mfcc = wav2mfcc(wave)
# X_test = mfcc.reshape(1, feature_dim_1, feature_dim_2, channel)
# preds = model.predict(X_test)[0]