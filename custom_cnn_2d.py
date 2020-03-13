# https://github.com/Hguimaraes/gtzan.keras/blob/master/nbs/1.1-custom_cnn_2d.ipynb

import os
import h5py
import librosa
import itertools
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Add, Dense, Activation, PReLU, Conv2D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau


# For reproducibility purposes
np.random.seed(42)

# path
PATH = '../../../../data/'

#########################################
############### Read Data ###############


"""
@description: Method to split a song into multiple songs using overlapping windows
"""
def splitsongs(X, y, window = 0.05, overlap = 0.5):
	# Empty lists to hold our results
	temp_X = []
	temp_y = []

	# Get the input song array size
	xshape = X.shape[0]
	chunk = int(xshape*window)
	offset = int(chunk*(1.-overlap))
	
	# Split the song and create new ones on windows
	spsong = [X[i:i+chunk] for i in range(0, xshape - chunk + offset, offset)]
	for s in spsong:
		if s.shape[0] != chunk:
			continue

		temp_X.append(s)
		temp_y.append(y)

	return np.array(temp_X), np.array(temp_y)


"""
@description: Method to convert a list of songs to a np array of melspectrograms
"""
def to_melspectrogram(songs, n_fft=1024, hop_length=256):
	# Transformation function
	melspec = lambda x: librosa.feature.melspectrogram(x, n_fft=n_fft,
		hop_length=hop_length, n_mels=128)[:,:,np.newaxis]

	# map transformation of input songs to melspectrogram using log-scale
	tsongs = map(melspec, songs)
	# np.array([librosa.power_to_db(s, ref=np.max) for s in list(tsongs)])
	return np.array(list(tsongs))


def split_convert(X, y):
	arr_specs, arr_genres = [], []
	
	# Convert to spectrograms and split into small windows
	for fn, genre in zip(X, y):
		signal, sr = librosa.load(fn)
		signal = signal[:song_samples]

		# Convert to dataset of spectograms/melspectograms
		signals, y = splitsongs(signal, genre)

		# Convert to "spec" representation
		specs = to_melspectrogram(signals)

		# Save files
		arr_genres.extend(y)
		arr_specs.extend(specs)
	
	return np.array(arr_specs), to_categorical(arr_genres)


def read_data(src_dir, genres, song_samples):    
	# Empty array of dicts with the processed features from all files
	arr_fn = []
	arr_genres = []

	# Get file list from the folders
	for x,_ in genres.items():
		folder = src_dir + x
		for root, subdirs, files in os.walk(folder):
			for file in files:
				file_name = folder + "/" + file

				# Save the file name and the genre
				arr_fn.append(file_name)
				arr_genres.append(genres[x])
	
	# Split into train and test
	X_train, X_test, y_train, y_test = train_test_split(
		arr_fn, arr_genres, test_size=0.3, random_state=42, stratify=arr_genres
	)
	
	# Split into small segments and convert to spectrogram
	X_train, y_train = split_convert(X_train, y_train)
	X_test, y_test = split_convert(X_test, y_test)

	return X_train, X_test, y_train, y_test

# Parameters
gtzan_dir = PATH + 'GTZAN/genres/'
song_samples = 660000
genres = {'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4, 
		  'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9}

# Read the data
X_train, X_test, y_train, y_test = read_data(gtzan_dir, genres, song_samples)


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# Histogram for train and test 
values, count = np.unique(np.argmax(y_train, axis=1), return_counts=True)
plt.bar(values, count)

values, count = np.unique(np.argmax(y_test, axis=1), return_counts=True)
plt.bar(values, count)
plt.show()


##############################################################
############### GTZAN Melspectrogram Generator ###############

from tensorflow.keras.utils import Sequence

class GTZANGenerator(Sequence):
	def __init__(self, X, y, batch_size=64, is_test = False):
		self.X = X
		self.y = y
		self.batch_size = batch_size
		self.is_test = is_test
	
	def __len__(self):
		return int(np.ceil(len(self.X)/self.batch_size))
	
	def __getitem__(self, index):
		# Get batch indexes
		signals = self.X[index*self.batch_size:(index+1)*self.batch_size]

		# Apply data augmentation
		if not self.is_test:
			signals = self.__augment(signals)
		return signals, self.y[index*self.batch_size:(index+1)*self.batch_size]
	
	def __augment(self, signals, hor_flip = 0.5, random_cutout = 0.5):
		spectrograms =  []
		for s in signals:
			signal = copy(s)
			
			# Perform horizontal flip
			if np.random.rand() < hor_flip:
				signal = np.flip(signal, 1)

			# Perform random cutoout of some frequency/time
			if np.random.rand() < random_cutout:
				lines = np.random.randint(signal.shape[0], size=3)
				cols = np.random.randint(signal.shape[0], size=4)
				signal[lines, :, :] = -80 # dB
				signal[:, cols, :] = -80 # dB

			spectrograms.append(signal)
		return np.array(spectrograms)
	
	def on_epoch_end(self):
		self.indexes = np.arange(len(self.X))
		np.random.shuffle(self.indexes)
		return None


##############################################################
############### Custom CNN (Melspectogram) ###################

def conv_block(x, n_filters, pool_size=(2, 2)):
    x = Conv2D(n_filters, (3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=pool_size, strides=pool_size)(x)
    x = Dropout(0.25)(x)
    return x

# Model Definition
def create_model(input_shape, num_genres):
    inpt = Input(shape=input_shape)
    x = conv_block(inpt, 16)
    x = conv_block(x, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 256)
    
    # Global Pooling and MLP
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', 
              kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)
    x = Dropout(0.25)(x)
    predictions = Dense(num_genres, 
                        activation='softmax', 
                        kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)
    
    model = Model(inputs=inpt, outputs=predictions)
    return model


model = create_model(X_train[0].shape, 10)

model.summary()

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

reduceLROnPlat = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.95,
    patience=3,
    verbose=1,
    mode='min',
    min_delta=0.0001,
    cooldown=2,
    min_lr=1e-5
)

# Generators
batch_size = 128
train_generator = GTZANGenerator(X_train, y_train)
steps_per_epoch = np.ceil(len(X_train)/batch_size)

validation_generator = GTZANGenerator(X_test, y_test)
val_steps = np.ceil(len(X_test)/batch_size)

hist = model.fit( # fit_generator deprecated -> model.fit support generator fot first argument
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=val_steps,
    epochs=150,
    verbose=1,
    callbacks=[reduceLROnPlat])


score = model.evaluate(X_test, y_test, verbose=0)
print("val_loss = {:.3f} and val_acc = {:.3f}".format(score[0], score[1]))

plt.figure(figsize=(15,7))

plt.subplot(1,2,1)
plt.plot(hist.history['accuracy'], label='train')
plt.plot(hist.history['val_accuracy'], label='validation')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='validation')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
# plt.show()
plt.savefig('valacc_' + str(score[1]) + 'acc_loss.png', dpi=300)



##############################################################
##################plot genre label predictions################


#http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


preds = np.argmax(model.predict(X_test), axis = 1)
y_orig = np.argmax(y_test, axis = 1)
cm = confusion_matrix(preds, y_orig)

keys = OrderedDict(sorted(genres.items(), key=lambda t: t[1])).keys()

plt.figure(figsize=(10,10))
plot_confusion_matrix(cm, keys, normalize=True)


#########################################
###############Save model################

# Save the model
model.save(PATH + 'models/custom_cnn_2d/valloss_'+str(score[0])+'custom_cnn_2d.h5')
