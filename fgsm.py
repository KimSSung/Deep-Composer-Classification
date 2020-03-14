import os
import h5py
import librosa
from librosa import display
import itertools
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Add, Dense, Activation, PReLU, Conv2D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau

from scipy.stats import mode


def splitsongs(X, y, window=0.05, overlap=0.5):
    # Empty lists to hold our results
    temp_X = []
    temp_y = []

    # Get the input song array size
    xshape = X.shape[0]
    chunk = int(xshape * window)
    offset = int(chunk * (1. - overlap))

    # Split the song and create new ones on windows
    spsong = [X[i:i + chunk] for i in range(0, xshape - chunk + offset, offset)]
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
                                                       hop_length=hop_length, n_mels=128)[:, :, np.newaxis]

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

#modify -> get only one of each genre!
def read_data(src_dir, genres, song_samples):
    # Empty array of dicts with the processed features from all files
    arr_fn = []
    arr_genres = []

    # Get file list from the folders
    for x, _ in genres.items():
        folder = src_dir + x #path to each genre directory
        for root, subdirs, files in os.walk(folder):
            for file in files: #visit only 1 file in each genre
                file_name = folder + "/" + file
                # Save the file name and the genre
                arr_fn.append(file_name)
                arr_genres.append(genres[x])
                break
            break
        break

    # # Split into train and test
    # X_train, X_test, y_train, y_test = train_test_split(
    #     arr_fn, arr_genres, test_size=0.3, random_state=42, stratify=arr_genres
    # )

    # Split into small segments and convert to spectrogram
    input, output = split_convert(arr_fn, arr_genres)

    return input, output


###############################################################################CODE BEGINS

PATH = '../../../../data/'
pretrained_model = keras.models.load_model(PATH + 'models/custom_cnn_2d/valloss_1.0265332298808627custom_cnn_2d.h5')
pretrained_model.summary()
print("==> MODEL LOADED")

gtzan_dir = PATH + 'GTZAN/genres/'
song_samples = 660000
genres = {'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4,
          'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9}
genre_list = ['metal', 'disco', 'classical', 'hiphop', 'jazz', 'country', 'pop', 'blues', 'reggae', 'rock']

# 1. Read the data
X,Y = read_data(gtzan_dir, genres, song_samples)
print("==> MEL SPECTOGRAM EXTRACTED:", X.shape, Y.shape)
X = tf.convert_to_tensor(X)
Y = tf.convert_to_tensor(Y)



#2. Create adv pattern / perturbations
loss_object = tf.keras.losses.CategoricalCrossentropy()
def create_adversarial_pattern(input, label):
    with tf.GradientTape() as tape:
        # all calc recorded in tape
        tape.watch(input)  # keep eye on "X"
        prediction = pretrained_model(input)
        loss = loss_object(Y, prediction)

    # Get the gradients of the loss w.r.t to the input image
    gradient = tape.gradient(loss, X)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad, prediction, loss

perturbations, prediction, original_loss = create_adversarial_pattern(X, Y)
cleaned_perturbations = tf.reshape(perturbations, [39,128,129])
print("==> GENERATED PERTURBATIONS")



# 3. Generate Attack
epsilons = [0, 0.01, 0.1, 0.12, 0.15]
ep_int = 3
def generate_adv_attack(eps):
    adv_x = X + eps*perturbations
    # adv_x = tf.clip_by_value(adv_x, 0, 1)
    return adv_x

adv_attack = generate_adv_attack(ep_int)
cleaned_adv_attack = tf.reshape(adv_attack, [39,128,129])
print("==> GENERATED ADV ATTACK")



# 4. Test Attack
orig_prediction = np.argmax(prediction, axis=1)
new_prediction = np.argmax(pretrained_model.predict(adv_attack), axis=1)
np_Y = np.argmax(Y, axis=1)
# new_score = pretrained_model.evaluate(adv_attack, Y, verbose=0)
# print(orig_prediction, new_prediction, np_Y)

print(  "==> RESULT\n",
        "- true label: ", genre_list[np_Y[0]],
        "\n- original prediction: ", genre_list[mode(orig_prediction).mode[0]],
        # "\nconfidence: ", ,
        # "\nloss: ",original_loss,

        "\n- adv attack prediction: ", genre_list[mode(new_prediction).mode[0]],
        # "\nconfidence = {:.3f}",
        # "\nloss = {:.3f}")
      )



# 5. Visualize
plt.figure(figsize=(10,4))
librosa.display.specshow(librosa.power_to_db(cleaned_perturbations[0], ref=np.max), x_axis='time', y_axis='mel', hop_length=256)
plt.colorbar(format = '%+2.0f dB')
string = "Mel-spectogram: Perturbation | ep=" + str(epsilons[ep_int])
plt.title(string)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,4))
librosa.display.specshow(librosa.power_to_db(cleaned_adv_attack[0], ref=np.max), x_axis='time', y_axis='mel', hop_length=256)
plt.colorbar(format = '%+2.0f dB')
string = "Mel-spectogram: Adversarial Attack | ep=" + str(epsilons[ep_int])
plt.title(string)
plt.tight_layout()
plt.show()










