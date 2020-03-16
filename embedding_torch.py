import py_midicsv
from os import *
from os.path import *
from tqdm import tqdm

import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten
from keras import optimizers
from sklearn.model_selection import train_test_split


# path
PATH = './sunjong/midi820/'
# Max length of notes
MAX_LEN = 1000
# genres
genres = ['Classical', 'Pop', 'Jazz', 'Country', 'Rock']
genre_num = 0 # label
# Load the MIDI file and parse it into CSV format
files = []
for genre in genres:
	dir_genre = PATH + genre
	for f in listdir(dir_genre):
		if isfile(join(dir_genre, f)) :
			new_path = dir_genre + '/' + f
			files.append((new_path, genre_num))

	genre_num += 1

note_list = []
note_on_indices_list = []
index = 0
labels = []
file_num = 0 # to print error position
for file, label in tqdm(files):

	try:
		csv_string = py_midicsv.midi_to_csv(file)
	except:
		# print(file_num, 'th Error file:', file)
		continue
	else:
		index = 0
		note = [] # note of note_on
		note_on_indices = [] # index of note_on in csv_string

		for string in csv_string:
			split = string.split(', ')

			# format: ( Track, Time, Note_on_c, Channel, Note, Velocity )
			if "Note_on_c" in string:
				note_on_indices.append(index)
				note.append(int(split[4]))
				if len(note) > MAX_LEN:
					break

			index += 1

		note_list.append(note)
		note_on_indices_list.append(note_on_indices)
		labels.append(label)
		# print('Success!')

	file_num += 1

note_list = torch.FloatTensor(note_list)
labels = torch.FloatTensor(labels)
train_X, test_X, train_Y, train_Y = train_test_split(note_list, labels, test_size=0.2, shuffle=False)


def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = pretrained_model(input_image)
    loss = loss_object(input_label, prediction)


num_genres = 5
model = Sequential()
model.add(Embedding(128, 20, input_length=MAX_LEN))
model.add(Flatten())
model.add(Dense(num_genres, activation='softmax'))

loss = keras.losses.CategoricalCrossentropy()
optimizer = optimizers.SGD(lr=0.001)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model.fit(train_X, train_y, epochs=10, verbose=1)

