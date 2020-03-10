from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, LambdaCallback
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

MODEL_SAVE_FOLDER_PATH = './model/'

if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
  os.mkdir(MODEL_SAVE_FOLDER_PATH)

model_path = MODEL_SAVE_FOLDER_PATH + 'midiclass-' + '{epoch:02d}-{val_loss:.4f}.hdf5'

# Save the model after every epoch
cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss',
                                verbose=1, save_best_only=True)

# Stop training when performance goes down
cb_early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# print weights
print_weights = LambdaCallback(on_epoch_end=lambda epoch, logs: print(model.layers[0].get_weights()))


# read pickle -> make input & output
genres = ['Classical', 'Jazz', 'Pop', 'Country', 'Rock']
genre_num = 0

# empty dataframe
column_list = ['file name', 'file size', 'song length', 'note length avg', 'max pitch',
				'min pitch', 'max pitch diff', 'min pitch diff', 'pitch avg', 'instruments', 'major','key']
input_tensor = pd.DataFrame(columns=column_list)
labels = np.array([])

for genre in genres:
	load_df = pd.read_pickle('./pickles/D_'+ genre + '.pickle')
	# remove './midi820/genre/' from filename
	# load_df['file name'] = load_df['file name'].str.replace(pat='./midi820/' + genre + '/', repl='', regex=False)

	# remove 'note length avg == 0'
	idx = load_df[load_df['note length avg'] == 0].index
	load_df = load_df.drop(idx)

	# labels
	df_length = len(load_df)
	this_label = np.full((df_length), genre_num)
	labels = np.hstack([labels, this_label])

	# append genre dataframe to input_tensor dataframe
	input_tensor = pd.concat([input_tensor, load_df], axis = 0)
	genre_num += 1 # next label

# drop file name & major/minor name -> not differentiable -> cannot be tensor
input_tensor = input_tensor.drop(['file name', 'key'], axis=1)

# input & output tensor
# input_tensor
output_tensor = np_utils.to_categorical(labels)

# train, valid split after shuffle
X_train, X_validation, Y_train, Y_validation = train_test_split(input_tensor, output_tensor, test_size = 0.3, random_state = None)






# MLP
model = Sequential()
classes = 5 # num of classes

# layer 1. 'glorot' = Xavier
activations = ['relu', 'relu', 'relu', 'relu', 'softmax']
model.add(Dense(256, input_dim = 10, activation=activations[0], kernel_initializer='glorot_uniform'))
model.add(Dropout(0.3))
model.get_weights()

# layer 2
model.add(Dense(256, activation=activations[1], kernel_initializer="glorot_uniform"))
model.add(Dropout(0.3)) #0~1

# layer 3
model.add(Dense(256, activation=activations[2], kernel_initializer="glorot_uniform"))
model.add(Dropout(0.3))

# layer 4
model.add(Dense(256, activation=activations[3], kernel_initializer="glorot_uniform"))
model.add(Dropout(0.3))

model.add(Dense(classes, activation=activations[4]))
model.summary() # show model





# configure model for training
#optim = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
optimizer = 'rmsprop'
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['acc'])



# model training
epoch = 200
history = model.fit(X_train, Y_train,
                    validation_data=(X_validation, Y_validation),
                    epochs=epoch, batch_size=100, verbose=1,
                    callbacks=[cb_checkpoint, cb_early_stopping, print_weights]) # callbacks=[cb_checkpoint,cb_early_stopping]


print('\nAccuracy: {:.4f}'.format(model.evaluate(X_validation, Y_validation, verbose=1)[1]))

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
