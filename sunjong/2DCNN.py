from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt
import os
import numpy

MODEL_SAVE_FOLDER_PATH = './model/'

if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
  os.mkdir(MODEL_SAVE_FOLDER_PATH)

model_path = MODEL_SAVE_FOLDER_PATH + 'midiclass-' + '{epoch:02d}-{val_loss:.4f}.hdf5'

cb_checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss',
                                verbose=1, save_best_only=True)

cb_early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# (X_train, Y_train), (X_validation, Y_validation) = loading

# X_train = 
# X_validation = 

# Y_train = 
# Y_validation = 

genres = 5 # num of classes

model = Sequential()
# relu etc. -> he is better
model.add(Conv2D(64, kernel_size=(2,2), strides=(2,2), activation='elu', kernel_initializer='he_normal')
# model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, kernel_size=(2,2), strides=(2,2), activation='elu', kernel_initializer='he_normal')
# model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, kernel_size=(2,2), strides=(2,2), activation='elu', kernel_initializer='he_normal')
# model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(512, kernel_size=(2,2), strides=(2,2), activation='elu', kernel_initializer='he_normal')
# model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(1024, activation='elu')) # fc layer
model.add(Dropout(0.5))
model.add(Dense(genres, activation='softmax'))
model.add(Dropout(0.5))

model.summary() # show model

optim = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=optim,
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    validation_data=(X_validation, Y_validation),
                    epochs=3, batch_size=200, verbose=0,
                    callbacks=[cb_checkpoint, cb_early_stopping])

print('\nAccuracy: {:.4f}'.format(model.evaluate(X_validation, Y_validation)[1]))

y_vloss = history.history['val_loss']
y_loss = history.history['loss']

# visualizing
x_len = numpy.arange(len(y_loss))
plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")
plt.plot(x_len, y_vloss, marker='.', c='red', label="Validation-set Loss")

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()