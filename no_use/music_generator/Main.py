import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
import matplotlib.pyplot as plt
import time
import os
from tempfile import TemporaryFile
import random
import midi_musical_matrix
import data
import multi_training
import pickle
from tensorflow.compat.v1.nn.rnn_cell import BasicLSTMCell
from tensorflow.compat.v1.nn.rnn_cell import LSTMStateTuple
from MyFunctions import Input_Kernel, LSTM_TimeWise_Training_Layer, LSTM_NoteWise_Layer, Loss_Function
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
# from keras.optimizers import Adadelta

# Import Midi files to list
# Working_Directory = os.getcwd()
Music_Directory = "../../../../data/new_midiset/"
Save_Directory = Music_Directory + 'pickles/'
Midi_Directories = ["Blues", "HipHopRap", "Jazz", "NewAge", "Pop"]

max_time_steps = 256 # only files atleast this many 16th note steps are saved
MAX_NUM = multi_training.MAX_NUM
num_training_pieces = int(MAX_NUM * 0.8) # 20%

# training_pieces = {} # dict
labels = []
genre_num = 0
genre_nums = [0,0,0,0,0]
blues, hip, jazz, newage, pop = [],[],[],[],[]
genre_matrices = []

mode = 'load'
if mode == 'save':
	print("saving........")
	# Gather the training pieces from the specified directories
	for f in range(len(Midi_Directories)):
		Training_Midi_Folder = Music_Directory + Midi_Directories[f]
		# training_pieces = {**training_pieces, **multi_training.loadPieces(Training_Midi_Folder, max_time_steps)}
		# genre_nums[f] = len(training_pieces) - genre_num
		# genre_num = len(training_pieces)

		matrix_list = multi_training.loadPieces(Training_Midi_Folder, max_time_steps) # genre dir path
		genre_nums[f] = len(matrix_list)

		# save pickle
		try:
			with open(Save_Directory + Midi_Directories[f] + '_matrix.pkl', 'wb') as f:
				pickle.dump(matrix_list, f)
			
			print('pickle saved!')
		
		except:
			print('save failed....')

		
	# train_len = len(training_pieces)
	# print(train_len)
	print('["Blues", "HipHopRap", "Jazz", "NewAge", "Pop"] :',genre_nums)
	print('Total:', sum(genre_nums))
	# print(training_pieces.keys())


	# labels

elif mode == 'load':
	print("loading........")
	# load pickle
	for f in range(len(Midi_Directories)):
		with open(Save_Directory + Midi_Directories[f] + '_matrix.pkl', 'rb') as f:
			matrix = pickle.load(f)
		print('loaded....')
		genre_matrices.append(matrix)

	blues = genre_matrices[0]
	hip = genre_matrices[1]
	jazz = genre_matrices[2]
	newage = genre_matrices[3]
	pop = genre_matrices[4]

	print('load finished')


# inputform = np.array(data.noteStateMatrixToInputForm(blues[0]))
# print(inputform.shape)

practice_batch_size = 15
practice_num_timesteps = 128

_, sample_state = multi_training.getPieceBatch(blues, practice_batch_size, practice_num_timesteps)
sample_state = np.array(sample_state)
sample_state = np.swapaxes(sample_state, axis1=1, axis2=2)
print('Sample of State Input Batch: shape = ', sample_state.shape)

training_pieces = blues[:num_training_pieces] + hip[:num_training_pieces] + jazz[:num_training_pieces] + newage[:num_training_pieces] + pop[:num_training_pieces]
validation_pieces = blues[num_training_pieces:] + hip[num_training_pieces:] + jazz[num_training_pieces:] + newage[num_training_pieces:] + pop[num_training_pieces:]
y_train = [0]*num_training_pieces + [1]*num_training_pieces + [2]*num_training_pieces + [3]*num_training_pieces + [4]*num_training_pieces
y_valid = [0]*(MAX_NUM-num_training_pieces) + [1]*(MAX_NUM-num_training_pieces) + [2]*(MAX_NUM-num_training_pieces) + [3]*(MAX_NUM-num_training_pieces) + [4]*(MAX_NUM-num_training_pieces)


##################################################################################
##################################################################################

'''
# Training

start_time = time.time()
N_epochs = 100
loss_hist=[]
loss_valid_hist=[]
restore_model_name = 'Long_Train'
save_model_name = 'Long_Train_256'
batch_size = 1
num_timesteps = 256
keep_prob=.5

# # Save Model
# Output_Directory = "./Output/" + save_model_name
# directory = os.path.dirname(Output_Directory)

# try:
# 	print('creating new destination folder')
# 	os.mkdir(directory)    
# except:
# 	print('destination folder exists')

# x_train = []
# for i in range(len(training_pieces)):
# 	temp = data.noteStateMatrixToInputForm(training_pieces[i])[:num_timesteps]
# 	x_train.append(temp)
# 	print('train:', np.asarray(temp).shape)

_, train_input_state = multi_training.getPieceBatch(training_pieces, batch_size, num_timesteps) # not using their 'convolution' filter
train_input_state = np.array(train_input_state)
x_train = np.swapaxes(train_input_state, axis1=1, axis2=2) 
# print(x_train.shape)

_, valid_input_state = multi_training.getPieceBatch(validation_pieces, batch_size, num_timesteps) # not using their 'convolution' filter
valid_input_state = np.array(valid_input_state)
x_valid = np.swapaxes(valid_input_state, axis1=1, axis2=2) 
# print(x_valid.shape)


model = Sequential()
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=(78,256)))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(Midi_Directories), activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
			  optimizer=Adadelta(learning_rate=1),
			  metrics=['accuracy'])


# Train
history = model.fit(x_train, y_train,
			batch_size=batch_size,
			epochs=N_epochs,
			verbose=1)

score = model.evaluate(x_valid, y_valid, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''

# Build the Model Graph:
tf.compat.v1.reset_default_graph()
print('Building Graph...')
#Capture number of notes from sample
num_notes = sample_state.shape[1]

tf.compat.v1.disable_eager_execution()

# Graph Input Placeholders
Note_State_Batch = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, num_notes, None, 2])
time_init = tf.compat.v1.placeholder(dtype=tf.int32, shape=())

#Generate expanded tensor from batch of note state matrices
# Essential the CNN 'window' of this network
Note_State_Expand = Input_Kernel(Note_State_Batch, Midi_low=24, Midi_high=101, time_init=time_init)

print('Note_State_Expand shape = ', Note_State_Expand.get_shape())


# lSTM Time Wise Training Graph 
num_t_units=[200, 200]
output_keep_prob = tf.compat.v1.placeholder(dtype=tf.float32, shape=())

# Generate initial state (at t=0) placeholder
timewise_state=[]
for i in range(len(num_t_units)):
    timewise_c=tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, num_t_units[i]]) #None = batch_size * num_notes
    timewise_h=tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, num_t_units[i]])
    timewise_state.append(LSTMStateTuple(timewise_h, timewise_c))

timewise_state=tuple(timewise_state)


timewise_out, timewise_state_out = LSTM_TimeWise_Training_Layer(input_data=Note_State_Expand, state_init=timewise_state, output_keep_prob=output_keep_prob)

print('Time-wise output shape = ', timewise_out.get_shape())



# #LSTM Note Wise Graph

# num_n_units = [100, 100]

# # Generate initial state (at n=0) placeholder
# notewise_state=[]
# for i in range(len(num_n_units)):
#     notewise_c=tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, num_n_units[i]]) #None = batch_size * num_timesteps
#     notewise_h=tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, num_n_units[i]])
#     notewise_state.append(LSTMStateTuple(notewise_h, notewise_c))

# notewise_state=tuple(notewise_state)


# y_out, note_gen_out = LSTM_NoteWise_Layer(timewise_out, state_init=notewise_state, output_keep_prob=output_keep_prob)

# p_out = tf.sigmoid(y_out)
# print('y_out shape = ', y_out.get_shape())
# print('generated samples shape = ', note_gen_out.get_shape())



# Loss Function and Optimizer

# loss, log_likelihood = Loss_Function(Note_State_Batch, y_out)
optimizer = tf.compat.v1.train.AdadeltaOptimizer(learning_rate = 1).minimize(loss)
#optimizer = tf.train.RMSPropOptimizer
print('Graph Building Complete')

# Training

start_time = time.time()
N_epochs = 50000
loss_hist=[]
loss_valid_hist=[]
restore_model_name = 'Long_Train'
save_model_name = 'Long_Train_256'
batch_size = 5
num_timesteps = 256
keep_prob=.5

# Save Model
Working_Directory = './'
Output_Directory = Working_Directory + "/Output/" + save_model_name
directory = os.path.dirname(Output_Directory)

try:
    print('creating new destination folder')
    os.mkdir(directory)    
except:
    print('destination folder exists')
            

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    # saver = tf.train.Saver()
    
    # # try to restore the pre_trained
    # if restore_model_name is not None:
    #     Load_Directory = Working_Directory + "/Output/" + restore_model_name
               
    #     print("Load the model from: {}".format(restore_model_name))
    #     saver.restore(sess, Load_Directory + '/{}'.format(restore_model_name))
        
    
    # Initial States
    timewise_state_val=[]
    for i in range(len(num_t_units)):
        c_t = np.zeros((batch_size*num_notes, num_t_units[i])) #start every batch with zero state in LSTM time cells
        h_t = np.zeros((batch_size*num_notes, num_t_units[i]))
        timewise_state_val.append(LSTMStateTuple(h_t, c_t))
        
    notewise_state_val=[]
    for i in range(len(num_n_units)):
        c_n = np.zeros((batch_size*num_timesteps, num_n_units[i])) #start every batch with zero state in LSTM time cells
        h_n = np.zeros((batch_size*num_timesteps, num_n_units[i]))
        notewise_state_val.append(LSTMStateTuple(h_n, c_n))
    
  

    # Training Loop
    for epoch in range(N_epochs+1):
        
        # Generate random batch of training data        
        if (epoch % 100 == 0):         
            print('Obtaining new batch of pieces')
            _, batch_input_state = multi_training.getPieceBatch(training_pieces, batch_size, num_timesteps) # not using their 'convolution' filter
            batch_input_state = np.array(batch_input_state)
            batch_input_state = np.swapaxes(batch_input_state, axis1=1, axis2=2)           
       

        # Run Session
        feed_dict = {Note_State_Batch: batch_input_state, timewise_state: timewise_state_val, notewise_state: notewise_state_val, time_init: 0, output_keep_prob: keep_prob}
        loss_run, log_likelihood_run, _, note_gen_out_run = sess.run([loss, log_likelihood, optimizer, note_gen_out], feed_dict=feed_dict)

        
        # Periodically save model and loss histories
        if (epoch % 1000 == 0) & (epoch > 0):
            # save_path = saver.save(sess, Output_Directory + '/{}'.format(save_model_name))
            # print("Model saved in file: %s" % save_path)
            np.save(Output_Directory + "/ training_loss.npy", loss_hist)
            np.save(Output_Directory + "/ valid_loss.npy", loss_valid)
        
        # Regularly Calculate Validation loss and store both training and validation losses
        if (epoch % 100) == 0 & (epoch > 0):
            # Calculation Validation loss
            _, batch_input_state_valid = multi_training.getPieceBatch(validation_pieces, batch_size, num_timesteps) # not using their 'convolution' filter
            batch_input_state_valid = np.array(batch_input_state_valid)
            batch_input_state_valid = np.swapaxes(batch_input_state_valid, axis1=1, axis2=2)    
            feed_dict = {Note_State_Batch: batch_input_state_valid, timewise_state: timewise_state_val, notewise_state: notewise_state_val, time_init: 0, output_keep_prob: keep_prob}
            loss_valid, log_likelihood_valid = sess.run([loss, log_likelihood], feed_dict=feed_dict)
            
            print('epoch = ', epoch, ' / ', N_epochs, ':')
            print('Training loss = ', loss_run, '; Training log likelihood = ', log_likelihood_run)
            print('Validation loss = ', loss_valid, '; Validation log likelihood = ', log_likelihood_valid)
            
            loss_hist.append(loss_run)
            loss_valid_hist.append(loss_valid)
        
        # Periodically generate Sample of music
            

end_time = time.time()

print('Training time = ', end_time - start_time, ' seconds')

# Plot the loss histories
plt.plot(loss_hist, label="Training Loss")
plt.plot(loss_valid_hist, label="Validation Loss")
plt.legend()
plt.show