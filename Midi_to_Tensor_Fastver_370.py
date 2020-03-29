import numpy as np
from scipy.io.wavfile import write
import py_midicsv
import pandas as pd
import os
import torch.nn as nn
import torchaudio
import torch
import sys

DATA_PATH = './../../../../data/midi370/'
CSV_PATH = './../../../../data/csv/'
SAVE_PATH = './../../../../data/tensors/'

#print(data_middle)
#print(data_note_on)
#print(data_note_off)

#set midi to frequency convert table
#It can transform midi notes to frequency
midi_freq = {127:13289.75, 126:12543.85, 125:11175.30, 124:10548.08,
			 123:9956.06, 122:9397.27, 121:8869.84, 120:8372.02, 119:7902.13,
			 118:7458.62, 117:7040.00, 116:6644.88, 115:6271.93, 114:5919.91,
			 113:5587.65, 112:5274.04, 111:4978.03, 110:4698.64, 109:4434.92,
			 108:4186.01, 107:3951.07, 106:3729.31, 105:3520.00, 104:3322.44,
			 103:3135.96, 102:2959.96, 101:2793.83, 100:2637.02, 99:2489.02,
			 98:2349.32, 97:2217.46, 96:2093.00, 95:1975.53, 94:1864.66,
			 93:1760.00, 92:1661.22, 91:1567.98, 90:1479.98, 89:1396.91,
			 88:1318.51, 87:1244.51, 86:1174.66, 85:1108.73, 84:1046.50,
			 83:987.77, 82:932.33, 81:880.00, 80:830.61, 79:783.99, 78:739.99,
			 77:698.46, 76:659.26, 75:622.25, 74:587.33, 73:554.37, 72:523.25,
			 71:493.88, 70:466.16, 69:440.00, 68:415.30, 67:392.00, 66:369.99,
			 65:349.23, 64:329.63, 63:311.13, 62:293.66, 61:277.18, 60:261.63,
			 59:246.94, 58:233.08, 57:220.00, 56:207.65, 55:196.00, 54:185.00,
			 53:174.61, 52:164.81, 51:155.56, 50:146.83, 49:138.59, 48:130.81,
			 47:123.47, 46:116.54, 45:110.00, 44:103.83, 43:98.00, 42:92.50,
			 41:87.31, 40:82.41, 39:77.78, 38:73.42, 37:69.30, 36:65.41,
			 35:61.74, 34:58.27, 33:55.00, 32:51.91, 31:49.00, 30:46.25,
			 29:43.65, 28:41.20, 27:38.89, 26:36.71, 25:34.65, 24:32.70,
			 23:30.87, 22:29.14, 21:27.50, 20:25.96, 19:24.50, 18:23.12,
			 17:21.83, 16:20.60, 15:19.45, 14:18.35, 13:17.32, 12:16.35,
			 11:15.43, 10:14.57, 9:13.75, 8:12.98, 7:12.25, 6:11.56,
			 5:10.91, 4:10.30, 3:9.72, 2:9.18, 1:8.66, 0:8.18}


#sampling rate
sr = 10000.0
Ts = 1/sr



def sound_wave (midi_note, start_time, duration,padding_size,sampling_rate = 10000.0):
	# midi_to_freq: input: midi note (int) , output: frequency (int) by dictionary
	# start_time: use for phase difference
	# duration: length of list
	#padding_size: We need to padding the data 804440
	#Sampling_rate = 10000.0

	global Ts
	t = np.arange(start_time,start_time + duration,Ts)
	sine_wave = np.sin(midi_freq[midi_note] * 2 * np.pi *t + 0)
	#print("Sine Wave shape :", sine_wave.shape[0])
	#print("Check ")

	# For example if Duration = 1 , start_time = 3, [3,4)arange array make, Padding Size = 7
	# 0 1 2 3 4 5 6 (index)   we need to pad 0 left 3 value and pad 3 values for right
	# print("start: ",int(start_time * 10000))
	# print("end: ",int(padding_size - (duration+start_time)*10000))
	if(start_time*10000 <= padding_size):
		pad_sine_wave = np.pad(sine_wave,(int(start_time * 10000), int(padding_size - (duration+start_time)*10000)),'constant',constant_values=0.0)
	else:
		# print('Error: Negative Value')
		pad_sine_wave=np.zeros((padding_size,))
	# print(pad_sine_wave.shape)
	if (pad_sine_wave.shape[0] < padding_size):
		pad_sine_wave= np.pad(pad_sine_wave,(int(padding_size - pad_sine_wave.shape[0]),0),'constant',constant_values = 0.0)
	elif (pad_sine_wave.shape[0] > padding_size):
	   pad_sine_wave = pad_sine_wave[int(pad_sine_wave.shape[0] - padding_size):]
	# print(pad_sine_wave.shape)
	return pad_sine_wave


#set for the time

'''
1. Parsing Midi files
Get all of the ./midiset/XXX.mid Files on the directory 
open with MidiFile and Parse all of the Data 
 '''

# Main Code

genre = ['Rock','Jazz','Classical','Country','Pop']
#genre = ['Classical']
#genre_dir = './midiset/' + genre

for genre_index in range (0,len(genre)):
	genre_dir = DATA_PATH +  genre[genre_index]
	file_dir = []
	file_name = []
	tensor_list = [] # tensor list init
	num = 0 # check total # for each genre


	'''
	file_dir: 'List' of all of midi file directory
	genre : 'String' Change the genre for the file directory
	genre_dir : 'String'  
	'''


	os.chdir(genre_dir)
	for fp in os.listdir(genre_dir):
		if os.path.isfile(os.path.join(genre_dir, fp)):
			file_path = genre_dir + '/' + fp
			if fp.endswith(".mid"):
				file_dir.append(file_path)
				file_name.append(fp)
			else:
				continue

	#print("File name\n" + str(file_name))
	#print("File \n" + str(file_dir))
	#print(type(file_dir))


	for cur_file in range(0,len(file_dir)):

		num += 1

		try:
			print('current: file :' + file_name[cur_file] + "< " + genre[genre_index] + ">")
			csv_string = py_midicsv.midi_to_csv(file_dir[cur_file])  #Set to File Directory Here!
			## Caution: csv_string is List[string]  : we need to manipulate this data preprocess
			## Split all of the string elements to list elements

			tmp_list = []
			for i in range(0,len(csv_string)):
				temp=np.array(csv_string[i].replace("\n","").replace(" ","").split(","))
				tmp_list.append(temp)

			data = pd.DataFrame(tmp_list)
			data.to_csv(CSV_PATH + genre[genre_index] + '/' +file_name[cur_file] +'.csv' ,header=False, index = False)

			# Manipulating Dataframe
			# Drop all of the other colunms

			#BitMask for inside the dataframe, DF
			#Define to cut midi files --> Too Long and it makes overflow

			MAX_DF = 500

			data_note = data [ (data[2]== 'Note_on_c') | (data[2]=='Note_off_c')]
			# print(data_note)

			#Manipulate data_note_on DF
			data_note_on = data[(data[2] == 'Note_on_c') & (data[5]!='0')]

			#Change the MAX_DF
			if data_note_on.shape[0] > 500:
				MAX_DF = 500
			else:
				MAX_DF = data_note_on.shape[0]

			data_note_on= data_note_on.loc[:,0:5]
			data_note_on.reset_index(drop=True)
			data_note_on.columns = ['Track','Time','Event_Type','Channel','Note','Velocity']
			data_note_on.index = range(0,data_note_on.shape[0])
			data_note_on.drop(data_note_on.index[MAX_DF:],inplace = True)

			#Manipulate data_note_off DF
			data_note_off = data[(data[2] == 'Note_off_c') | ((data[2]=='Note_on_c') & (data[5] == '0'))]
			data_note_off= data_note_off.loc[:,0:5]
			data_note_off.reset_index(drop=True)
			data_note_off.columns = ['Track','Time','Event_Type','Channel','Note','Velocity']
			data_note_off.index = range(0,data_note_off.shape[0])

			#Cheking
			# print(data_note_on)
			# print(data_note_off)

			# Make the new DataFrame for Middle State
			# Cut the shape of DataFrame

			data_middle = pd.DataFrame(index = range(0,MAX_DF),
									   columns= ['Track','Duration','Event_Type','Channel','Note','Velocity'])

			# Shape returns tuple
			# print(data_middle)
			# print(data_note_on)

			#Calculate duration
			for i in range(0,MAX_DF):
				for j in range(0,data_note_off.shape[0]):
					if ((data_note_on.iloc[i])['Note'] == (data_note_off.iloc[j])['Note'] and (data_note_on.iloc[i])['Track'] == (data_note_off.iloc[j])['Track'] ):
						data_middle.iloc[i] = data_note_on.iloc[i]
						(data_middle.iloc[i])['Duration'] = int((data_note_off.iloc[j])['Time']) - int((data_note_on.iloc[i])['Time'])
						data_note_off = data_note_off.drop(index = j)
						data_note_off.index = range(0, data_note_off.shape[0])
						break
					else:
						continue

			#Initialize Data to List
			t_start_list = []
			t_duration = []
			note = []

			for i in range (0,MAX_DF):
				t_start_list.append(int(data_note_on.iloc[i,1])/1000)
				t_duration.append(int(data_middle.iloc[i,1])/1000)
				note.append(int(data_note_on.iloc[i,4]))


			total_t = np.arange(0, t_start_list[MAX_DF-1]+t_duration[MAX_DF-1],1/sr)
			PADDING_SIZE = total_t.shape[0]

			#SET THE BASE WAVE TO ADD UP
			base_sine_wave = np.sin(0 * 2 * np.pi * total_t + 0)
			abs_sum_wave = 0
			sum_wave = 0

			for i in range(0,MAX_DF):
				abs_sum_wave = abs_sum_wave + np.abs(sound_wave(midi_note= note[i],start_time = t_start_list[i],duration = t_duration[i],padding_size= PADDING_SIZE))
				sum_wave = sum_wave + sound_wave(midi_note= note[i],start_time = t_start_list[i],duration = t_duration[i],padding_size= PADDING_SIZE)



			### We should focus on scaled
			scaled = np.int16((sum_wave / np.max(abs_sum_wave)) * 32767)
			temp = scaled.shape[0]
			# print(temp)

			#Change the scaled for tensor
			tensor_format = scaled.reshape((1,temp))
			tensor_format = torch.from_numpy(tensor_format)
			# print(tensor_format)


			# append tensor to tensor_list
			print("tensor shape: ", tensor_format.shape)
			print('--------------------------------------------------------')
			tensor_list.append(tensor_format)


			# with open(SAVE_PATH + "Tensor_Data_"+ str(genre[genre_index]) +".txt","a") as f:
			# 	f.write('current: file :' + file_name[cur_file] + "< " + genre[genre_index] + ">" + "\n")
			# 	f.write(str(temp))
			# 	f.write("\n")

			# write('/nfs/home/ryan0507/pycharm_maincomputer/wav/'+genre[genre_index] + '/' + file_name[cur_file] + '.wav', 10000, scaled)



		except:
			# with open("./error_log/Error_Record_"+ str(genre[genre_index]) +".txt", "a") as f:
			# 	f.write("Error Occured With: " + file_name[cur_file] + "\n")
			# 	f.write(str(sys.exc_info()[0]))
			print("Error on file: ", file_name[cur_file])
			print('--------------------------------------------------------')

			os.remove(file_dir[cur_file])
			pass


	# save tensor....
	try:
		print('total # of genre:', num)
		torch.save(tensor_list, './../../../../data/tensors/'+ genre[genre_index] + '.pt')
		print(genre[genre_index] + ' - tensor_list len:', len(tensor_list))
		print(genre[genre_index] + ' - save success!')
		print('--------------------------------------------------------')
	except:
		print(genre[genre_index] + ' - save failed....')
		print('--------------------------------------------------------')