from music21 import converter, corpus, instrument, midi, note
from music21 import chord, pitch, environment, stream, analysis, duration
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from os.path import *
from os import *
import pandas as pd
from mido import MidiFile
import pickle


data = {
	'file name' : [],
	'file size' : [],
	'song length' : [],
	'note length avg' : [],
	'max pitch' : [],
	'min pitch' : [],
	'max pitch diff' : [],
	'min pitch diff' : [],
	'pitch avg' : [],
	'instruments' : [],
	'major' : [],
	'key' : []
}

#remove drum & low level manipulation
def open_midi(midi_path, remove_drums):
	mf = midi.MidiFile()
	mf.open(midi_path)
	mf.read()
	mf.close()
	if (remove_drums):
		for i in range(len(mf.tracks)):
			mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10]

	return midi.translate.midiFileToStream(mf)


#calculate # of instruments in a single song
def num_instruments(midi):
	partStream = midi.parts.stream()
	num_inst = 0
	for p in partStream:
		#aux = p
		#print (p.partName)
		num_inst += 1

	return num_inst


#return array[notes' avg length in quarterlength, notes' avg pitch in ps]
def note_length_avg(midi_obj):
	ret = []
	note_count = 0
	dur_sum=0
	pitch_sum=0
	for n in midi_obj.flat.notes:
		if(type(n) == note.Note):
			dur_sum += n.duration.quarterLength
			pitch_sum += n.pitch.ps
			note_count += 1

	if (note_count <=  0):
		ret.append(0)
		ret.append(0)
	else:
		ret.append(dur_sum/note_count)
		ret.append(pitch_sum/note_count)
	return ret

#return array[min pitch, max pitch, min diff, max diff]
def max_min_pitch(midi):
	p = analysis.discrete.Ambitus()
	min_pitch, max_pitch = p.getPitchSpan(midi)
	min_diff, max_diff = p.getPitchRanges(midi)
	ret = [min_pitch.ps, max_pitch.ps, min_diff, max_diff]

	return ret


#return file size in KB
def get_file_size(file_path):
	try:
		n = getsize(file_path)
		KBsize = n / 1024

	except error:
		print("No file found")

	return KBsize


#return file length in seconds
def get_song_length(file_path):
	try:
		mid = MidiFile(file_path, clip=True)
		return mid.length
	except:
		print("Error: get_song_length")
		pass


#return 0/1 for minor/major
def major_minor(key):
	if(key.find('major') == -1):
		return 0
	else:
		return 1


#return key info
def get_key(midi):
	return str(analysis.discrete.analyzeStream(midi, 'key'))



#실제 실행 코드
#######################################################################################
genre = 'Classical'
dir_genre = './midi820/' + genre
column_list = ['file name', 'file size', 'song length', 'note length avg', 'max pitch',
				'min pitch', 'max pitch diff', 'min pitch diff', 'pitch avg', 'instruments', 'major', 'key']
files = []
for f in listdir(dir_genre):
	if isfile(join(dir_genre, f)) :
		new_path = dir_genre + '/' + f
		files.append(new_path)


count_file = 0
error_list = []
df = pd.DataFrame(data, columns=column_list)    #empty data frame
df.to_pickle('./' + genre + '.pickle')      #picklefy

for each in files:

	try:
		count_file += 1
		base_midi = open_midi(each, True)
		only_filename = each.split('/')[3]
		data['file name'].append(only_filename)
		data['file size'].append(get_file_size(each))
		data['song length'].append(get_song_length(each))
		avg = note_length_avg(base_midi)
		data['note length avg'].append(avg[0])
		data['pitch avg'].append(avg[1])
		pitch = max_min_pitch(base_midi)
		data['max pitch'].append(pitch[1])
		data['min pitch'].append(pitch[0])
		data['max pitch diff'].append(pitch[3])
		data['min pitch diff'].append(pitch[2])
		data['instruments'].append(num_instruments(base_midi))
		key_info = get_key(base_midi)
		major = major_minor(key_info)
		data['major'].append(major)
		data['key'].append(key_info)

	except:
		print('error on ',count_file,',th file | file name: \"',files[count_file-1],'\"')
		for key in list(data.keys()):
			if key == 'file name':
				data[key].append(each)
			else: data[key].append(np.nan)
		continue

	else:
		print(count_file)
		if (count_file % 10 == 0):
			df = pd.DataFrame(data, columns=column_list)
			print(df)
			df.to_pickle('./' + genre + '.pickle')
			# new_df = pd.read_pickle('./' + genre + '.pickle') ; pickle -> dataframe


df = pd.DataFrame(data, columns=column_list)
df.to_pickle('./' + genre + '.pickle')
print(error_list)