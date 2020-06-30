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
import mido as md
import operator
import pickle


genres = ['Classical', 'Jazz', 'Pop', 'Country', 'Rock']

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


def get_closest_bin(x):
    round = bin(new_veloc)[2:].find('1')  #find the first appearance of 1
    return round




##################################################################

#iterate over 5 genres
for genre in genres:
    dir_genre = './midi820/' + genre
    files = []
    for f in listdir(dir_genre):
        if isfile(join(dir_genre, f)) :
            new_path = dir_genre + '/' + f
            files.append(new_path)



    for each in files:
        try:
            mid = MidiFile(each) #mido

        except:
            print("ERROR OCCURED ON: " + each)
            print("SKIPPING ERROR TRACK!")



        for i, track in enumerate(mid.tracks):
           list = []
           new_row = [0] * 391
           new_row[388] = 1 #insert CLS
           list.append(new_row)

           veloc = 0   #for comparison
           tempo = 500000 #this is default tempo of midi in microsec
           counter = 0
           for msg in track:
               if (msg.type == 'set_tempo'):  # if tempo changes, time attr calculation gets affected
                   tempo = msg.tempo
               if(not msg.is_meta):
                   if((msg.type == 'note_on') or (msg.type == 'note_off')):
                         new_veloc = msg.velocity
                         new_note = msg.note
                         new_time = msg.time
                         new_type = msg.type

                         # order: velocity(32bin) -> note event(128 pitches) -> time(ms)
                         if(new_veloc != veloc): #index: 356-387
                             new_row = [0] * 391  # initialize array as 0
                             veloc = new_veloc
                             new_row[356 + get_closest_bin(new_veloc)] = 1
                             list.append(new_row)
                             if(len(list) == 1199):
                                 break


                         if(new_type == 'note_on'): #index: 0-127
                             new_row = [0] * 391  # initialize array as 0
                             new_row[int(new_note)] = 1
                             list.append(new_row)
                             if (len(list) == 1199):
                                 break

                         elif(new_type == 'note_off'): #index: 128-255
                             new_row = [0] * 391  # initialize array as 0
                             new_row[128 + int(new_note)] = 1
                             list.append(new_row)
                             if (len(list) == 1199):
                                 break

                         if(new_time > 0): #index: 256-355
                             new_row = [0] * 391  # initialize array as 0
                             time = new_time
                             new_time = md.tick2second(msg.time, mid.ticks_per_beat, tempo)*100 #sec to 10msec (10msec = 1unit => total 100units = 1sec)
                             if(new_time > 100): # max = 1000ms = 100units = 1sec
                                 new_time = 100
                             new_row[256 + int(new_time)] = 1
                             list.append(new_row)
                             if (len(list) == 1199):
                                 break


           new_row = [0] * 391  # initialize array as 0
           new_row[389] = 1  # add SEP
           list.append(new_row)

           #add padding if shorter than 1200
           while len(list) < 1200:
               new_row = [0] * 391  # initialize array as 0
               new_row[390] = 1   #add padding
               list.append(new_row)


           #save as each track
           df = pd.DataFrame(list)
           str_file = "./onehotvectors/" + genre + "/" + each.replace("./midi820/"+genre+'/','') + "-track" + str(i) +".pickle"
           df.to_pickle(str_file)
           print("saved: " + str_file)
          # print(each.replace("./midi820/"+genre+'/',''))
           # print(df)
