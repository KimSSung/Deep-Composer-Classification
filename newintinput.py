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



def get_closest_bin(x): #32 digits
    round = str(bin(new_veloc)[2:])
    if(len(round) > 2): #100~
        if ((round[1] == '1') and (round.find('1', 2) != -1)): #round up!
            return len(round)+1
    return len(round)

#############################################################################################


#iterate over 5 genres
for genre in genres:
    dir_genre = './midi_origin/' + genre
    files = []
    for f in listdir(dir_genre):
        if isfile(join(dir_genre, f)) :
            new_path = dir_genre + '/' + f
            files.append(new_path)

    ls_genre = [] #empty list

    for each in files:
        try:
            mid = MidiFile(each) #mido

        except:
            print("ERROR OCCURED ON: " + each)
            print("SKIPPING ERROR TRACK!")

        else:
            list = []

            # insert CLS
            list.append(366)

            for i, track in enumerate(mid.tracks):

               timer = 0   #STOP TRACK after 5 sec
               veloc = 0   #for comparison
               tempo = 500000 #this is default tempo of midi in microsec

               if (len(list) >= 999):
                   break

               for msg in track:
                   # print(msg)
                   if (msg.type == 'set_tempo'):  # if tempo changes, time attr calculation gets affected
                       tempo = msg.tempo
                   if(not msg.is_meta):
                       if((msg.type == 'note_on') or (msg.type == 'note_off')):
                             new_veloc = msg.velocity
                             new_note = msg.note
                             new_time = msg.time
                             new_type = msg.type

                             if (new_time > 0):  # index: 256-355

                                 new_time_sec = md.tick2second(msg.time, mid.ticks_per_beat, tempo)
                                 timer += new_time_sec
                                 new_time_ms = new_time_sec * 100  # sec to 10msec (10msec = 1unit => total 100units = 1sec)
                                 if (new_time_ms > 100):  # max = 1000ms = 100units = 1sec
                                     new_time_ms = 100

                                 list.append(256 + int(new_time_ms))
                                 if (len(list) >= 999):
                                     break

                             # order: time(ms) -> velocity(10bin) -> note event(128 pitches)
                             if((new_veloc != veloc) and (new_veloc != 0)): #index: 356-365
                                 veloc = new_veloc
                                 list.append(356 + get_closest_bin(new_veloc))
                                 if(len(list) >= 999):
                                     break

                             if(new_type == 'note_on'): #index: 0-127
                                 list.append(int(new_note))
                                 if (len(list) >= 999):
                                     break

                             elif(new_type == 'note_off'): #index: 128-255
                                 list.append(128 + int(new_note))
                                 if (len(list) >= 999):
                                     break

                             if(timer >= 5): #STOP track after 5 sec
                                 break   #next track



               #after each track add SEP
               list.append(367)


            #after each file
            #add PAD if shorter than 1000
            while len(list) < 1000:
                list.append(368)

            ls_genre.append(list)

    #save as one file
    df = pd.DataFrame(ls_genre)
    #print(df)
    str_file = "./newintinput/" + genre + ".pickle"
    df.to_pickle(str_file)
    print("saved: " + str_file)
