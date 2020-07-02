from music21 import converter, corpus, instrument, midi, note, tempo
from music21 import chord, pitch, environment, stream, analysis, duration
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from os.path import *
from os import *
import pandas as pd
import operator
from tqdm import tqdm
import torch

'''
THINGS TO CONSIDER(CHANGE):
time = 400
remove_drum = True/False
genres = []
genre_dir = '/data...'
count_files = 500

'''
################################    FUNCTIONS    #################################

def open_midi(midi_path, remove_drums):
    mf = midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()

    if(remove_drums):
        for i in range(len(mf.tracks)):
            mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10] #channel 10 = drum -> remove
    return midi.translate.midiFileToStream(mf)

def extract_notes(midi_part):
    parent_element = []
    ret = []
    ret_vel = []
    for nt in midi_part.flat.notes:
        if isinstance(nt, note.Note):
            ret.append(max(0.0, nt.pitch.ps))
            parent_element.append(nt)
            ret_vel.append(nt.volume.velocity)
        elif isinstance(nt, chord.Chord):
            vel = nt.volume.velocity
            for pitch in nt.pitches:
                ret.append(max(0.0, pitch.ps))
                parent_element.append(nt)
                ret_vel.append(vel)

    return ret, parent_element, ret_vel

def generate_matrix(mid):
    time = 400

    note_matrix_3d_vel = [[[0 for k in range(128)] for i in range(time)] for j in range(129)] #0-128 = 129

    s2 = instrument.partitionByInstrument(mid)
    if s2 == None:
        print("SKIP: No tracks found...")
        return -1

    None_count = 0
    for e in s2: #each part(instrument)
        instr_index = e.getInstrument().midiProgram
        if(instr_index == None):
            instr_index = 128 #put into none channel
            # e.show("text")
            None_count += 1
            print("\ttrack{}: valid or not? --> {} None".format(instr_index,None_count))
            # continue

        y, parent_element, velocity = extract_notes(e) #send track
        if (len(y) < 1): #no notes in this track
            if(instr_index != 128):
                None_count+=1
            print("\ttrack{}: no notes --> {} None".format(instr_index, None_count))
            continue

        x = [int(n.offset / 0.5) for n in parent_element]

        vel_count=0
        for i,j in zip(x,y):
            if (i >= time): # x=offset(time-series)
                break
            else:
                note_matrix_3d_vel[instr_index][i][int(j)] = velocity[vel_count]
                vel_count += 1
                # print("{}, {}, {}, {}".format(instr_index,i,j,velocity[vel_count]))

    if(None_count == len(s2)):
        print("SKIP: all tracks are None....")
        return -1

    return note_matrix_3d_vel

################################    RUN    #################################

# genres = ['Classical'] #best
# genres = ['Country']
# genres = ['Rock']
# genres = ['Jazz','Blues', 'HipHopRap']
genres = ['GameMusic']
# genres=  ['Pop']

for genre in tqdm(genres):

    count_file = 0
    genre_data = []   #genre collection
    genre_data_file = []    #genre filenames

    # genre_dir = "/data/midi820/" + genre + "/"
    # genre_dir = "/data/new_midiset/" + genre + "/"
    # genre_dir = "/data/Pop/"
    genre_dir = "/data/game_music/"
    for file in glob.glob(genre_dir + "*.mid"):
        try:
            mid = open_midi(file, False) #unusual way of opening midi -> returns Stream obj
            note_matrix_3d = generate_matrix(mid)
            # print(np.shape(note_matrix_3d))
        except:
            print("ERROR OCCURED ON: " + file)
            print("SKIPPING ERROR TRACK!")
        else:
            if (note_matrix_3d == -1): continue
            # genre_data.append(note_matrix_3d)  # append each song data (instrument)
            # genre_data_file.append(file)

            #save INDIVIDIUAL SONG
            np.save('/data/midi820_400/'+ genre + '/' +file.replace(genre_dir, ""), note_matrix_3d)  # save as .npy

            count_file += 1
            print("{} success: {}".format(count_file, file.replace(genre_dir, genre + "/")))

            # only generate 500 files
            if (count_file == 300):
                break

    # after circulating all files in each genre
    # genre_3d_array_instr = np.array(genre_data)
    # genre_filename_array = np.array(genre_data_file)


    #50 inputs for fgsm
    # np.save('/data/midi820_fgsm/instr/' + genre + '_input', genre_data)  # save as .npy
    # np.save('/data/midi820_fgsm/instr/' + genre + '_filename', genre_data_file)  # save as .npy

    #generate as chunk
    # np.save('/data/midi820_drum/'+genre+'_input' , genre_data) #save as .npy
    # np.save('/data/midi820_drum/'+genre+'_filename' , genre_data_file) #save as .npy
    # np.save('/data/midi820_cnn/'+genre+'_input' , genre_3d_array_vel) #save as .npy
    # np.save('/data/midi820_cnn/'+genre+'_filename' , genre_filename_array) #save as .npy
