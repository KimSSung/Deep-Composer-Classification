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

    note_matrix_3d_vel = [[[0 for k in range(128)] for i in range(400)] for j in range(128)]

    s2 = instrument.partitionByInstrument(mid)
    if s2 == None:
        print("None obj!")
        return -1

    for e in s2: #each part(instrument)
        instr_index = e.getInstrument().midiProgram
        if(instr_index == None): continue
        y, parent_element, velocity = extract_notes(e)
        if (len(y) < 1): continue
        x = [int(n.offset / 0.5) for n in parent_element]

        vel_count=0
        for i,j in zip(x,y):
            if (i >= 400): # x=offset(time-series)
                break
            else:
                note_matrix_3d_vel[instr_index][i][int(j)] = velocity[vel_count]
                vel_count += 1
                # print("{} , {}, {}, {}".format(instr_index,i,j,velocity[vel_count]))

    return note_matrix_3d_vel

################################    RUN    #################################

genres = ['Classical','Rock', 'Country'] #best
# genres = ['Jazz', 'HipHopRap','Blues']
# genres = ['Rock']
# genres = ['Country']
# genres = ['Classical']
# genres = ['Jazz']
# genres = ['Blues']
for genre in tqdm(genres):

    count_file = 0
    # genre_data = []   #genre collection
    # genre_data_file = []    #genre filenames

    genre_dir = "/data/midi820/" + genre + "/"
    # genre_dir = "/data/new_midiset/" + genre + "/"
    for file in glob.glob(genre_dir + "*.mid"):
        try:
            mid = open_midi(file, True) #unusual way of opening midi -> returns Stream obj
            note_matrix_3d = generate_matrix(mid)

        except:
            print("ERROR OCCURED ON: " + file)
            print("SKIPPING ERROR TRACK!")
        else:
            if (note_matrix_3d == -1): continue
            # genre_data.append(note_matrix_3d)  # append each song data (instrument)
            # genre_data_file.append(file)

            # save each song as .npy
            np.save('/data/midi820_128channel/genres/'+genre+'/'+file.replace(genre_dir, ""), note_matrix_3d)  # save as .npy

            count_file += 1
            print("{} success: {}".format(count_file, file.replace(genre_dir, genre + "/")))

            # only generate 100 files
            if (count_file == 100):
                break

    # after circulating all files in each genre
    # genre_3d_array_instr = np.array(genre_data)
    # genre_filename_array = np.array(genre_data_file)






    #50 inputs for fgsm
    # np.save('/data/midi820_fgsm/instr/' + genre + '_input', genre_data)  # save as .npy
    # np.save('/data/midi820_fgsm/instr/' + genre + '_filename', genre_filename_data_file)  # save as .npy

    #generate as chunk
    # np.save('/data/midi820_128ch/'+genre+'_input' , genre_3d_array_instr) #save as .npy
    # np.save('/data/midi820_128ch/'+genre+'_filename' , genre_filename_array) #save as .npy
    # np.save('/data/midi820_cnn/'+genre+'_input' , genre_3d_array_vel) #save as .npy
    # np.save('/data/midi820_cnn/'+genre+'_filename' , genre_filename_array) #save as .npy