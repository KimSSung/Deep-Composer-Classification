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

def generate_matrix(mid, prob):

    orig = [[[0 for k in range(128)] for i in range(400)] for j in range(128)]
    attack1 = [[[0 for k in range(128)] for i in range(400)] for j in range(128)] #attack frequency "pitch"
    attack2 = [[[0 for k in range(128)] for i in range(400)] for j in range(128)] #attack time "time-series"
    attack3 = [[[0 for k in range(128)] for i in range(400)] for j in range(128)] #attack strength "velocity"

    s2 = instrument.partitionByInstrument(mid)
    if s2 == None:
        print("None obj!")
        return -1, -1, -1, -1

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
                value = velocity[vel_count]
                vel_count += 1
                orig[instr_index][i][int(j)] = value
                rn = np.random.rand()
                if(rn < prob):
                    l = 1
                    plmi = np.random.rand()
                    if(plmi > 0.5):l=-1

                    if(int(j)==0):l= 1
                    if(int(j)==127): l=-1
                    attack1[instr_index][i][int(j)+l] = value
                    if(i==0): l = 1
                    if(i==399): l=-1
                    attack2[instr_index][i+l][int(j)] = value

                    rng = np.random.rand()
                    style = (127 - 30)*rng+30
                    if(value == 0):style=0
                    attack3[instr_index][i][int(j)] = int(style)

                else:
                    attack1[instr_index][i][int(j)] = value
                    attack2[instr_index][i][int(j)] = value
                    attack3[instr_index][i][int(j)] = value

    return orig, attack1, attack2, attack3



################################    RUN    #################################

# genres = ['Classical','Rock', 'Country'] #best
# genres = ['Jazz', 'HipHopRap','Blues']
# genres = ['Rock']
# genres = ['Country']
genres = ['Classical']
# genres = ['Jazz']
# genres = ['Blues']
for genre in genres:

    count_file = 0
    prob = 0.2

    genre_dir = "/data/midi820/" + genre + "/"
    # genre_dir = "/data/new_midiset/" + genre + "/"
    for file in glob.glob(genre_dir + "*.mid"):
        try:
            mid = open_midi(file, True) #unusual way of opening midi -> returns Stream obj


        except:
            print("ERROR OCCURED ON: " + file)
            print("SKIPPING ERROR TRACK!")
        else:
            o, n1, n2, n3 = generate_matrix(mid, prob)
            if (o == -1): continue

            # save each song as .npy
            np.save('/data/attack_test/'+file.replace(genre_dir, "")+"_orig_"+str(prob), o)  # save as .npy
            np.save('/data/attack_test/'+file.replace(genre_dir, "")+"_pitch_"+str(prob), n1)  # save as .npy
            np.save('/data/attack_test/'+file.replace(genre_dir, "")+"_time_"+str(prob), n2)  # save as .npy
            np.save('/data/attack_test/'+file.replace(genre_dir, "")+"_vel_"+str(prob), n3)  # save as .npy

            count_file += 1
            print("{} success: {}".format(count_file, file.replace(genre_dir, genre + "/")))

            if(count_file == 5):
                break

