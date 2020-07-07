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

    if remove_drums:
        for i in range(len(mf.tracks)):
            mf.tracks[i].events = [
                ev for ev in mf.tracks[i].events if ev.channel != 10
            ]  # channel 10 = drum -> remove
    return midi.translate.midiFileToStream(mf)


def get_instr(mid, ilist, f, count):

    s2 = instrument.partitionByInstrument(mid)
    if s2 == None:
        print("None obj!")
        return

    temp = []
    for e in s2:  # each part(instrument)
        instr_index = e.getInstrument().midiProgram
        if instr_index == None:
            continue
        ilist[instr_index] += 1
        temp.append(instr_index)

    print("{}th {}: {}".format(count, f, temp))
    return


################################    RUN    #################################

# genres = ['Classical','Rock', 'Country'] #best
# genres = ['Jazz', 'HipHopRap','Blues']
# genres = ['Rock']
genres = ["Country"]
# genres = ['Classical']
# genres = ['Jazz']
# genres = ['Blues']
for genre in genres:

    count_file = 0
    instr_distrib = [0 for i in range(129)]

    genre_dir = "/data/midi820/" + genre + "/"
    # genre_dir = "/data/new_midiset/" + genre + "/"
    for file in glob.glob(genre_dir + "*.mid"):
        try:
            mid = open_midi(
                file, True
            )  # unusual way of opening midi -> returns Stream obj

        except:
            print("ERROR OCCURED ON: " + file)
            print("SKIPPING ERROR TRACK!")
        else:
            count_file += 1
            get_instr(mid, instr_distrib, file, count_file)

            # if(count_file == 100):
            #     break
            # if(count_file % 100 == 0):
            #     print(instr_distrib)

    print(instr_distrib)
    x = [j for j in range(129)]
    plt.bar(x, instr_distrib)
    string = "Instrument distribution for " + str(genre)
    plt.title(string)
    plt.ylabel("#")
    plt.xlabel("Instrument number")
    plt.show()
