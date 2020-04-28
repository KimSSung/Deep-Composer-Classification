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
    # print(len(mf.tracks)) #each track consists of MidiEvent objects

    if(remove_drums):
        for i in range(len(mf.tracks)):
            mf.tracks[i].events = [ev for ev in mf.tracks[i].events if ev.channel != 10] #channel 10 = drum -> remove
    # print("track numbers: {}".format(len(mf.tracks)))
    return midi.translate.midiFileToStream(mf)


def extract_notes(chordified_midi):
    matrix_time_series = []
    pitch_in_str = []
    cur_tempo = 0

    for nt in chordified_midi.elements: #notes and rests x -> notes

        # stop at 600(30 sec)
        if len(matrix_time_series) >= 600:
            return matrix_time_series, pitch_in_str

        if isinstance(nt, tempo.MetronomeMark): #change of tempo
            cur_tempo = nt

        elif isinstance(nt, chord.Chord):  # chordify -> no (single) notes
            dur = round(cur_tempo.durationToSeconds(nt.duration),2) #duration in seconds

            pitch_in_str.append(nt)
            each_row = [0 for i in range(129)]  # each row
            for pitch in nt.pitches:
                each_row[int(max(0.0, pitch.ps))] = 1

            pitch_in_str.append(dur)
            for n in range(int(dur/0.05)):
                matrix_time_series.append(each_row)

        elif isinstance(nt, note.Rest):
            dur = round(cur_tempo.durationToSeconds(nt.duration),2) #duration in seconds
            for n in range(int(dur/0.05)):
                each_row = [0 for i in range(129)]  # each row
                matrix_time_series.append(each_row)

            pitch_in_str.append(nt)
            pitch_in_str.append(dur)

    return matrix_time_series, pitch_in_str



def draw_scatter_plot(note_matrix):
    x = []
    y = []
    for i in range(note_matrix.shape[0]):
        for j in range(note_matrix.shape[1]):
            if note_matrix.iloc[i, j] == 1:
                x.append(i)
                y.append(j)


    plt.scatter(x=x, y=y, c='DarkBlue', s=7)
    plt.title(file)
    plt.ylabel("pitch")
    plt.xlabel("time-series: 0.05sec")
    plt.show()

    return



################################    RUN    #################################



genres = ['Blues', 'HipHopRap', 'NewAge']
for genre in tqdm(genres):

    count_file = 0
    genre_data = [] #add to genre collection
    # genre_data_str = [] #chord - chord.dur - rest - rest.dur ...

    genre_dir = "../../../../data/backup/" + genre + "/"
    for file in glob.glob(genre_dir + "*.mid"):
        count_file += 1
        try:
            # mid = converter.parse(file) #original way of opening midi
            mid = open_midi(file, True) #unusual way of opening midi -> returns Stream obj
            merged_mid = mid.chordify()  # merge all parts
            note_matrix, note_str = extract_notes(merged_mid)
            note_matrix = note_matrix[:400]
            # print(len(note_matrix))


        except:
            print("ERROR OCCURED ON: " + file)
            print("SKIPPING ERROR TRACK!")

        else:
            if (len(note_matrix) < 400):
                print("{} SKIPPING: only {} in {}".format(count_file, len(note_matrix),file.replace(genre_dir, genre + "/")))
                continue
            print("{} success: {}".format(count_file, file.replace(genre_dir,genre+"/")))
            note_matrix_df = pd.DataFrame(note_matrix)
            # note_str_df = pd.DataFrame(note_str)
            # print(note_matrix_df)
            genre_data.append(note_matrix) #append each song data

            # draw_scatter_plot(note_matrix_df)


    # after circulating all files in each genre
    genre_3d_array = np.array(genre_data)
    np.save('../../../../data/temp/'+genre+'_input' , genre_3d_array) #save as .npy


for genre in ['Blues', 'HipHopRap', 'NewAge']:
    loaded = np.load('../../../../data/temp/'+genre+'_input.npy')
    print(genre,' len - ',loaded.shape)


###################################  NOT USED ANYMORE  #####################################

# def add_delta_time(chordified_midi):
#     for chord in chordified_midi:
#         chord.offset
#     return

# def compare_flat(chordified_midi):
#     flat = []
#     not_flat = []
#     for nt in chordified_midi.elements:
#         if isinstance(nt, chord.Chord):
#             for pitch in nt.pitches:
#                 not_flat.append(nt)
#
#     for nt in chordified_midi.flat.notes:
#         for pitch in nt.pitches:
#             flat.append(nt)
#
#     print(len(flat))
#     print(len(not_flat))