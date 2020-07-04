import py_midicsv
import os
import numpy as np
# import pandas as pd
# import csv
import sys

GENRES = ['Classical', 'Rock', 'Country', 'GameMusic']
SAVED_NUMPY_PATH = '/data/midi820_128channel/'

### Set File directory
origin_midi_dir = '/data/3genres/'  # Get the Hedaer and other data at original Midi Data
# classical_numpy = 'C:/Users/hahal/PycharmProjects/MidiClass/attack_npy_filename/Classical_input.npy'
# classical_name_numpy = 'C:/Users/hahal/PycharmProjects/MidiClass/attack_npy_filename/Classical_filename.npy'
output_file_dir = '/data/scn15/'
# csv_output_dir = './attack2midi/csv/'
ATTACK_PATH = '/data/scn15/'


# --------------------------------------------------------------------------
# origin_midi_dir = '' #Get the Hedaer and other data at original Midi Data
# classical_numpy = 'C:\\Users\\icarus\\Desktop\\Rock_input.npy'
# classical_name_numpy = 'C:\\Users\\icarus\\Desktop\\Rock_filename.npy'
# attack = 'C:\\Users\\icarus\\Desktop\\ep_0.025_orig.npy'
# output_file_dir = 'C:\\Users\\icarus\\Desktop\\Alborada del Gracioso.mid'
# output_file_dir2 = 'C:\\Users\\icarus\\Desktop\\Andante alla marcia.mid'
# csv_output_dir = 'C:\\Users\\icarus\\Desktop\\New_Midi.csv'
# output_file_dir = '/data/csv_to_midi/Classical/'
# csv_output_dir = '/data/checking_csv/'


### Functions
def start_track_string(track_num):
    return str(track_num) + ', 0, Start_track\n'


def title_track_string(track_num):
    return str(track_num) + ', 0, Title_t, "Test file"\n'


def program_c_string(track_num, channel, program_num):
    return str(track_num) + ', 0, Program_c, ' + str(channel) + ', ' + str(int(program_num)) + '\n'


def note_on_event_string(track_num, delta_time, channel, pitch, velocity):
    return str(track_num) + ', ' + str(delta_time) + ', Note_on_c, ' + str(channel) + ', ' + str(pitch) + ', ' + str(
        velocity) + '\n'


def note_off_event_string(track_num, delta_time, channel, pitch, velocity):
    return str(track_num) + ', ' + str(delta_time) + ', Note_off_c, ' + str(channel) + ', ' + str(pitch) + ', ' + str(
        velocity) + '\n'


def end_track_string(track_num, delta_time):
    return str(track_num) + ', ' + str(delta_time) + ', End_track\n'



'''
### Check whether all cell have 0-127 velocity
count = 0
good_files = []
num = 0
vel = 0
noise = 0
for file in os.listdir(ATTACK_PATH):
   skip = 0
   if os.path.isfile(os.path.join(ATTACK_PATH, file)):
      loaded = np.load(os.path.join(ATTACK_PATH, file))
      if num == 30: break
      if 'vel' in file: vel += 1
      elif 'noise' in file: noise += 1
      for i in range(0,129):
         for j in range(0,400):
            for k in range(0, 128):
               if not (0 <= loaded[i][j][k] <= 127):
                  # print('loaded [i,j,k] =','[',i,'',j,'',k,'] => ', loaded[i][j][k])
                  count += 1
                  skip = 1
                  break
            if skip == 1: break
         if skip == 1: break
      if skip == 0: good_files.append(file)
      num += 1
for file in good_files:
   print(file)
print('# of good file:', count)
print('vel:', vel)
print('noise:', noise)
'''

### Codes for "Classical.npy, ..." ==> per Genre
'''
### load npy for each genres
for genre in GENRES:
   #changed_num = 0
   print('##############################')
   print('GENRE start:', genre)
   saved_name_numpy = SAVED_NUMPY_PATH + genre + '_filename.npy'
   saved_numpy = SAVED_NUMPY_PATH + genre + '_input.npy'
   load_full_data = np.load(saved_numpy)
   load_full_file_names = np.load(saved_name_numpy)
   # print(len(load_full_data)) # 100
   # print(load_full_data.shape) # (100, 400, 128)
   for idx in range(len(load_full_data)):
     #if changed_num == 30: break # check 30 for each genre first
     #changed_num += 1
     load_data = load_full_data[idx]
     load_file_name = load_full_file_names[idx]
     only_file_name = load_file_name.split('/')[4]
     genre = load_file_name.split('/')[3]
     # print(genre)
     # print(only_file_name)
     # origin_midi_dir = origin_midi_dir + genre + '/'  # add genre to path
'''

new_csv_string = []

## Set all of the new_csv_string
# Header: Track, Delta Time, Type, Number of Tracks, Ticks for Quater Note
total_track = 0
# def header_string(total_track = 0): # We should put tempo, in real data from origin midi data
#     return '0, 0, Header, 1, ' + str(total_track) + ', 168\n'
track_num = 1  # Set the Track number
# track_num + string

program_num = 0
delta_time = 0
channel = 0
pitch = 60
velocity = 90

# ## Read numpy_array with npy
# instrument_dict = {}
# for channel_instrument in range(0,128):
#     for row in range(0, 400):
#         for col in range(0, 128):
#             if channel_instrument in instrument_dict.keys() or load_data[channel_instrument][row][col] == -1:
#                 continue
#             else:
#                 instrument_dict[channel_instrument] = 1

# total_track = len(instrument_dict) + 2  # instr num + two -1(one for header, one for tempo & key & title ....)


#######################################################
###################### REAL PART ######################
#######################################################
### Codes for "scn15_format0.npy, ..." ==> per MIDI File

off_note = 0
attack_type = ['vel', 'noise']  # 'orig', 'pitch', 'time',
# only_file_name = 'alb_esp4_format0.mid'
success_num = 0

for file in os.listdir(ATTACK_PATH):
    if os.path.isfile(os.path.join(ATTACK_PATH, file)):

        if 'vel' in file:
            atype = 'vel'
        elif 'noise' in file:
            continue
        else: # origin input2midi
          atype = 'origin'

        only_file_name = file.replace(atype + '_', '').replace('.npy', '')

        for genre in GENRES:
            if genre in only_file_name:
                only_file_name = only_file_name.replace(genre + '_', '')
                break

        # FOR SIMULATION
        # if file != 'scn15_11_format0.mid.npy': continue
        # only_file_name = 'scn15_11_format0.mid'

        # print(only_file_name)

        new_csv_string = []
        load_data = np.load(os.path.join(ATTACK_PATH, file))

        origin_file = origin_midi_dir + only_file_name
        # print("Original file:", only_file_name)

        try:
            origin_file_csv = py_midicsv.midi_to_csv(origin_file)
        except:
            continue

        else:
            print("current file:", file)
            # for string in origin_file_csv:
            #    if 'Program_c' in string: print(string)
            total_track = 2
            current_used_instrument = [-1, -1]
            # find total track num
            for instrument_num, lst in enumerate(load_data):  # instrument_num : 0-127
                if np.sum(lst) != (off_note) * 400 * 128:
                    total_track += 1
                    current_used_instrument.append(instrument_num)

            # slower by 4.8
            header = origin_file_csv[0].split(', ')
            # print('Before header:', header)
            header[-1] = str(int(int(header[-1][:-1]) / 4.0)) + '\n'
            header[-2] = str(int(total_track))
            # print('After header:', header)
            new_csv_string.append(', '.join(header))  # header_string(total_track) + change last to 168 (too fast)
            new_csv_string.append(origin_file_csv[1])  # start_track_string(track_num)

            for string in origin_file_csv:
                if 'SMPTE_offset' in string:
                    # print(string)
                    continue
                elif 'Time_signature' in string or 'Tempo' in string:
                    new_csv_string.append(string)

                elif 'Program_c' in string:
                    break

            new_csv_string.append(end_track_string(track_num, delta_time))
            # print('Before Real Data Part:')
            # for string in new_csv_string: print(string)

            # ## Real Data Part # deleted after add 128 instrument dim
            # current_used_instrument = [-1, -1]
            # for instrument_num in instrument_dict.keys():
            #     current_used_instrument.append(instrument_num)
                # print(lst.shape)

            # print(total_track)

            # Set the track_string_list to identify different instrument time line
            track_string_list = [[] for i in range(0, total_track)]
            track_string_list[0].append(-1)  # To Generate Error -> Header File
            track_string_list[1].append(-1)  # To Generate Error -> Meta File

            note_on_list = [[] for i in range(0, total_track)]
            note_on_list[0].append(-1)
            note_on_list[1].append(-1)

            note_off_list = [[] for i in range(0, total_track)]
            note_off_list[0].append(-1)
            note_off_list[1].append(-1)

            print(load_data.shape[0], ' ', load_data.shape[1], ' ', load_data.shape[2])
            for channel_instrument in range(0, load_data.shape[0]):
                for row in range(0, load_data.shape[1]):
                    for col in range(0, load_data.shape[2]):

                        if load_data[channel_instrument][row][col] == off_note:
                            continue
                        else:
                            # Set the different condition for attacked Midi Files
                            # print('music21 instrument:', load_data[row][col]) # 0-59
                            # print('py_midicsv instrument:', program_num_map[load_data[row][col]])

                            if len(track_string_list[current_used_instrument.index(channel_instrument)]) != 0:
                                program_num = channel_instrument  # program_num = instrment num
                                pitch = col
                                channel = 0
                                delta_time = 50 * row
                                end_delta_time = 50 * (row + 1)
                                velocity = int(
                                    load_data[channel_instrument][row][col])  # TODO: We should consider later
                                note_on_list[track_num].append([track_num, delta_time, channel, pitch, velocity])
                                note_off_list[track_num].append([track_num, end_delta_time, channel, pitch, velocity])
                            else:
                                # Set the track_string_list new track header - program_c event
                                track_num = current_used_instrument.index(channel_instrument)
                                if channel_instrument == 128:
                                    program_num = 1
                                else:
                                    program_num = channel_instrument
                                channel = 0
                                pitch = col
                                delta_time = 50 * row
                                end_delta_time = 50 * (row + 1)
                                velocity = int(
                                    load_data[channel_instrument][row][col])
                                track_string_list[track_num].append(start_track_string(track_num))
                                track_string_list[track_num].append(title_track_string(track_num))
                                track_string_list[track_num].append(program_c_string(track_num, channel, program_num))
                                note_on_list[track_num].append([track_num, delta_time, channel, pitch, velocity])
                                note_off_list[track_num].append([track_num, end_delta_time, channel, pitch, velocity])

                    for num in range(2, len(note_on_list)):  # num = track num
                        for notes in range(0, len(note_on_list[num])):
                            track_string_list[num].append(
                                note_on_event_string(note_on_list[num][notes][0], note_on_list[num][notes][1],
                                                     note_on_list[num][notes][2], note_on_list[num][notes][3],
                                                     note_on_list[num][notes][4]))
                    for num in range(2, len(note_off_list)):
                        for notes in range(0, len(note_off_list[num])):
                            track_string_list[num].append(
                                note_off_event_string(note_off_list[num][notes][0], note_off_list[num][notes][1],
                                                      note_off_list[num][notes][2], note_off_list[num][notes][3],
                                                      note_off_list[num][notes][4]))
                    note_on_list = [[] for i in range(0, total_track)]
                    note_off_list = [[] for i in range(0, total_track)]

            end_delta_time = 400 * 50
            for i in range(2, len(track_string_list)):
                for j in track_string_list[i]:
                    new_csv_string.append(j)
                new_csv_string.append(end_track_string(i, end_delta_time))
            new_csv_string.append(end_of_file_string)
            # print('NEW STRING')

            # data = pd.DataFrame(new_csv_string)
            # data.to_csv(csv_output_dir,index = False)

            midi_object = py_midicsv.csv_to_midi(new_csv_string)

            with open(output_file_dir + '/New_' + atype + '_' + only_file_name, "wb") as output_file:
                midi_writer = py_midicsv.FileWriter(output_file)
                midi_writer.write(midi_object)
                print('Good Midi File')

                success_num += 1

            # # For Cheking Error Data, Represent to csv files
            # csv_string = py_midicsv.midi_to_csv(output_file_dir + 'New_' + only_file_name)
            # tmp_list = []
            # for i in range(0, len(csv_string)):
            #     temp = np.array(csv_string[i].replace("\n", "").replace(" ", "").split(","))
            #     tmp_list.append(temp)
            # data = pd.DataFrame(tmp_list)
            # data.to_csv(csv_output_dir + 'New_' + only_file_name[:-4] + '.csv', header=False, index=False)

            # break # for checking one midi

print('total success:', success_num)