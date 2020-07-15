import py_midicsv
import os
import numpy as np
import pandas as pd
import csv
import sys


class Converter:
    def __init__(self, args):
        self.config = args
        self.GENRES = ["Classical", "Rock", "Country", "GameMusic"]
        self.atype = "origin"
        # self.csv_printable = self.cofnig.csv_printable
        self.csv_printable = True

        ### Set File directory
        # Get the Hedaer and other data at original Midi Data
        self.ATTACK_PATH = "/data/inputs/"
        self.SAVED_NUMPY_PATH = "/data/midi820_128channel/"
        self.origin_midi_dir = "/data/3genres/"
        self.output_file_dir = "./attack2midi/converted/"
        self.csv_output_dir = "./attack2midi/csv/"

    # --------------------------------------------------------------------------
    # functions

    def start_track_string(self, track_num):
        return str(track_num) + ", 0, Start_track\n"

    def title_track_string(self, track_num):
        return str(track_num) + ', 0, Title_t, "Test file"\n'

    def program_c_string(self, track_num, channel, program_num):
        return (
            str(track_num)
            + ", 0, Program_c, "
            + str(channel)
            + ", "
            + str(int(program_num))
            + "\n"
        )

    def note_on_event_string(self, track_num, delta_time, channel, pitch, velocity):
        return (
            str(track_num)
            + ", "
            + str(delta_time)
            + ", Note_on_c, "
            + str(channel)
            + ", "
            + str(pitch)
            + ", "
            + str(velocity)
            + "\n"
        )

    def note_off_event_string(self, track_num, delta_time, channel, pitch, velocity):
        return (
            str(track_num)
            + ", "
            + str(delta_time)
            + ", Note_off_c, "
            + str(channel)
            + ", "
            + str(pitch)
            + ", "
            + str(velocity)
            + "\n"
        )

    def end_track_string(self, track_num, delta_time):
        return str(track_num) + ", " + str(delta_time) + ", End_track\n"

    def run(self):

        self.set_atype()
        self.convert_genre()

        return

    def set_atype(self):

        for file in os.listdir(self.ATTACK_PATH):
            if os.path.isfile(os.path.join(self.ATTACK_PATH, file)):

                # for simulation one music
                if "vel" in file:
                    self.atype = "vel"
                elif "noise" in file:
                    continue
                else:  # origin input2midi
                    self.atype = "origin"
        return

    def convert_genre(self):
        # load npy for each genres
        end_of_file_string = "0, 0, End_of_file\n"

        for genre in self.config.genres:

            changed_num = 0

            print("##############################")
            print("GENRE start:", genre)

            saved_name_numpy = self.SAVED_NUMPY_PATH + genre + "_filename.npy"
            saved_numpy = self.SAVED_NUMPY_PATH + genre + "_input.npy"

            load_full_data = np.load(saved_numpy)
            load_full_file_names = np.load(saved_name_numpy)

            # print(len(load_full_data)) # 100
            # print(load_full_data.shape) # (100, 400, 128)

            for idx in range(len(load_full_data)):

                if changed_num == 30:
                    break  # check 30 for each genre first
                changed_num += 1

                load_data = load_full_data[idx]
                load_file_name = load_full_file_names[idx]
                only_file_name = load_file_name.split("/")[4]
                genre = load_file_name.split("/")[3]
                # print(genre)
                # print(only_file_name)

                # origin_midi_dir = origin_midi_dir + genre + '/'  # add genre to path

                new_csv_string = []

                ## Set all of the new_csv_string
                # Header: Track, Delta Time, Type, Number of Tracks, Ticks for Quater Note
                total_track = 0
                track_num = 1  # Set the Track number
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

                origin_file = self.origin_midi_dir + only_file_name
                print("Original file:", only_file_name)
                origin_file_csv = py_midicsv.midi_to_csv(origin_file)

                # for string in origin_file_csv:
                #    if 'Program_c' in string: print(string)

                # slower by 4.8
                header = origin_file_csv[0].split(", ")
                # print('Before header:', header)
                header[-1] = str(int(int(header[-1][:-1]) / 4.8)) + "\n"
                # print('After header:', header)
                new_csv_string.append(
                    ", ".join(header)
                )  # header_string(total_track) + change last to 168 (too fast)
                new_csv_string.append(
                    origin_file_csv[1]
                )  # start_track_string(track_num)

                for string in origin_file_csv:
                    if "SMPTE_offset" in string:
                        # print(string)
                        continue
                    elif "Time_signature" in string or "Tempo" in string:
                        new_csv_string.append(string)

                    elif "Program_c" in string:
                        break

                new_csv_string.append(self.end_track_string(track_num, delta_time))
                # print('Before Real Data Part:')
                # for string in new_csv_string: print(string)

                total_track = 2
                current_used_instrument = [-1, -1]
                # find total track num
                for instrument_num, lst in enumerate(
                    load_data
                ):  # instrument_num : 0-127
                    if np.sum(lst) != (-1) * 400 * 128:
                        total_track += 1
                        current_used_instrument.append(instrument_num)

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

                for channel_instrument in range(0, 128):
                    for row in range(0, 400):
                        for col in range(0, 128):

                            if load_data[channel_instrument][row][col] == 0:
                                continue
                            else:
                                # Set the different condition for attacked Midi Files
                                # print('music21 instrument:', load_data[row][col]) # 0-59
                                # print('py_midicsv instrument:', program_num_map[load_data[row][col]])

                                if (
                                    len(
                                        track_string_list[
                                            current_used_instrument.index(
                                                channel_instrument
                                            )
                                        ]
                                    )
                                    != 0
                                ):
                                    program_num = channel_instrument  # program_num = instrment num
                                    pitch = col
                                    channel = 0
                                    delta_time = 50 * row
                                    end_delta_time = 50 * (row + 1)
                                    velocity = load_data[channel_instrument][row][col]
                                    note_on_list[track_num].append(
                                        [
                                            track_num,
                                            delta_time,
                                            channel,
                                            pitch,
                                            velocity,
                                        ]
                                    )
                                    note_off_list[track_num].append(
                                        [
                                            track_num,
                                            end_delta_time,
                                            channel,
                                            pitch,
                                            velocity,
                                        ]
                                    )
                                else:
                                    # Set the track_string_list new track header - program_c event
                                    track_num = current_used_instrument.index(
                                        channel_instrument
                                    )
                                    program_num = channel_instrument
                                    channel = 0
                                    pitch = col
                                    delta_time = 50 * row
                                    end_delta_time = 50 * (row + 1)
                                    velocity = load_data[channel_instrument][row][col]
                                    track_string_list[track_num].append(
                                        self.start_track_string(track_num)
                                    )
                                    track_string_list[track_num].append(
                                        self.title_track_string(track_num)
                                    )
                                    track_string_list[track_num].append(
                                        self.program_c_string(
                                            track_num, channel, program_num
                                        )
                                    )
                                    note_on_list[track_num].append(
                                        [
                                            track_num,
                                            delta_time,
                                            channel,
                                            pitch,
                                            velocity,
                                        ]
                                    )
                                    note_off_list[track_num].append(
                                        [
                                            track_num,
                                            end_delta_time,
                                            channel,
                                            pitch,
                                            velocity,
                                        ]
                                    )

                        for num in range(2, len(note_on_list)):  # num = track num
                            for notes in range(0, len(note_on_list[num])):
                                track_string_list[num].append(
                                    self.note_on_event_string(
                                        note_on_list[num][notes][0],
                                        note_on_list[num][notes][1],
                                        note_on_list[num][notes][2],
                                        note_on_list[num][notes][3],
                                        note_on_list[num][notes][4],
                                    )
                                )
                        for num in range(2, len(note_off_list)):
                            for notes in range(0, len(note_off_list[num])):
                                track_string_list[num].append(
                                    self.note_off_event_string(
                                        note_off_list[num][notes][0],
                                        note_off_list[num][notes][1],
                                        note_off_list[num][notes][2],
                                        note_off_list[num][notes][3],
                                        note_off_list[num][notes][4],
                                    )
                                )
                        note_on_list = [[] for i in range(0, total_track)]
                        note_off_list = [[] for i in range(0, total_track)]

                end_delta_time = 400 * 50
                for i in range(2, len(track_string_list)):
                    for j in track_string_list[i]:
                        new_csv_string.append(j)
                    new_csv_string.append(self.end_track_string(i, end_delta_time))
                new_csv_string.append(end_of_file_string)

                # print('NEW STRING')

                # data = pd.DataFrame(new_csv_string)
                # data.to_csv(csv_output_dir,index = False)

                midi_object = py_midicsv.csv_to_midi(new_csv_string)

                with open(
                    self.output_file_dir + genre + "/New_" + only_file_name, "wb"
                ) as output_file:
                    midi_writer = py_midicsv.FileWriter(output_file)
                    midi_writer.write(midi_object)
                    print("Good Midi File")

                # # For Cheking Error Data, Represent to csv files
                # csv_string = py_midicsv.midi_to_csv(output_file_dir + 'New_' + only_file_name)
                # tmp_list = []
                # for i in range(0, len(csv_string)):
                #     temp = np.array(csv_string[i].replace("\n", "").replace(" ", "").split(","))
                #     tmp_list.append(temp)
                # data = pd.DataFrame(tmp_list)
                # data.to_csv(csv_output_dir + 'New_' + only_file_name[:-4] + '.csv', header=False, index=False)

                # break # for checking one midi
