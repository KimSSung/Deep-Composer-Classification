import py_midicsv
import os
import numpy as np
import pandas as pd
import csv
import sys
from config import get_config


class Converter:
    def __init__(self, args):
        self.config = args

        self.atype = ""  # default

        self.csv_printable = True

        ### Set File directory
        # Get the Header and other data at original Midi Data
        self.npy_root_path = self.config.to_convert_path
        self.npy_path_list = []  # String list object /data/inputs/composer#/...
        self.midi_header_path_list = (
            []
        )  # Save matched version composer#/midi# -> "/data/3 Etudes, Op.65"
        self.origin_midi_dir = "/data/MAESTRO/maestro-v2.0.0/"  # Classical/...
        self.output_file_dir = "/data/converted_music/midi/"
        self.csv_output_dir = "/data/converted_music/csv/"
        self.mapping_csv_dir = "/data/inputs/name_id_map.csv"

        # To get original Header with matching
        self.composer = ""
        self.orig_midi_name = ""
        self.maestro_midi_name = ""
        self.success_num = 0
        self.limit_success_num = 10

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

        # Get all path for npy_path by root
        self.load_npy_path()

        print(">>>>>> Converting <<<<<<")

        for index, cur_npy in enumerate(self.npy_path_list):

            self.name_id_map_restore(cur_npy)
            print("PATH: " + cur_npy + "\n")

            self.convert_file(cur_npy)

            if self.success_num == self.limit_success_num:
                break

        return

    def convert_file(self, file):

        # TODO: Modify Discrete Sound
        off_note = 0
        success_num = 0
        new_csv_string = []

        total_track = 0
        track_num = 1  # Set the Track number

        program_num = 0
        delta_time = 0
        channel = 0
        pitch = 60
        velocity = 90

        # FOR CHECKING

        if os.path.isfile(file):

            file_name = file.split("/")[-1]
            if "vel" in file_name:
                self.atype = "vel"
            elif "noise" in file_name:
                self.atype = "noise"
            else:  # origin input2midi
                self.atype = "origin"

            if self.atype in file_name:

                only_file_name = file_name.replace(self.atype + "_", "").replace(
                    ".npy", ""
                )

            else:

                only_file_name = file_name.replace(".npy", "")

            this_genre = ""  # genre of this midi

            # Genre UNUSED FOR NOW
            # for genre in self.config.genres:
            #     if genre in self.config.to_convert_path:
            #         this_genre = genre
            #         break
            # if 'to_convert_path' contains genre name -> this_genre = genre name
            # else this_genre = ''
            # print(this_genre)

            # FOR SIMULATION
            # if file != 'scn15_11_format0.mid.npy': continue
            # only_file_name = 'scn15_11_format0.mid'

            # print(only_file_name)

            new_csv_string = []
            load_data = np.load(file)

            if this_genre == "":  # 'to_convert_path' not contain genre name

                origin_file = self.origin_midi_dir + self.get_origin_file_name(
                    self.composer, self.orig_midi_name
                )

            else:  # 'to_convert_path' contains genre name
                origin_file = (
                    self.origin_midi_dir
                    + this_genre
                    + "/"
                    + self.get_origin_file_name(self.composer, self.orig_midi_name)
                )

            print("Original file:", origin_file)

            try:
                origin_file_csv = py_midicsv.midi_to_csv(origin_file)
            except:
                print("MIDI_TO_CSV ERROR !!")

            else:
                print("current file:", file)
                # for string in origin_file_csv:
                #    if 'Program_c' in string: print(string)

                total_track = 2
                current_used_instrument = [-1, -1]
                # find total track num
                for instrument_num, lst in enumerate(
                    load_data
                ):  # instrument_num : 0-127
                    if np.sum(lst) != (off_note) * 400 * 128:
                        total_track += 1
                        current_used_instrument.append(instrument_num)

                # slower by 4.8
                header = origin_file_csv[0].split(", ")
                # print('Before header:', header)
                header[-1] = str(int(int(header[-1][:-1]) / 2.0)) + "\n"
                header[-2] = str(int(total_track))
                # print('After header:', header)
                new_csv_string.append(
                    ", ".join(header)
                )  # header_string(total_track) + change last to 168 (too fast)
                new_csv_string.append(
                    origin_file_csv[1]
                )  # self.start_track_string(track_num)

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

                # print(load_data.shape[0], " ", load_data.shape[1], " ", load_data.shape[2])
                for channel_instrument in range(0, load_data.shape[0]):
                    for row in range(0, load_data.shape[1]):
                        for col in range(0, load_data.shape[2]):

                            if load_data[channel_instrument][row][col] == off_note:
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
                                    velocity = int(
                                        load_data[channel_instrument][row][col]
                                    )

                                    # Check if the note is continuous or not

                                    # Append Note_on when before event don't exist

                                    # if (load_data[channel_instrument][row-1][col] == 0) and row!=0:
                                    note_on_list[track_num].append(
                                        [
                                            track_num,
                                            delta_time,
                                            channel,
                                            pitch,
                                            velocity,
                                        ]
                                    )

                                    # Append Note_off when after event don't exist
                                    # if (load_data[channel_instrument][row+1][col] == 0) and row!= (load_data.shape[1]-2):
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
                                    if channel_instrument == 128:
                                        program_num = 1
                                    else:
                                        program_num = channel_instrument
                                    channel = 0
                                    pitch = col
                                    delta_time = 50 * row
                                    end_delta_time = 50 * (row + 1)
                                    velocity = int(
                                        load_data[channel_instrument][row][col]
                                    )
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
                new_csv_string.append("0, 0, End_of_file\n")  # end of file string
                # print('NEW STRING')

                # data = pd.DataFrame(new_csv_string)
                # data.to_csv(csv_output_dir,index = False)

                midi_object = py_midicsv.csv_to_midi(new_csv_string)

                with open(
                    self.output_file_dir
                    + "New_"
                    + self.atype
                    + "_"
                    + self.orig_midi_name
                    + "_"
                    + only_file_name
                    + ".mid",
                    "wb",
                ) as output_file:
                    midi_writer = py_midicsv.FileWriter(output_file)
                    midi_writer.write(midi_object)
                    print("Good Midi File")

                    self.success_num += 1

                # For Cheking Error Data, Represent to csv files
                if self.csv_printable:
                    self.checking_csv(only_file_name)

    def checking_csv(self, only_file_name):
        csv_string = py_midicsv.midi_to_csv(
            self.output_file_dir
            + "New_"
            + self.atype
            + "_"
            + self.orig_midi_name
            + "_"
            + only_file_name
            + ".mid"
        )
        tmp_list = []
        for i in range(0, len(csv_string)):
            temp = np.array(csv_string[i].replace("\n", "").replace(" ", "").split(","))
            tmp_list.append(temp)
        data = pd.DataFrame(tmp_list)
        data.to_csv(
            self.csv_output_dir
            + "New_"
            + self.orig_midi_name
            + "_"
            + only_file_name
            + ".csv",
            header=False,
            index=False,
        )
        print(".csv saved!")

    def load_npy_path(self):
        """
        return: list of all the npy_path(abs_path)
        """

        # TODO: Change to get config
        self.npy_root_path = os.path.abspath("/data/inputs/")

        self.npy_path_list = []
        mapping_csv_df = pd.read_csv(
            self.mapping_csv_dir, encoding="UTF-8", index_col=False
        )  # read mapping csv
        mapping_csv_df = mapping_csv_df.drop(
            mapping_csv_df.columns[[0]], axis="columns"
        )
        print(mapping_csv_df)

        # Find all of the npy converted files

        for dirpath, dirnames, filenames in os.walk(self.npy_root_path):

            for filename in filenames:

                if filename.endswith(".npy"):

                    current_npy_path = str(dirpath) + "/" + str(filename)
                    self.npy_path_list.append(current_npy_path)
                    # print('Saved file path: ',current_npy_path) # Debug

        return self.npy_path_list

    def name_id_map_restore(self, cur_npy_string):

        # Set self.composer, self.orig_midi_name for right place
        split_string_list = cur_npy_string.split("/")  # List
        composer_num = -1
        midi_num = -1

        # Find the composer num, midi num position
        for index, dir in enumerate(split_string_list):

            if "composer" in dir:

                temp = list(filter(str.isdigit, dir))
                temp_str = ""

                for st in temp:
                    temp_str = temp_str + st

                composer_num = int(temp_str)

            if "midi" in dir:

                temp = list(filter(str.isdigit, dir))
                temp_str = ""

                for st in temp:
                    temp_str = temp_str + st

                midi_num = int(temp_str)

        mapping_csv_df = pd.read_csv(
            self.mapping_csv_dir, encoding="UTF-8", index_col=False
        )  # read mapping csv
        mapping_csv_df = mapping_csv_df.drop(
            mapping_csv_df.columns[[0]], axis="columns"
        )

        is_composer = mapping_csv_df["composer_id"] == composer_num
        is_song = mapping_csv_df["midi_id"] == midi_num

        subset_df = mapping_csv_df[is_composer & is_song]

        self.composer = subset_df.iloc[0].loc["composer"]
        self.orig_midi_name = subset_df.iloc[0].loc["orig_name"]
        print("Current Composer ", self.composer)
        print("Current Song", self.orig_midi_name)

    def get_origin_file_name(self, composer, orig_midi_name):
        """
        Get midi_filename at MAESTRO Dataset CSV

        composer : (str) composer name for csv
        orig_midi_name : (str) Restored Canonical_tilte for csv matching

        return : (str) MAESTRO Midi File name mapped
        """

        maestro_csv_df = pd.read_csv(
            "/data/MAESTRO/maestro-v2.0.0/maestro-v2.0.0_cleaned.csv", encoding="euc-kr"
        )

        is_composer = maestro_csv_df["canonical_composer"] == composer
        is_title = maestro_csv_df["canonical_title"] == orig_midi_name

        subset_df = maestro_csv_df[is_composer & is_title]

        self.maestro_midi_name = subset_df.iloc[0].loc["midi_filename"]

        return self.maestro_midi_name


####### Test Code #######

if __name__ == "__main__":
    config, unparsed = get_config()
    temp = Converter(config)
    temp.run()
    # temp.load_npy_path()
    # print(temp.name_id_map_restore('/data/inputs/composer10/midi17/ver4_seg7.npy'))
    # print(temp.composer)
    # print(temp.orig_midi_name)
    # print(temp.get_origin_file_name(temp.composer,temp.orig_midi_name))
    # print(temp.maestro_midi_name)
