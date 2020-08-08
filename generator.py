from music21 import converter, corpus, instrument, midi, note, tempo
from music21 import chord, pitch, environment, stream, analysis, duration
import glob
import numpy as np
from tqdm import tqdm
import os
import re
import pandas as pd
import random
from config import get_config


random.seed(123)
##for visualize
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


class Generator:
    def __init__(self, args):
        self.config = args
        self.chars = {
            ",": "",
            ".": "",
            '"': "",
            "'": "",
            "/": "",
            "(": "",
            ")": "",
            "{": "",
            "}": "",
            "[": "",
            "]": "",
            "!": "",
            "?": "",
            "#": "",
            "$": "",
            "%": "",
            "&": "",
            "*": "",
            " ": "",
        }
        self.song_dict = dict()
        self.name_id_map = pd.DataFrame(
            columns=["composer", "composer_id", "orig_name", "midi_id", "saved_fname"]
        )  # df to store mapped info
        self.errors = {
            1: list(),
            2: list(),
            3: list(),
            4: list(),
        }  # mark errors

    def run(self):

        dataset_dir = self.config.midi_files_path
        input_path = self.config.input_save_path
        data_list, composers = self.get_data_list(
            dataset_dir + "maestro-v2.0.0_cleaned.csv"
        )

        for i, composer in tqdm(enumerate(composers)):
            success = 0  # count files for each composer
            track_list = list()  # for uniq track id

            print(
                "\n################################## {} ####################################\n".format(
                    composer
                )
            )

            for data in data_list:
                track_comp, orig_name, file_name = data[0], data[1], data[2]

                if track_comp is composer:
                    try:
                        mid = self.open_midi(dataset_dir + data[2])
                        segments = self.generate_segments(mid)  # list of segments
                    except:
                        print("ERROR: failed to open {}\t".format(file_name))
                    else:
                        if isinstance(segments, int):
                            self.errors[segments].append(file_name)
                            self.print_error(segments, file_name)
                            continue
                        version = self.fetch_version(orig_name)
                        track_id = self.fetch_id(
                            track_list, orig_name
                        )  # assign uniq id to midi

                        fsave_dir = (
                            input_path + "composer" + str(i) + "/midi" + str(track_id)
                        )
                        # self.save_input(segments, fsave_dir, version)  # TODO: enable
                        self.name_id_map = self.name_id_map.append(
                            {
                                "composer": composer,
                                "composer_id": i,
                                "orig_name": orig_name,
                                "midi_id": track_id,
                                "saved_fname": file_name,
                            },
                            ignore_index=True,
                        )

                        # print result
                        success += 1
                        print(
                            "{} success: {} => {} => midi{}_ver{}".format(
                                success, file_name, orig_name, track_id, version
                            )
                        )

        # save mapped list
        # self.name_id_map.to_csv(input_path + "name_id_map.csv", sep=",")  # TODO: enable

        # print error records
        print("\n\n#####ERROR RECORDS#####")
        for index, (code, names) in enumerate(self.errors.items()):
            print("error{}: {}".format(code, len(names)))  # value = list()
            for j, song in enumerate(names):
                print("{}".format(song))  # print each midi name

        return

    def get_data_list(self, fdir):  # return preprocessed list of paths
        data = pd.read_csv(fdir, encoding="euc-kr")  # cleaned csv
        data = data.drop(
            ["split", "year", "audio_filename", "duration"], axis=1
        )  # drop unnecessary columns
        data_list = list(
            zip(
                data["canonical_composer"],
                data["canonical_title"],
                data["midi_filename"],
            )
        )
        composers = data["canonical_composer"].unique()

        return data_list, composers

    def open_midi(self, file):
        mf = midi.MidiFile()
        mf.open(file)
        mf.read()
        mf.close()
        return midi.translate.midiFileToStream(mf)

    def generate_segments(self, mid):  # mid = each track(song)

        stm_instr = instrument.partitionByInstrument(mid)
        if stm_instr == None:  # 1. no tracks in stream
            return 1

        generated_input = []
        for pt in stm_instr:  # each part(instrument) -> piano
            instr_index = pt.getInstrument().midiProgram
            if instr_index != 0:  # 2. not piano track
                return 2

            on, off, dur, pitch, vel = self.extract_notes(
                pt
            )  # send track -> get lists of info
            if len(on) < 1:  # 3. no notes in this track
                return 3

            ##segmentation
            track_dur = off[len(off) - 1]
            seg_loc_list = self.get_seg_loc(self.config.overlap, track_dur)
            if len(seg_loc_list) < self.config.segment_num:  # 4. not enough segments
                return 4
            rnd_selected = random.sample(
                seg_loc_list, self.config.segment_num
            )  # randomly select n segments
            rnd_selected.sort()

            for pair in rnd_selected:  # iterate: each segment tuple (start, end)
                zipped = zip(on, off, dur, pitch, vel)
                segment = self.get_matrix(
                    pair[0], pair[1], zipped, self.config.augmentation
                )
                generated_input.append(segment)

        return generated_input  # list of matrices

    def get_matrix(self, start, end, notes, augmentation):
        if augmentation == "transpose":
            segment = [
                [[[0 for k in range(128)] for i in range(400)] for j in range(2)]
                for l in range(13)
            ]  # 13 x 2 x 400 x 128 (augmentation: +-6 semitones + original)

            # iterate: each note
            for j, note in enumerate(notes):

                x_index = int((note[0] - start) / 0.05)  # time
                y_index = int(note[3])  # pitch

                if (note[0] >= start and note[0] < end) or (
                    note[1] > start and note[1] <= end
                ):  # if note belongs to current segment
                    for t in range(
                        int(note[2] / 0.05)
                    ):  # iterate: each 0.05 unit of a single note's duration
                        if (x_index + t) >= 400:
                            break
                        segment[6][1][x_index + t][y_index] = int(note[4])
                        for r in range(1, 6):
                            segment[6 - r][1][x_index + t][
                                max(0, min(y_index - r, 128))
                            ] = int(note[4])
                            segment[6 + r][1][x_index + t][
                                max(0, min(y_index + r, 128))
                            ] = int(note[4])

                # onset (binary)
                if note[0] >= start and note[0] < end:
                    segment[6][0][x_index][y_index] = 1
                    for r in range(1, 6):
                        segment[6 - r][0][x_index][
                            max(0, min(y_index - r, 128))
                        ] = 1  # 0~5
                        segment[6 + r][0][x_index][
                            max(0, min(y_index + r, 128))
                        ] = 1  # 7~12
            return segment

        elif augmentation == "tempo_stretch":  # TODO: sunjong
            segment = [
                [[0 for k in range(128)] for i in range(400)] for j in range(2)
            ]  # 2 x 400 x 128 (filled with 0 -> default)

            # iterate: each note event ([0]:on [1]:off [2]:dur [3]:pitch [4]:vel)
            for j, note in enumerate(notes):

                """ this is where note events are MARKED in matrix cells
                    play around with this loop to implement tempo stretch
                    this is the only part you'll need to modify."""
                x_index = int((note[0] - start) / 0.05)  # time
                y_index = int(note[3])  # pitch

                if (note[0] >= start and note[0] < end) or (
                    note[1] > start and note[1] <= end
                ):  # if note belongs to current segment
                    for t in range(
                        int(note[2] / 0.05)
                    ):  # iterate: each 0.05 unit of a single note's duration
                        if (x_index + t) >= 400:
                            break
                        segment[1][x_index + t][y_index] = int(note[4])

                # onset (binary)
                if note[0] >= start and note[0] < end:
                    segment[0][x_index][y_index] = 1

            return segment

        else:  # no augmentation
            segment = [
                [[0 for k in range(128)] for i in range(400)] for j in range(2)
            ]  # 2 x 400 x 128

            # iterate: each note
            for j, note in enumerate(notes):

                x_index = int((note[0] - start) / 0.05)  # time
                y_index = int(note[3])  # pitch

                if (note[0] >= start and note[0] < end) or (
                    note[1] > start and note[1] <= end
                ):  # if note belongs to current segment
                    for t in range(
                        int(note[2] / 0.05)
                    ):  # iterate: each 0.05 unit of a single note's duration
                        if (x_index + t) >= 400:
                            break
                        segment[1][x_index + t][y_index] = int(note[4])

                # onset (binary)
                if note[0] >= start and note[0] < end:
                    segment[0][x_index][y_index] = 1

            return segment

    def extract_notes(self, track):
        offset_list = track.secondsMap
        on, off, dur, pitch, vel = [], [], [], [], []
        for evt in offset_list:
            element = evt["element"]
            if type(element) is note.Note:
                on.append(evt["offsetSeconds"])
                off.append(evt["endTimeSeconds"])
                dur.append(evt["durationSeconds"])
                pitch.append(element.pitch.ps)
                vel.append(element.volume.velocity)
            elif type(element) is chord.Chord:
                for nt in element.notes:
                    on.append(evt["offsetSeconds"])
                    off.append(evt["endTimeSeconds"])
                    dur.append(evt["durationSeconds"])
                    pitch.append(nt.pitch.ps)
                    vel.append(nt.volume.velocity)

        return on, off, dur, pitch, vel

    # return tuple of all possible start-end pairs (in Seconds)
    def get_seg_loc(self, overlap, dur):
        seg_pairs = list()  # list of tuples
        seg_length = 20
        total_seg = int(dur / seg_length)
        for i in range(total_seg):
            seg_pairs.append((i * seg_length, (i + 1) * seg_length))
            if overlap and i < (total_seg - 1):
                seg_pairs.append(
                    (
                        i * seg_length + 0.5 * seg_length,
                        (i + 1) * seg_length + 0.5 * seg_length,
                    )
                )

        return seg_pairs

    def fetch_version(self, track):
        track = track.lower()  # case-insensitive comparison
        track = track.translate(str.maketrans(self.chars))  # remove symbols
        if track in self.song_dict:
            self.song_dict[track] = self.song_dict[track] + 1  # update
        else:
            self.song_dict.update({track: 0})

        return self.song_dict[track]

    def fetch_id(self, lookup, name):
        name = name.lower()  # case-insensitive comparison
        name = name.translate(str.maketrans(self.chars))  # remove symbols
        if name not in lookup:
            lookup.append(name)

        return lookup.index(name)

    def save_input(self, matrices, save_dir, vn):
        if self.config.augmentation == "transpose":
            for i in range(13):  # 0-12 (0~5 / 6 / 7~12)
                aug_dir = save_dir + "_aug" + str(i)
                if not os.path.exists(aug_dir):
                    os.makedirs(aug_dir)
                for j, mat in enumerate(matrices):
                    np.save(aug_dir + "/ver" + str(vn) + "_seg" + str(j), mat[i])

        else:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for i, mat in enumerate(matrices):
                np.save(
                    save_dir + "/ver" + str(vn) + "_seg" + str(i), mat
                )  # save as .npy
        return

    def print_error(self, code, fname):
        if code == 1:
            print("ERROR{}: no tracks found".format(code, fname))
        if code == 2:
            print("ERROR{}: instr not mapped to piano".format(code, fname))
        if code == 3:
            print("ERROR{}: no notes found".format(code, fname))
        if code == 4:
            print("ERROR{}: not enough segments".format(code, fname))
        return


if __name__ == "__main__":
    config, unparsed = get_config()
    temp = Generator(config)
    temp.run()
