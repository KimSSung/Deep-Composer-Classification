from music21 import converter, corpus, instrument, midi, note, tempo
from music21 import chord, pitch, environment, stream, analysis, duration
import glob
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import random
from config import get_config
import math


random.seed(123)


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

    def run(self):

        dataset_dir = self.config.load_path
        input_path = self.config.save_path
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
                        segment = self.generate_segment(mid)
                    except:
                        print("ERROR: failed to open {}\t".format(file_name))
                    else:
                        # assign uniq id to midi
                        version = self.fetch_version(orig_name)
                        track_id = self.fetch_id(track_list, orig_name)

                        fsave_pth = (
                            input_path + "composer" + str(i) + "/midi" + str(track_id)
                        )
                        self.save_input(segment, fsave_pth, version)  # TODO: enable
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
        self.name_id_map.to_csv(input_path + "name_id_map.csv", sep=",")  # TODO: enable
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

    def generate_segment(self, mid):
        stm_instr = instrument.partitionByInstrument(mid)
        for pt in stm_instr.parts:
            on, off, dur, pitch, vel = self.extract_notes(pt)

            track_dur_sec = pt.seconds  # last release
            track_dur_len = int(math.ceil(track_dur_sec / 0.05))
            segment = [
                [[0 for k in range(128)] for i in range(track_dur_len)]
                for j in range(2)
            ]  # 2 x duration x 128

            print(
                "generating... dur: {:.2f}sec || len: {}".format(
                    track_dur_sec, track_dur_len
                )
            )
            # iterate: each note
            for j, note in enumerate(zip(on, off, dur, pitch, vel)):

                x_index = int(note[0] // 0.05)  # time
                y_index = int(note[3])  # pitch

                # onset (binary)
                segment[0][x_index][y_index] = 1

                # note events (velocity)
                for t in range(int(note[2] // 0.05)):
                    # iterate: each 0.05 unit of a single note's duration
                    segment[1][x_index + t][y_index] = int(note[4])

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

    def save_input(self, matrix, save_pth, vn):
        if not os.path.exists(save_pth):
            os.makedirs(save_pth)

        np.save(save_pth + "/ver" + str(vn), matrix)  # save as .npy


if __name__ == "__main__":
    config, unparsed = get_config()
    temp = Generator(config)
    temp.run()
