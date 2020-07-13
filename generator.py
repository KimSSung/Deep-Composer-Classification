from music21 import converter, corpus, instrument, midi, note, tempo
from music21 import chord, pitch, environment, stream, analysis, duration
import glob
import numpy as np
from tqdm import tqdm
import os
from config import get_config

##Assumption: /genres/[genre]/[.mid]
class Generator:
    def __init__(self, args):
        self.config = args

    def run(self):
        genres_dir = self.config.midi_files_path
        for genre in tqdm(self.config.genres):  # each genre

            print("\n######################## {} #######################".format(genre))

            count_file = 0
            for file in glob.glob(genres_dir + genre + "/*.mid"):  # each file

                fname = file.replace(genres_dir + genre + "/", "")
                fsave_dir = self.config.input_save_path + genre

                try:
                    mid = self.open_midi(file, self.config.remove_drum)
                    note_matrix = self.generate_matrix(mid)
                    if note_matrix == -1:  # TODO: reformat as error
                        print("ERROR: all tracks are None\tSKIPPING...")
                        continue
                    self.save_file(note_matrix, fsave_dir, fname)

                except:
                    print("ERROR: failed to generate {}\tSKIPPING...".format(file))

                else:
                    count_file += 1
                    print("{} success: {}".format(count_file, fname))
                    if count_file == self.config.input_generate_num:
                        break

        return

    def open_midi(self, file, rm_drum):
        mf = midi.MidiFile()
        mf.open(file)
        mf.read()
        mf.close()

        if rm_drum:  # drum(track 10) -> remove
            for i in range(len(mf.tracks)):
                mf.tracks[i].events = [
                    ev for ev in mf.tracks[i].events if ev.channel != 10
                ]
        return midi.translate.midiFileToStream(mf)

    def save_file(self, data, genre_folder, name):
        if not os.path.exists(genre_folder):
            os.makedirs(genre_folder)
        np.save(genre_folder + "/" + name, data)  # save as .npy
        return

    # TODO: major reformation
    def generate_matrix(self, mid):
        time = 400
        note_matrix_3d_vel = [
            [[0 for k in range(128)] for i in range(time)] for j in range(129)
        ]

        s2 = instrument.partitionByInstrument(mid)
        if s2 == None:
            print("SKIP: No tracks found...")
            return -1

        None_count = 0
        for e in s2:  # each part(instrument)
            instr_index = e.getInstrument().midiProgram
            if instr_index == None:
                instr_index = 128  # put into none channel
                None_count += 1
                # print(
                #     "\ttrack{}: valid or not? --> {} None".format(
                #         instr_index, None_count
                #     )
                # )

            y, parent_element, velocity = self.extract_notes(e)  # send track
            if len(y) < 1:  # no notes in this track
                if instr_index != 128:
                    None_count += 1
                # print("\ttrack{}: no notes --> {} None".format(instr_index, None_count))
                continue

            x = [int(n.offset / 0.05) for n in parent_element]

            vel_count = 0
            for i, j in zip(x, y):
                if i >= time:  # x=offset(time-series)
                    break
                else:
                    note_matrix_3d_vel[instr_index][i][int(j)] = velocity[vel_count]
                    vel_count += 1

        if None_count == len(s2):
            return -1

        return note_matrix_3d_vel

    def extract_notes(self, track):
        parent_element = []
        ret = []
        ret_vel = []
        for nt in track.flat.notes:
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


# # Testing
# config, unparsed = get_config()
# for arg in vars(config):
#     argname = arg
#     contents = str(getattr(config, arg))
#     print(argname + " = " + contents)
# temp = Generator(config)
# temp.run()
