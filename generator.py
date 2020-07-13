from music21 import converter, corpus, instrument, midi, note, tempo
from music21 import chord, pitch, environment, stream, analysis, duration
import glob
import numpy as np
from tqdm import tqdm

##Assumption: /genres/[genre]/[.mid]
class Generator:
    def __init__(self, args):
        self.config = args

        self.run(self.config.genres, self.config.midi_file_path)

    def run(self, genres, genres_dir):
        for genre in tqdm(genres):  # each genre
            print("\n\n############# {} ############\n\n".format(genre))
            count_file = 0
            for file in glob.glob(genres_dir + genre + "*.mid"):  # each file
                fname = file.replace(genres_dir, "")
                try:
                    mid = self.open_midi(file, self.config.remove_drum)
                except:
                    print("\nERROR: failed to open {}\n".format(file))

                else:
                    note_matrix = self.generate_matrix(mid)
                    if note_matrix == -1:
                        continue
                    np.save(
                        self.config.input_save_path + genre + "/" + fname, note_matrix
                    )  # save as .npy

                    count_file += 1
                    print("{} success: {}".format(count_file, fname))

                    if count_file == self.config.genre_datanum:
                        break

        return

    def open_midi(self, file, rm):
        mf = midi.MidiFile()
        mf.open(file)
        mf.read()
        mf.close()

        if rm:
            for i in range(len(mf.tracks)):
                mf.tracks[i].events = [
                    ev for ev in mf.tracks[i].events if ev.channel != 10
                ]  # channel 10 = drum -> remove
        return midi.translate.midiFileToStream(mf)

    def generate_matrix(self, mid):
        time = self.config.time_series

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
                # e.show("text")
                None_count += 1
                print(
                    "\ttrack{}: valid or not? --> {} None".format(
                        instr_index, None_count
                    )
                )
                # continue

            y, parent_element, velocity = self.extract_notes(e)  # send track
            if len(y) < 1:  # no notes in this track
                if instr_index != 128:
                    None_count += 1
                print("\ttrack{}: no notes --> {} None".format(instr_index, None_count))
                continue

            x = [int(n.offset / 0.5) for n in parent_element]

            vel_count = 0
            for i, j in zip(x, y):
                if i >= time:  # x=offset(time-series)
                    break
                else:
                    note_matrix_3d_vel[instr_index][i][int(j)] = velocity[vel_count]
                    vel_count += 1
                    # print("{}, {}, {}, {}".format(instr_index,i,j,velocity[vel_count]))

        if None_count == len(s2):
            print("SKIP: all tracks are None....")
            return -1

        return note_matrix_3d_vel

    def extract_notes(self, track):
        return
