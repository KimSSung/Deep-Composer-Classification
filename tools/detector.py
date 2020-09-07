import numpy as np


class Detector:
    """
    Input WHOLE shape for Numpy (4 dimension)
    shape: (400,128) 2d array

    Output: Marked Indices Numpy (where should be attack)
    """

    def __init__(self, input_npy):

        self.SONG = input_npy.shape[0]
        self.TRACK = input_npy.shape[1]
        self.ROW = input_npy.shape[2]
        self.COL = input_npy.shape[3]

        self.load_data_path = ""
        self.input_npy = input_npy
        self.mod_dict = {
            "C": 0,
            "C#": 1,
            "D": 2,
            "D#": 3,
            "E": 4,
            "F": 5,
            "F#": 6,
            "G": 7,
            "G#": 8,
            "A": 9,
            "A#": 10,
            "B": 11,
        }
        self.CHK_DURATION = 5
        self.chord_table = {}
        self.note_appeared = {}
        self.note_used = {i: 0 for i in range(0, 12)}
        self.note_used_npy = np.zeros((12))
        self.perturbed_npy = np.zeros((self.SONG, self.TRACK, self.ROW, self.COL))
        self.MARK_NUM = 1
        self.chord_inference = ""
        self.chord_name_list = []
        self.chord_numpy = np.zeros((1, 1))
        self.test_numpy = np.zeros((1, 1))

    def run(self):

        # Set chord Table for numpy
        self.input_npy = (self.input_npy).squeeze()
        self.set_chord_table()
        self.set_chord_name_list()
        self.set_chord_numpy()

        for track in range(self.TRACK):

            for time in range(self.ROW//self.CHK_DURATION):

                unit_time = time * self.CHK_DURATION
                if np.sum(self.input_npy[track][unit_time]) == 0.0:
                    continue
                self.detect_note(track, unit_time)
                self.set_note_used_numpy()
                self.test_probability()
                self.mark_npy(unit_time, self.chord_inference)

                #COPY for continous chord inference
                for copy_npy_row in range(self.CHK_DURATION):
                    self.mark_npy(unit_time + copy_npy_row, self.chord_inference)

        self.modify_note_range()
        # TODO: Erase Print
        # print(self.perturbed_npy)

        self.perturbed_npy[0][0] = 0
        return self.perturbed_npy

    def detect_note(self, track, row):

        """
        Find indices and update how many times note used
        update self.note_used {} after this function execute
        """

        # Initialize Clear the dictionary
        for key in self.note_used.keys():
            self.note_used[key] = 0

        indices = (self.input_npy)[track][row].nonzero()[0]
        # print(indices)

        # If there is no elements
        for index in indices:
            self.note_used[int(index) % 12] = self.note_used[int(index) % 12] + 1

        return

    def mod12_note(self, note):
        """
        Return Midi Note Number to Chord

        input: Midi Note Number
        return: Italic Representation Note
        """

        mod_dict = {
            "C": 0,
            "C#": 1,
            "D": 2,
            "D#": 3,
            "E": 4,
            "F": 5,
            "F#": 6,
            "G": 7,
            "G#": 8,
            "A": 9,
            "A#": 10,
            "B": 11,
        }

        return mod_dict[note % 12]

    def test_probability(self):
        """
        Use self.note_used dictionary that used
        Use self.chord_numpy for mark
        Set chord_inference = '' string to chord name
        Return self.chord_numpy[max_row]
        """

        chord_iterator = 0
        for index, harmony in enumerate(self.chord_table.keys()):

            self.test_numpy[index] = self.chord_numpy[index] * self.note_used_npy
            chord_iterator += 1

        # print(self.test_numpy)
        score_numpy = np.sum(self.test_numpy, axis=1)
        name_index = np.argmax(score_numpy)

        # TODO: When same probability appears we should handle it
        self.chord_inference = self.chord_name_list[name_index]

        return

    def set_chord_table(self):

        mod_dict = {
            "C": 0,
            "C#": 1,
            "D": 2,
            "D#": 3,
            "E": 4,
            "F": 5,
            "F#": 6,
            "G": 7,
            "G#": 8,
            "A": 9,
            "A#": 10,
            "B": 11,
            "Db": 3,
            "Gb": 6,
            "Ab": 8,
            "Bb": 9,
        }
        num_to_chord = {
            0: "C",
            1: "C#",
            2: "D",
            3: "D#",
            4: "E",
            5: "F",
            6: "F#",
            7: "G",
            8: "G#",
            9: "A",
            10: "A#",
            11: "B",
        }

        # Base Case for C
        self.chord_table["C_maj"] = [mod_dict["C"], mod_dict["E"], mod_dict["G"]]
        self.chord_table["C_min"] = [mod_dict["C"], mod_dict["D#"], mod_dict["G"]]
        self.chord_table["C_7"] = [
            mod_dict["C"],
            mod_dict["E"],
            mod_dict["G"],
            mod_dict["Bb"],
        ]
        self.chord_table["C_M7"] = [
            mod_dict["C"],
            mod_dict["E"],
            mod_dict["G"],
            mod_dict["B"],
        ]
        self.chord_table["C_aug6"] = [mod_dict["C"], mod_dict["E"], mod_dict["G"]]

        harmony_str = [
            "_maj",
            "_min",
            "_7",
            "_M7",
            "_aug6",
        ]  # TODO: Add more chord specifically

        for root_key in num_to_chord.keys():

            for chord in harmony_str:
                cur_chord = num_to_chord[root_key] + chord

                if cur_chord not in self.chord_table.keys():
                    self.chord_table[cur_chord] = []

                    for note in self.chord_table["C" + chord]:
                        self.chord_table[cur_chord].append((note + root_key) % 12)

        return

    def set_chord_name_list(self):

        for chord_name in self.chord_table.keys():

            self.chord_name_list.append(chord_name)

        return

    def set_chord_numpy(self):
        """
        Set self.chord_numpy
        MARK numpy what note is located for whole numpy
        """
        col = 12
        row = len(self.chord_name_list)
        self.chord_numpy = np.zeros((row, col), int)

        cur_row = 0

        for chord in self.chord_table.keys():

            for index in range(0, 12):

                if index in self.chord_table[chord]:
                    self.chord_numpy[cur_row][index] = 1
                else:
                    continue

            cur_row += 1

        self.test_numpy = np.zeros((row, col), int)

        return

    def set_note_used_numpy(self):
        """
        Function to set self.note_used dictionary -> numpy
        Result : self.test_numpy -> update how many time used

        Return None
        """
        tmp = []
        for note in self.note_used.keys():
            tmp.append(self.note_used[note])

        self.note_used_npy = np.array(tmp)

        # TODO: Delete Print statement
        # print(tmp)

        return

    def mark_npy(self, cur_row, chord_inference):
        """
        output: Marked With 1, (2X400X128)
        Return Nothing (optional)
        input: indices list (int[128])
        """

        # TODO: Change the indices for each cases
        for track in range(self.TRACK):
            for i in range(0, int(self.COL / 12)):

                # Slice perturbed_npy
                self.perturbed_npy[
                    0, track, cur_row, i * 12 : (i + 1) * 12
                ] = self.chord_numpy[self.chord_name_list.index(chord_inference)]

        return

    def modify_note_range(self):

        # Set perturbed npy to midi note
        self.perturbed_npy[0, :, :, 0:21] = 0
        self.perturbed_npy[0, :, :, 109:] = 0

        return


if __name__ == "__main__":

    # temp = np.zeros((1,2,100,128))
    # temp2 = np.ones((1,2,100,128))
    # temp3 = np.zeros((1,2,100,128))
    # temp4 = np.ones((1,2,100,128))
    #
    # input = np.vstack((temp, temp2, temp3, temp4))
    #
    #
    # for k in range(0,2):
    #     for i in range(122,150):
    #         for j in range(64,80):
    #             input[0][k][i][j] = 0

    input = np.load('/data/temp/chord/09-07-13-31/orig_composer3_midi56_ver0_seg6.npy')
    t = Detector(input)
    t.run()
    print(t.note_used)
