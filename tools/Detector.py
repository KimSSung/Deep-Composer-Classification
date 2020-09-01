import numpy as np

class Detector:
    '''
    Input SINGLE TRACK for Numpy
    shape: (400,128) 2d array

    Output: Marked Indices Numpy (where should be attack)
    '''
    def __init__(self, input_npy):

        self.load_data_path = ''
        self.input_npy = input_npy
        self.mod_dict = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10,
                    'B': 11}
        self.chord_table = {}
        self.note_appeared = {}
        self.note_used = {i:0 for i in range(0,12)}
        self.perturbed_npy = np.zeros((400,128))
        self.MARK_NUM = 1

    def run(self):

        #Set chord Table for numpy
        self.set_chord_table()

        for time in range(0,400):

            self.detect_note(time)


        return


    def detect_note(self, row):

        '''
        Find indices and update how many times note used
        '''

        #Initialize Clear the dictionary
        for key in self.note_used.keys():
            self.note_used[key] = 0

        indices = (self.input_npy)[row].nonzero()[0]
        print(indices)

        # If there is no elements
        for index in indices:

            self.note_used[index%12] = self.note_used[index%12] + 1

        return




    def mod12_note(self, note):
        '''
        Return Midi Note Number to Chord

        input: Midi Note Number
        return: Italic Representation Note
        '''

        mod_dict = {'C':0, 'C#':1, 'D':2, 'D#':3, 'E':4, 'F':5, 'F#':6, 'G':7, 'G#':8, 'A':9, 'A#':10, 'B':11}


        return (mod_dict[note%12])


    def test_probability(self):

        # for chord in self.chord_table.keys():

        return


    def set_chord_table(self):

        mod_dict = {'C':0, 'C#':1, 'D':2, 'D#':3, 'E':4, 'F':5, 'F#':6, 'G':7, 'G#':8, 'A':9, 'A#':10, 'B':11,
                    'Db':3, 'Gb':6, 'Ab':8 , 'Bb': 9}
        num_to_chord = {0:'C', 1: 'C#', 2:'D', 3:'D#', 4:'E', 5:'F', 6:'F#', 7:'G', 8: 'G#', 9:'A', 10:'A#', 11:'B'}

        # Base Case for C
        self.chord_table['C_maj'] = [mod_dict['C'], mod_dict['E'], mod_dict['G']]
        self.chord_table['C_min'] = [mod_dict['C'], mod_dict['D#'], mod_dict['G']]
        self.chord_table['C_7'] = [mod_dict['C'], mod_dict['E'], mod_dict['G'], mod_dict['Bb']]
        self.chord_table['C_M7'] = [mod_dict['C'], mod_dict['E'], mod_dict['G'], mod_dict['B']]
        self.chord_table['C_aug6'] = [mod_dict['C'], mod_dict['E'], mod_dict['G']]

        harmony_str = ['_maj','_min', '_7','_M7','_aug6'] #TODO: Add more chord specifically

        for root_key in num_to_chord.keys():

            for chord in harmony_str:
                cur_chord =  num_to_chord[root_key] + chord

                if cur_chord not in self.chord_table.keys():
                    self.chord_table[cur_chord] = []

                    for note in self.chord_table['C' + chord]:
                        self.chord_table[cur_chord].append( (note + root_key) % 12)

        return

    def mark_npy(self, cur_row, indices):
        '''
        output: Marked With 1, (2X400X128)
        Return Nothing (optional)
        input: indices list (int[128])
        '''

        for note in indices:
            self.perturbed_npy[cur_row][indices] = self.MARK_NUM

        return


if __name__=='__main__':


    temp = np.zeros((100,128))
    temp2 = np.ones((100,128))
    temp3 = np.zeros((100,128))
    temp4 = np.ones((100,128))

    input = np.vstack((temp, temp2, temp3, temp4))

    for i in range(122,150):
        for j in range(64,80):
            input[i][j] = 0

    t = Detector(input)
    t.set_chord_table()
    # print(t.set_chord_table())
    # print(t.chord_table)
    t.detect_note(133)
    print(t.note_used)
    t.detect_note(0)
    print(t.note_used)




