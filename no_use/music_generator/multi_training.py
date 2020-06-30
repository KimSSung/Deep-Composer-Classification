import os, random
from midi_musical_matrix import *
from data import *
import pickle
import numpy
import signal

#batch_width = 10 # number of sequences in a batch (use as argument)
#batch_len = 16*8 # length of each sequence
division_len = 16 # interval between possible start locations
MAX_NUM = 20

def loadPieces(dirpath, max_time_steps):
    # pieces = {} # dict
    pieces = []

    num = 0
    for fname in os.listdir(dirpath):
        if fname[-4:] not in ('.mid','.MID'):
            continue

        name = fname[:-4]

        try:
            outMatrix = midiToNoteStateMatrix(os.path.join(dirpath, fname))
        except:
            print('Skip bad file = ', name)
            # remove bad file
            if os.path.isfile(os.path.join(dirpath, fname)):
                os.remove(os.path.join(dirpath, fname))
                print(name, " removed!")
            

            outMatrix=[]
            
        if len(outMatrix) < max_time_steps:
            continue

        # pieces[name] = outMatrix # dict
        num += 1
        pieces.append(outMatrix)
        if num == MAX_NUM: break
        # print ("Loaded {}".format(name))

    return pieces # return list of matrix



def getPieceSegment(pieces, num_time_steps):
    # piece_output = random.choice(list(pieces.values()))
    piece_output = random.choice(pieces)
    
    start = random.randrange(0,len(piece_output)-num_time_steps,division_len)
    # print "Range is {} {} {} -> {}".format(0,len(piece_output)-num_time_steps,division_len, start)

    seg_out = piece_output[start:start+num_time_steps]
    seg_in = noteStateMatrixToInputForm(seg_out)

    return seg_in, seg_out

def getPieceBatch(pieces, batch_size, num_time_steps):
    i,o = zip(*[getPieceSegment(pieces, num_time_steps) for _ in range(batch_size)])
    return np.array(i), np.array(o)

def trainPiece(model,pieces,epochs,start=0):
    stopflag = [False]
    def signal_handler(signame, sf):
        stopflag[0] = True
    old_handler = signal.signal(signal.SIGINT, signal_handler)
    for i in range(start,start+epochs):
        if stopflag[0]:
            break
        error = model.update_fun(*getPieceBatch(pieces))
        if i % 100 == 0:
            print ("epoch {}, error={}".format(i,error))
        if i % 500 == 0 or (i % 100 == 0 and i < 1000):
            xIpt, xOpt = map(numpy.array, getPieceSegment(pieces))
            noteStateMatrixToMidi(numpy.concatenate((numpy.expand_dims(xOpt[0], 0), model.predict_fun(batch_len, 1, xIpt[0])), axis=0),'output/sample{}'.format(i))
            pickle.dump(model.learned_config,open('output/params{}.p'.format(i), 'wb'))
    signal.signal(signal.SIGINT, old_handler)