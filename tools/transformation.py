import torchvision
import torch
import numpy as np
import random

# from .data_loader import MIDIDataset

# torch.manual_seed(333)
# random.seed(333)


class Segmentation(object):
    """ basic segmentation
        - randomly crop (2x) 400 x 128
        - overlap is NOT restricted or handled
        - save (start, end) in seconds
        - returns {X, Y, (start,end)}
    """

    def __call__(self, data):
        duration = len(data["X"][0])
        start = random.randint(0, duration - 401)
        end = start + 400

        X_crop = data["X"][:, start:end, :].copy()  # IMPORTANT: NOT A VIEW BUT A "COPY"

        return {"X": X_crop, "Y": data["Y"], "loc": (start, end)}


class Transpose(object):
    """ [-6, +6] semitones
    segment = {X, Y, (start,end)}
    np.roll() COPIES data to a new nd-array"""

    def __call__(self, segment):
        X, Y, loc = segment["X"], segment["Y"], segment["loc"]
        semitone = random.randint(-6, 6)  # randomly select [-6 ~ +6]
        new_X = list()
        for i in range(2):
            tmp = np.roll(X[i], semitone, axis=1)  # X[i].shape = (400,128)
            if semitone > 0:
                tmp[:, :semitone] = 0
            elif semitone < 0:
                tmp[:, semitone:] = 0
            new_X.append(tmp)

        new_X = np.array(new_X)  # (2,400,128)

        return {"X": new_X, "Y": Y, "loc": loc}


# class Tempo_Stretch(object):
#     TODO: implement tempo stretch here
#     """ segment = {X, Y, (start,end)}"""
#
#     def __init__(self, arg):
#       self.arg
#
#     def __call__(self, segment):
#         X, Y, loc = segment["X"], segment["Y"], segment["loc"]
#         new_X =
#
#         return {"X": new_X, "Y": Y, "loc": loc}


class ToTensor(object):
    """Convert numpy.ndarray to tensor
        segment = {X, Y, (start,end)}"""

    def __call__(self, segment):
        X, Y, loc = segment["X"], segment["Y"], segment["loc"]
        new_segment = {
            "X": torch.tensor(X, dtype=torch.float32),
            "Y": torch.tensor(Y, dtype=torch.long),
            "loc": loc,
        }
        return new_segment
