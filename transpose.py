import torchvision
from tools.data_loader import MIDIDataset
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset


class Transpose(object):
    """ augmentation: [-6, +6] semitones
        Args: semitone
    """

    def __init__(self, semitone):
        assert isinstance(semitone, int)
        self.semitone = semitone

    def __call__(self, segment):
        """segment = (X,Y)"""
        X, Y = segment[0], segment[1]  # X.shape = (2,400,128)
        new_X = list()
        for i in range(2):
            tmp = np.roll(X[i], self.semitone, axis=1)  # X[i].shape = (400,128)
            if self.semitone > 0:
                tmp[:, : self.semitone] = 0
            elif self.semitone < 0:
                tmp[:, self.semitone :] = 0
            new_X.append(tmp)
        new_X = np.array(new_X)  # (2,400,128)
        new_segment = (new_X, Y)

        return new_segment


class ToTensor(object):
    """Convert numpy.ndarray to tensor"""

    def __call__(self, segment):
        """segment = (X,Y)"""
        X, Y = segment[0], segment[1]
        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.long),
        )


##TEST
# v = MIDIDataset(
#     "/data/split/test.txt",
#     age=False,
#     # transform=torchvision.transforms.Compose([Transpose(3), ToTensor()]),
#     transform=torchvision.transforms.Compose([ToTensor()]),
# )
# v_loader = DataLoader(v, batch_size=1, shuffle=True)
# for batch in v_loader:
#     print(batch[0].size())
#     break
