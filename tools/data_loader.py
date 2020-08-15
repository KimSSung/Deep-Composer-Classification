# MIDIDataset

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from glob import glob
from tools.transforms import ToTensor, Transpose, Segmentation
import random


class MIDIDataset(Dataset):
    def __init__(
        self, path, classes=13, omit=None, seg_num=20, age=False, transform=None
    ):
        self.path = path
        self.classes = [x for x in range(classes)]
        if omit is not None:  # ex) omit = [2, 10]
            for _, el in enumerate(omit):
                self.classes.remove(el)

        self.seg_num = seg_num  # seg num per song
        self.transform = transform

        self.x_path = []
        self.y = []

        for i, label in enumerate(self.classes):
            comp_dir = self.path + "composer" + str(label) + "/"
            for comp in glob(comp_dir):
                for midi_dir in glob(comp + "*/"):
                    ver_npy = glob(midi_dir + "/*.npy")
                    # randomly select n segments pth
                    tmp = [random.choice(ver_npy) for j in range(self.seg_num)]
                    self.x_path.extend(tmp)
                    self.y.append(i)
                    """ 중간에 composer 뺄 경우 i가 그만큼 당겨서 label 됨"""

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = np.load(self.x_path[idx], allow_pickle=True)
        Y = self.y[idx]
        data = {"X": X, "Y": Y}

        if self.transform is None:
            self.transform = transforms.Compose([Segmentation(), ToTensor()])
        data = self.transform(data)

        return data


##TEST
# v = MIDIDataset(
#     path="/data/inputs_full/",
#     transform=transforms.Compose([Segmentation(), Transpose(), ToTensor()]),  # checked
#     omit=[2],  # checked
#     seg_num=10, #checked
# )
# v_loader = DataLoader(v, batch_size=1, shuffle=True)
# for batch in v_loader:
#     random.seed(123)
#     print("{} {}".format(batch["Y"], batch["loc"]))
