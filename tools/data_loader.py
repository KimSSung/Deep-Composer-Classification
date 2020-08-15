# MIDIDataset

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from glob import glob, iglob
from tools.transforms import ToTensor, Transpose, Segmentation
import random


class MIDIDataset(Dataset):
    def __init__(
        self, txt_file, classes=13, omit=None, seg_num=20, age=False, transform=None
    ):
        self.txt_file = txt_file
        self.classes = [x for x in range(classes)]

        self.seg_num = seg_num  # seg num per song
        self.transform = transform

        self.x_path = []
        self.y = []

        self.map = dict()
        if omit is not None:
            for c in self.classes:
                if c in omit:
                    continue
                label = c - sum(c > o for o in omit)
                self.map[c] = label

        txt_list = open(self.txt_file, "r")
        for midi_pth in txt_list:  # each midi
            label = midi_pth[midi_pth.find("composer") + len("composer")]
            ver_npy = glob(midi_pth.replace("\n", "") + "*.npy")  # list
            # randomly select n segments pth
            tmp = [random.choice(ver_npy) for j in range(self.seg_num)]
            self.x_path.extend(tmp)
            self.y.append(self.map[int(label)])
            """ 중간에 composer 뺄 경우 i가 그만큼 당겨서 label 됨"""

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = np.load(self.x_path[idx], allow_pickle=True)
        Y = self.y[idx]
        data = {
            "X": X,
            "Y": Y,
        }

        if self.transform is None:
            self.transform = transforms.Compose([Segmentation(), ToTensor()])
        data = self.transform(data)

        return data


##TEST
# if __name__ == "__main__":
# v = MIDIDataset(
#     txt_file="/data/split/train.txt",
#     transform=transforms.Compose([Segmentation(), Transpose(), ToTensor()]),  # checked
#     omit=[2],  # checked
#     # seg_num=10, #checked
# )
# v_loader = DataLoader(v, batch_size=1, shuffle=True)
# for batch in v_loader:
#     random.seed(123)
#     print("{} {}".format(batch["Y"], batch["loc"]))
