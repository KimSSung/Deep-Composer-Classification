# MIDIDataset

import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from glob import glob
from tools.transformation import ToTensor, Transpose, Segmentation, TempoStretch
import random


class MIDIDataset(Dataset):
    def __init__(
        self, txt_file, classes=13, omit=None, seg_num=40, age=False, transform=None
    ):
        self.txt_file = txt_file
        self.classes = [x for x in range(classes)]

        self.seg_num = seg_num  # seg num per song
        self.transform = transform

        self.x_path = []
        self.y = []
        self.order = []

        self.map = {}

        self.omitlist = []
        if omit:
            self.omitlist = omit.split(",")  # ['2', '5']. str list.

        # omit = list of string
        if self.omitlist is not None:
            for c in self.classes:
                if str(c) in self.omitlist:
                    continue
                label = c - sum(c > int(o) for o in self.omitlist)
                self.map[c] = label

        # print(self.map)

        txt_list = open(self.txt_file, "r")
        for midi_pth in txt_list:  # each midi
            temp = midi_pth.split("/")
            comp_num = -1
            for i in temp:
                if "composer" in i:
                    comp_num = int(i.replace("composer", ""))
                    break
            # print(comp_num)

            ver_npy = glob(midi_pth.replace("\n", "") + "*.npy")  # list
            # randomly select n segments pth
            tmp = [random.choice(ver_npy) for j in range(self.seg_num)]
            self.x_path.extend(tmp)
            self.order.extend([k for k in range(self.seg_num)])  # seg 위치/순서
            self.y.extend([self.map[comp_num]] * self.seg_num)
            """ 중간에 composer 뺄 경우 i가 그만큼 당겨서 label 됨"""

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = np.load(self.x_path[idx], allow_pickle=True)
        Y = self.y[idx]
        fd = self.x_path[idx].find("composer")
        pth = self.x_path[idx][fd:]
        data = {"X": X, "Y": Y, "pth": pth}

        trans = [Segmentation(self.order[idx])]  # cannot be called from outside
        if self.transform == "Transpose":
            trans.append(Transpose(6))  # TODO: receive as variable
        elif self.transform == "Tempo":
            trans.append(TempoStretch())
        trans.append(ToTensor())
        data = transforms.Compose(trans)(data)

        return data


##TEST
# if __name__ == "__main__":
#     # seed = 333
#     # random.seed(seed)  # python random module
#     # np.random.seed(seed)  # np module
#     # torch.manual_seed(seed)  # for both CPU & GPU
#     # # torch.cuda.manual_seed(seed)
#     # # torch.cuda.manual_seed_all(seed)
#     # torch.backends.cudnn.benchmark = False
#     # torch.backends.cudnn.deterministic = True
#
#     t = MIDIDataset(txt_file="/data/split/train.txt", transform="Transpose")
#     t_loader = DataLoader(t, batch_size=1, shuffle=True)
#     for i, batch in enumerate(t_loader):
#         # print("{} {}".format(batch["Y"], batch["pth"]))
#
#         mat_notes = np.array(batch["X"][0][1])  # note channel
#         nzero = mat_notes.nonzero()
#         x = nzero[0]
#         y = nzero[1]
#
#         # draw plot
#         plt.ylim(0, 128)
#         # plt.xlim(0, 400)
#         plt.title(batch["Y"])
#         plt.xlabel("/0.05 sec")
#         plt.ylabel("pitch")
#         plt.scatter(x=x, y=y, c="red", s=2)
#         plt.show()
#
#         if i > 3:
#             break
