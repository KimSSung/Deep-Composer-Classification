import os
import sys
import torch.utils.data
import torch
import numpy as np
from tqdm import tqdm
from models.resnet import resnet50
import copy
import torch.nn as nn
from train.config import get_config

# to import from sibling folders
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from models.resnet import resnet18, resnet101, resnet152, resnet50

# dataloader
from tools.data_loader import MIDIDataset


torch.manual_seed(123)


class Attacker:
    def __init__(self, args):
        self.config = args

        # for GPU use
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_genres = len(self.config.genres)
        self.data_load(self.config.specific_files, self.config.t_or_v, self.config.orig)

        # TODO: get model + loss f from main.py(train + attack)
        self.model = resnet50(self.config.input_shape[0], num_genres)
        self.model_load()

        self.criterion = nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(self.device)


    def model_load(self):
        self.model.eval()
        checkpoint = torch.load(
            self.config.model_save_path
            + self.config.model_fname  #명시
        )
        self.model.load_state_dict(checkpoint["model.state_dict"])
        print("==> MODEL LOADED")
        self.model = self.model.to(self.device)
        print("==> MODEL ON GPU")
        return


    def data_load(self, spec_files, option, orig):
        if option is "t":
            print("attack on TRAIN data")
            if orig:
                self.data_loader= torch.load(
                    self.config.trainloader_save_path + "train_loader.pt"
                    )
            else:
                self.data_loader=torch.load(
                    self.config.trainloader_save_path + "adv_train_loader.pt"
                )
        elif option is "v":
            print("attack on VALID data")
            if orig:
                self.data_loader=  torch.load(
                    self.config.validloader_save_path + "valid_loader.pt" #default config (orig + valid)
                )
            else:
                self.data_loader = torch.load(
                    self.config.validloader_save_path + "adv_valid_loader_TandAT.pt"
                )

        self.split_data(spec_files) #split into single batch (for attack)
        return


    def split_data(self, spec_files):
        self.input_total = []
        self.output_total = []
        self.fname_total = []

        for v in self.data_loader:
            for i in range(len(v[0])):  # 20
                self.input_total.append(torch.unsqueeze(v[0][i], 0))  # torch [1,129,400,128]
                self.output_total.append(torch.unsqueeze(v[1][i], 0))  # tensor [(#)]
            self.fname_total.extend(v[2])

        if not spec_files: #list not empty
            tin, tout, tfn = [], [], []
            for f in spec_files:
                i = self.fname_total.index(f) #find index
                tin.append(self.input_total[i])
                tout.append(self.output_total[i])
                tfn.append(self.fname_total[i])
                self.input_total, self.output_total, self.fname_total = tin, tout, tfn
        return

    def test(self):

        return

    def generate(self, ):
        if
        return

    def draw_plot(self):
        return

    def fgsm(self, ):

    def deepfool(self,):
    def attack(self, ):


