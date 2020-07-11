import os
import sys
import copy
from tqdm import tqdm
import torch.utils.data
import torch
import numpy as np
import matplotlib.pyplot as plt
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
            self.config.model_save_path + self.config.model_fname  # 명시
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
                self.data_loader = torch.load(
                    self.config.trainloader_save_path + "train_loader.pt"
                )
            else:
                self.data_loader = torch.load(
                    self.config.trainloader_save_path + "adv_train_loader.pt"
                )
        elif option is "v":
            print("attack on VALID data")
            if orig:
                self.data_loader = torch.load(
                    self.config.validloader_save_path
                    + "valid_loader.pt"  # default config (orig + valid)
                )
            else:
                self.data_loader = torch.load(
                    self.config.validloader_save_path + "adv_valid_loader_TandAT.pt"
                )

        self.split_data(spec_files)  # split into single batch (for attack)
        print("==> DATA LOADED")
        return

    def split_data(self, spec_files):
        self.input_total = []
        self.output_total = []
        self.fname_total = []

        for v in self.data_loader:
            for i in range(len(v[0])):  # 20
                self.input_total.append(
                    torch.unsqueeze(v[0][i], 0)
                )  # torch [1,129,400,128]
                self.output_total.append(torch.unsqueeze(v[1][i], 0))  # tensor [(#)]
            self.fname_total.extend(v[2])

        if not spec_files:  # list not empty
            tin, tout, tfn = [], [], []
            for f in spec_files:
                i = self.fname_total.index(f)  # find index
                tin.append(self.input_total[i])
                tout.append(self.output_total[i])
                tfn.append(self.fname_total[i])
                self.input_total, self.output_total, self.fname_total = tin, tout, tfn
        return

    def run(self):
        self.accuracies = []
        for ep in tqdm(self.config.epsilons):  # iteration: fgsm(n) others(1)
            orig_correct, atk_correct = self.test()
            orig_acc = orig_correct / len(self.input_total)
            atk_acc = atk_correct / len(self.input_total)

            print("Epsilon: {}".format(ep))
            print(
                "Before: {} / {} = {}".format(
                    orig_correct, len(self.input_total), orig_acc
                )
            )
            print(
                "After: {} / {} = {}".format(
                    atk_correct, len(self.input_total), atk_acc
                )
            )
            self.accuracies.append(atk_acc)

        # draw plot
        if self.config.plot:
            self.draw_plot(self.accuracies)
        return

    def test(self):  # call this function to run attack
        orig_wrong = 0
        atk_correct = 0
        for i, (X, truth, name) in enumerate(
            zip(self.input_total, self.output_total, self.fname_total)
        ):
            name = name.replace("/", "_")
            X, truth = X.to(self.device), truth.to(self.device)

            # check initial performance
            init_out = self.model(X)
            init_pred = torch.max(init_out, 1)[1].view(truth.size()).data

            # if correct, skip
            if init_pred.item() != truth.item():
                orig_wrong += 1
                continue  # next X

            # if wrong, ATTACK
            loss = self.criterion(init_out, truth)  # compute loss
            self.model.zero_grad()
            loss.backward()
            X_grad = X.grad.data

            # generate attack (single X)
            attack = self.generate(self.config.attack_type, X, X_grad)

            # re-test
            new_out = self.model(attack)
            confidence = torch.softmax(new_out[0], dim=0)
            target_confidence = torch.max(confidence).item() * 100
            new_pred = torch.max(new_out, 1)[1].view(truth.size()).data
            if new_pred.item() == truth.item():
                atk_correct += 1
            else:
                # save successful attacks
                # np.save("", attack)
                pass

        return len(self.input_total) - orig_wrong, atk_correct

    def generate(self, atk, data, data_grad):
        if atk is "fgsm":
            attack = self.fgsm(data, data_grad, self.config.epsilons)
        elif atk is "deepfool":
            attack = self.deepfool(data, self.config.max_iter)
        elif atk is "random":
            attack = self.random(data, self.config.epsilons)
        else:
            raise ("Type error. It should be one of (fgsm, deepfool, random)")
        return attack

    def fgsm(self, data, data_grad, eps):
        sign_data_grad = data_grad.sign()
        perturbed_input = data + eps * sign_data_grad
        perturbed_input = torch.clamp(perturbed_input, 0, 127)
        return perturbed_input

    def deepfool(self, data, max_iter):
        return perturbed_input

    def random(self, data, eps):
        return perturbed_input

    def tempo(self, data):
        return perturbed_input

    def draw_plot(self, acc):
        # option1
        # plt.figure(figsize=(5,5))
        # plt.plot(epsilons, accuracies, "*-")
        # plt.yticks(np.arange(0, 1.1, step=0.1))
        # plt.xticks(np.arange(0, 7, step=1))
        # plt.title("Accuracy vs +- Cell Range")
        # plt.xlabel("Cell Range")
        # plt.ylabel("Accuracy")
        # plt.show()

        # option2
        plt.figure(figsize=(5, 5))
        plt.plot(self.config.epsilons, acc, "*-")
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.xticks(np.arange(0, 70, step=5))
        plt.title("Accuracy vs Epsilon")
        plt.xlabel("Epsilon")
        plt.ylabel("Accuracy")
        plt.show()
        return
