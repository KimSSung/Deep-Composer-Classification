# Attacker class

import os
import sys
import copy
from tqdm import tqdm
import torch.utils.data
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from config import get_config
from models.resnet import resnet18, resnet34, resnet101, resnet152, resnet50
from _collections import OrderedDict
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support


class Attacker:
    def __init__(self, args):
        self.config = args

        # for GPU use #TODO: remove
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.label_num = self.config.composers
        self.data_loader = None
        self.input_total = []
        self.output_total = []
        self.pth_total = []
        self.data_load(self.config.orig)

        self.model_fname = None
        self.model_type = None
        self.model = self.get_model()
        self.model_load()

        self.criterion = nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(self.device)

    def data_load(self, orig):
        if orig:
            self.data_loader = torch.load(
                self.config.atk_path + "dataset/test/valid_loader.pt"
            )
            print("attack on VALID")
        else:
            pass
            # self.valid_loader_1 = torch.load(
            #     self.config.validloader_save_path + "adv_valid_loader_TandAT.pt"
            # )
            # print("adv_valid_loader_TandAT loaded!")
            # self.valid_loader_2 = torch.load(
            #     self.config.validloader_save_path + "adv_valid_loader_T.pt"
            # )
            # print("adv_valid_loader_T loaded!")
        self.split_data()  # split into single batch (for attack)
        print("==> DATA LOADED")
        return

    def split_data(self):

        for v in self.data_loader:
            print(v["Y"])
            for i in range(len(v["Y"])):
                self.input_total.append(torch.unsqueeze(v["X"][i], 0))
                # unsqueeze -> torch [1,2,400,128]
                self.output_total.append(torch.unsqueeze(v["Y"][i], 0))
                # tensor [(#)]
            self.pth_total.extend(v["pth"])
        return

    def get_model(self):
        self.model_fname = os.listdir(self.config.atk_path + "model/")[0]
        self.model_type = self.model_fname.split("_")[0]  # ex: resnet50
        if self.model_type == "resnet18":
            return resnet18(int(self.config.input_shape[0]), self.label_num)
        elif self.model_type == "resnet34":
            return resnet34(int(self.config.input_shape[0]), self.label_num)
        elif self.model_type == "resnet50":
            return resnet50(int(self.config.input_shape[0]), self.label_num)
        elif self.model_type == "resnet101":
            return resnet101(int(self.config.input_shape[0]), self.label_num)
        elif self.model_type == "resnet152":
            return resnet152(int(self.config.input_shape[0]), self.label_num)

    def model_load(self):
        self.model.eval()
        # self.model = nn.DataParallel(self.model).to(self.device)
        self.model.to(self.device)
        checkpoint = torch.load(
            self.config.atk_path + "model/" + str(self.model_fname)
        )  # 명시
        self.model.load_state_dict(checkpoint["model.state_dict"])
        print("==> MODEL LOADED: {}".format(self.model_type))
        return

    def run(self):
        self.accuracies = []
        for ep in tqdm(self.config.epsilons):  # iteration: fgsm(n) others(1)
            truths, preds, orig_correct, atk_correct = self.test(ep)
            orig_acc = orig_correct / len(self.output_total)
            atk_acc = atk_correct / len(self.output_total)

            w_f1score = f1_score(truths, preds, average="weighted")
            precision, recall, f1, supports = precision_recall_fscore_support(
                truths, preds, average=None, labels=list(range(13)), warn_for=tuple()
            )

            print("Epsilon: {}".format(ep))
            print("#########Before########")
            print(
                "Accuracy: {} / {} = {:4f}".format(
                    orig_correct, len(self.output_total), orig_acc
                )
            )
            print("F1 score: {:4f}".format(w_f1score))
            print("{:<30}{:<}".format("Precision", "Recall"))
            for p, r in zip(precision, recall):
                print("{:<30}{:<}".format(p, r))

            print("\n#########After#########")
            print(
                "Accuracy: {} / {} = {:4f}".format(
                    atk_correct, len(self.output_total), atk_acc
                )
            )
            self.accuracies.append(atk_acc)

        # draw plot
        if self.config.plot:
            self.draw_plot(self.accuracies, self.config.attack_type)
        return

    def test(self, epsilon):  # call this function to run attack
        orig_wrong = 0
        atk_correct = 0
        ground_truth = []
        output_pred = []
        for i, (X, truth, pair) in enumerate(
            tqdm(zip(self.input_total, self.output_total, self.pth_total))
        ):
            X, truth = X.to(self.device), truth.to(self.device)
            X = X.detach()
            X.requires_grad = True  # for attack
            # check initial performance
            init_out = self.model(X)
            init_pred = torch.max(init_out, 1)[1].view(truth.size()).data

            # for f1 score
            ground_truth.append(truth.item())
            output_pred.append(init_pred.item())

            print("preds:", output_pred)
            print("truths:", ground_truth)
            print("-------------------------------------------------")

            # if wrong, skip
            if init_pred.item() != truth.item():
                orig_wrong += 1
                continue  # next X

            # if correct, ATTACK
            loss = self.criterion(init_out, truth)  # compute loss
            self.model.zero_grad()
            loss.backward()
            X_grad = X.grad.data

            # generate attack (single X)
            attack = self.generate(
                self.config.attack_type, X, X_grad, init_out, epsilon
            )

            # re-test
            new_out = self.model(attack)
            confidence = torch.softmax(new_out[0], dim=0)
            target_confidence = torch.max(confidence).item() * 100
            new_pred = torch.max(new_out, 1)[1].view(truth.size()).data
            if new_pred.item() == truth.item():
                atk_correct += 1
            else:
                # TODO: save successful attacks
                if self.config.save_atk:
                    pass
                    # np.save(self.config.save_atk_path + name, attack)
                    # print("saved: {}".format(name))
                pass

        return (
            ground_truth,
            output_pred,
            len(self.output_total) - orig_wrong,
            atk_correct,
        )

    def generate(self, atk, data, data_grad, init_out, eps):
        if atk is "fgsm":
            attack = self.fgsm(data, data_grad, eps)
        elif atk is "deepfool":
            attack = self.deepfool(data, init_out, self.config.max_iter)
        elif atk is "random":
            attack = self.random(data, self.config.epsilons)
        else:
            raise ("Type error. It should be one of (fgsm, deepfool, random)")
        return attack

    def fgsm(self, data, data_grad, eps):
        # ORIGINAL FGSM
        # sign_data_grad = data_grad.sign()
        # perturbed_input = data + eps * sign_data_grad
        # perturbed_input = torch.clamp(perturbed_input, 0, 127)

        # manipulation : NON ZERO ATTACK
        sign_data_grad = data_grad.sign()
        indices = torch.nonzero(data)  # get all the attack points
        perturbed_input = data + 0 * sign_data_grad
        for index in indices:
            i, j, k, l = index[0], index[1], index[2], index[3]
            orig_vel = int(data[i][j][k][l].item())  # int
            att_sign = int(sign_data_grad[i][j][k][l].item())
            if att_sign != 0:  # meaningless -> almost all nonzero
                perturbed_input[i][j][k][l] = max(
                    0, min(orig_vel + att_sign * eps, 127)
                )  # clamp

        return perturbed_input

    def deepfool(self, data, model_out, max_iter):
        indices = torch.nonzero(data)

        f_out = model_out.detach().numpy().flatten()
        I = (np.array(f_out)).argsort()[::-1]
        # index of greatest->least  ex:[2, 0, 1, 3]
        label = I[0]  # true class index

        # initialize variables
        input_shape = data.numpy().shape
        w = np.zeros(input_shape)  # (1, 129, 400, 128)
        r_tot = np.zeros(input_shape)
        loop_i = 0
        k_i = label  # initialize as true class

        perturbed_input = copy.deepcopy(data)  # copy entire tensor object
        x = perturbed_input.clone().requires_grad_(True)
        fs = self.model(x)  # forward
        print("loop", end=": ")
        while k_i == label and loop_i < max_iter:  # repeat until misclassifies
            print("{}".format(loop_i), end=" ")

            pert = np.inf  # for comparison (find min pert)
            fs[0, I[0]].backward(retain_graph=True)
            grad_orig = x.grad.data.numpy().copy()

            for k in range(
                1, len(self.config.genres)
            ):  # find distance to closest class(hyperplane)

                # get gradient of another class "k"
                fs[0, I[k]].backward(retain_graph=True)
                cur_grad = x.grad.data.numpy().copy()

                # set new w_k and new f_k (numpy)
                w_k = cur_grad - grad_orig
                f_k = (fs[0, I[k]] - fs[0, I[0]]).data.numpy()

                pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

                # determine w_k to use
                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            # compute r_i and r_tot from min(w & pert)
            r_i = (pert + 1e-4) * w / np.linalg.norm(w)

            # MANIPULATION: apply only to nonzero cells
            r_i_valid = np.zeros(input_shape)
            for index in indices:
                i, j, k = index[1], index[2], index[3]
                if r_i[0][i][j][k] != 0:
                    r_i_valid[0][i][j][k] = r_i[0][i][j][k]  # copy cell

            # scale
            r_i_scaled = np.int_(r_i_valid * 1e4)  # 1-2digit int
            # total r
            r_tot = np.float32(r_tot + r_i_scaled)  # r_tot += r_i
            # reset perturbed_input using total r
            perturbed_input = input + torch.from_numpy(r_tot)
            perturbed_input = torch.clamp(perturbed_input, 0, 127)
            # new pred
            x = perturbed_input.clone().requires_grad_(True)
            fs = self.model(x)
            k_i = np.argmax(fs.data.numpy().flatten())

            loop_i += 1

        print("")
        return perturbed_input

    def random(self, data, eps):
        # TODO: no grad, completely random attack
        return

    def tempo(self, data):
        # TODO: tempo attack is optional, implement if necessary
        return

    def draw_plot(self, acc, type):
        if type is "fgsm":
            plt.figure(figsize=(5, 5))
            plt.plot(self.config.epsilons, acc, "*-")
            plt.yticks(np.arange(0, 1.1, step=0.1))
            plt.xticks(np.arange(0, 70, step=5))
            plt.title("Accuracy vs Epsilon")
            plt.xlabel("Epsilon")
            plt.ylabel("Accuracy")
            plt.show()
        return


# if __name__ == "__main__":
#     # Testing
#     config, unparsed = get_config()
#     # for arg in vars(config):
#     #     argname = arg
#     #     contents = str(getattr(config, arg))
#     #     print(argname + " = " + contents)
#     temp = Attacker(config)
#     temp.run()
