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
from models.wresnet import wide_resnet50_2, wide_resnet101_2
from models.wresnet import resnet18, resnet34, resnet50, resnet101, resnet152

# from models.resnet_ver2 import resnet18, resnet34, resnet101, resnet152, resnet50
from _collections import OrderedDict
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from datetime import date, datetime
from tools import visualization
from tools.detector import Detector


class Attacker:
    def __init__(self, args):
        self.config = args

        # for GPU use #TODO: remove
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.label_num = self.config.composers
        self.input_shape = (2, 400, 128)
        self.seg_num = 90

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

        self.date = date.today().strftime("%m-%d") + datetime.now().strftime("-%H-%M")
        self.epsilons = [float(e) for e in self.config.epsilons.split(",")]
        print("==> ATTACK {}".format(self.config.attack_type), end=' ')
        if self.config.attack_type == "fgsm": print("eps: ", self.epsilons)
        else: print("var:", self.config.variable)
        print("==> TARGET LABEL: {}".format(self.config.target_label))
        print("==> SAVE {} at {}".format(self.config.save_atk, self.date))

    def data_load(self, orig):
        if orig:
            self.data_loader = torch.load(
                self.config.load_path + "dataset/valid/valid_loader.pt"
            )
            print("attack on VALID")
        else:
            pass  # TODO: adv attack on attack
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
            for i in range(len(v["Y"])):
                self.input_total.append(torch.unsqueeze(v["X"][i], 0))
                # unsqueeze -> torch [1,2,400,128]
                self.output_total.append(torch.unsqueeze(v["Y"][i], 0))
                # unsqueeze -> tensor [(#)]
            self.pth_total.extend(v["pth"])
        return

    def get_model(self):
        self.model_fname = os.listdir(self.config.load_path + "model/")[0]
        self.model_type = self.model_fname.split("_")[0]  # ex: resnet50
        if self.model_type == "resnet18":
            return resnet18(int(self.input_shape[0]), self.label_num)
        elif self.model_type == "resnet34":
            return resnet34(int(self.input_shape[0]), self.label_num)
        elif self.model_type == "resnet50":
            return resnet50(int(self.input_shape[0]), self.label_num)
        elif self.model_type == "resnet101":
            return resnet101(int(self.input_shape[0]), self.label_num)
        elif self.model_type == "resnet152":
            return resnet152(int(self.input_shape[0]), self.label_num)
        elif self.model_type == "wresnet50":
            return wide_resnet50_2(
                in_channels=int(self.input_shape[0]), num_classes=self.label_num
            )
        elif self.model_type == "wresnet101":
            return wide_resnet101_2(
                in_channels=int(self.input_shape[0]), num_classes=self.label_num
            )

    def model_load(self):
        checkpoint = torch.load(
            self.config.load_path + "model/" + str(self.model_fname)
        )
        self.model.load_state_dict(checkpoint["model.state_dict"])
        print("==> MODEL LOADED: {}".format(self.model_type))
        self.model.eval()
        # self.model = nn.DataParallel(self.model).to(self.device)
        self.model.to(self.device)
        print("==> MODEL ON GPU")
        return

    def run(self):
        self.accuracies = []
        for ep in tqdm(self.epsilons):  # iteration: fgsm(n) others(1)
            truths, init_preds, new_preds, orig_correct, atk_correct = self.test(ep)
            # all batch unit
            orig_acc = orig_correct / len(truths)
            atk_acc = atk_correct / len(truths)

            init_w_f1score = f1_score(truths, init_preds, average="weighted")
            init_precision, init_recall, f1, supports = precision_recall_fscore_support(
                truths,
                init_preds,
                average=None,
                labels=list(range(13)),
                warn_for=tuple(),
            )
            new_w_f1score = f1_score(truths, new_preds, average="weighted")
            new_precision, new_recall, f1, supports = precision_recall_fscore_support(
                truths,
                new_preds,
                average=None,
                labels=list(range(13)),
                warn_for=tuple(),
            )

            print("Epsilon: {}".format(ep))
            print("saved {} items".format(orig_correct - atk_correct))
            print("#########Before########")
            print(
                "Accuracy: {} / {} = {:4f}".format(orig_correct, len(truths), orig_acc)
            )
            print("F1 score: {:4f}".format(init_w_f1score))
            print("{:<30}{:<}".format("Precision", "Recall"))
            for p, r in zip(init_precision, init_recall):
                print("{:<30}{:<}".format(p, r))

            print("\n#########After#########")
            print("Accuracy: {} / {} = {:4f}".format(atk_correct, len(truths), atk_acc))
            print("F1 score: {:4f}".format(new_w_f1score))
            print("{:<30}{:<}".format("Precision", "Recall"))
            for p, r in zip(new_precision, new_recall):
                print("{:<30}{:<}".format(p, r))

            if self.config.confusion:
                self.draw_confusion_matrix(truths, new_preds)

            self.accuracies.append(atk_acc)

        # draw plot
        if self.config.plot:
            self.draw_plot(self.accuracies, self.config.attack_type)

        return

    def test(self, epsilon):  # call this function to run attack
        orig_wrong = 0
        atk_correct = 0

        # for f1 score
        ground_truth = []
        init_pred_history, new_pred_history = [], []
        init_preds, new_preds = [], []

        # re-initialize every validation
        init_out_history, new_out_history = [], []
        X_history, pth_history, attack_history = [], [], []

        for i, (X, truth, pth) in enumerate(
            tqdm(zip(self.input_total, self.output_total, self.pth_total))
        ):
            X, truth = X.to(self.device), truth.to(self.device)
            X = X.detach() # remove graph from data_load
            X.requires_grad = True  # reinitialize for attack

            # check initial performance
            init_out = self.model(X)
            init_pred = torch.max(init_out, 1)[1].view(truth.size()).data

            # for batch-unit validation
            init_out_history.append(init_out.tolist())
            init_pred_history.append(init_pred.item())

            # if wrong, do nothing
            if init_pred.item() != truth.item():
                new_out_history.append(init_out.tolist())
                new_pred_history.append(init_pred.item())

            else:  # if correct, ATTACK
                # untargeted
                if self.config.target_label is None:
                    loss = self.criterion(init_out, truth)  # compute loss
                # targeted
                else:
                    if self.config.target_label in range(self.label_num):
                        target = torch.tensor([self.config.target_label])
                        target = target.to(self.device)
                        loss = self.criterion(init_out, target)  # compute loss
                    else:
                        raise Exception('Incorrect target label. Should be in rang[0,{}]'.format(self.label_num))

                self.model.zero_grad()
                loss.backward()
                X_grad = X.grad.data

                # generate attack (single 'X')
                attack = self.generate(
                    self.config.attack_type, X, X_grad, init_out, epsilon
                )
                attack_history.append(attack)

                # re-test
                new_out = self.model(attack)
                new_pred = torch.max(new_out, 1)[1].view(truth.size()).data

                # for batch-unit validation
                X_history.append(X)
                pth_history.append(pth)
                new_out_history.append(new_out.tolist())
                new_pred_history.append(new_pred.item())

            ##new validation
            if (i + 1) % self.seg_num == 0:  # ex) every 90
                seq = int(i / self.seg_num)  # 0-89 = seq0
                init_batch_hist = init_pred_history[(i - self.seg_num) : i]
                new_batch_hist = new_pred_history[(i - self.seg_num) : i]
                init_batch_pred = self.get_batch_pred(init_out_history, init_batch_hist)
                new_batch_pred = self.get_batch_pred(new_out_history, new_batch_hist)

                # for acc calc
                ground_truth.append(truth.item())
                init_preds.append(init_batch_pred)
                new_preds.append(new_batch_pred)

                if init_batch_pred != truth.item():  # intially wrong
                    orig_wrong += 1
                elif new_batch_pred != truth.item():  # attack successful
                    if self.config.save_atk:  # save attacks
                        for i, (xi, atk, path) in enumerate(
                            zip(X_history, attack_history, pth_history)
                        ):
                            self.save_attack(xi, atk, i, path, epsilon)
                else:  # attack unsuccessful
                    atk_correct += 1

                print(
                    "   {}th true: {} | init_pred: {} | new_pred: {}".format(
                        seq+1, truth.item(), init_batch_pred, new_batch_pred
                    )
                )

                # re-initialize
                init_out_history, new_out_history = [], []
                X_history, pth_history, attack_history = [], [], []

        return (
            ground_truth,
            init_preds,
            new_preds,
            len(ground_truth) - orig_wrong,
            atk_correct,
        )

    def get_batch_pred(self, out_history, pred_history):
        out_history = torch.tensor(out_history).squeeze()  # (90,13)
        out_scaled = torch.softmax(out_history, dim=1)
        confidence = torch.sum(out_scaled, dim=0)
        confidence = torch.div(confidence, self.seg_num)  # avg

        occ = [pred_history.count(x) for x in range(self.label_num)]
        max_vote = max(occ)
        occ = np.array(occ)
        dup_list = np.where(max_vote == occ)[0]
        if len(dup_list) > 1:
            max_confidence = -1.0
            for dup in dup_list:
                if confidence[dup] > max_confidence:
                    batch_prediction = dup
        else:
            batch_prediction = max(pred_history, key=pred_history.count)

        return batch_prediction

    def generate(self, atk, data, data_grad, init_out, eps):
        if atk == "random":
            attack = self.random(data, rndness=self.config.variable)
        elif atk == "fgsm":
            attack = self.fgsm_original(data, data_grad, eps)
        elif atk == "fgsm_nonzero":
            attack = self.fgsm_nonzero(data, data_grad, eps)
        elif atk == "deepfool":
            attack = self.deepfool(data, init_out, self.config.max_iter)
        elif atk == "column":
            attack = self.notes_by_col(data,data_grad, notes=int(self.config.variable))
        elif atk == "chord":
            attack = self.chord_attack(data, data_grad, dur=int(self.config.variable))
        elif atk == "melody_no_change":
            attack = self.melody_no_change(data, data_grad, dur=int(self.config.variable))
        else:
            raise Exception("Type error. Please use valid attack names")
        return attack

    def random(self, data, rndness, vel=40):
        # no grad, completely random attack
        factor = torch.full((400,128), rndness)
        rndarray = torch.bernoulli(factor).to(self.device)
        perturbed_input = data.detach().clone()  # copy data
        perturbed_input[0][1] = data[0][1] + vel * rndarray

        perturbed_input = torch.clamp(perturbed_input, min=0, max=128)
        return perturbed_input

    def notes_by_col(self, data, data_grad, notes):
        pos_data_grad = torch.clamp(data_grad, min=0) # positive values
        perturbed_input = data.detach().clone()  # copy data
        nonzero_x = torch.unique(torch.nonzero(perturbed_input[0][1]))

        for column in nonzero_x:  # nonzero column
            idx = torch.topk(pos_data_grad[0][1][column], k=notes, dim=0)[1]  # top k gradients
            perturbed_input[0][1][column][idx] += 70

        perturbed_input = torch.clamp(perturbed_input, min=0, max=128)
        return perturbed_input

    def fgsm_original(self, data, data_grad, eps):
        sign_data_grad = data_grad.sign()
        perturbed_input = data / 128  # normalize 0-1
        perturbed_input = perturbed_input + eps * sign_data_grad
        perturbed_input = perturbed_input * 128  # amplify back to 0-128

        perturbed_input = torch.clamp(perturbed_input, min=0, max=128)
        return perturbed_input

    def fgsm_nonzero(self, data, data_grad, eps):
        sign_data_grad = data_grad.sign()
        indices = torch.nonzero(data[0][1])  # only attack channel[1]
        perturbed_input = data + 0 * sign_data_grad
        for index in indices:
            x, y = index[0], index[1]
            orig_vel = int(data[0][1][x][y].item())  # int
            att_sign = int(sign_data_grad[0][1][x][y].item())
            if att_sign != 0:  # meaningless -> almost all nonzero
                scaled_att_vel = orig_vel / 128 + att_sign * eps
                perturbed_input[0][1][x][y] = max(0, min(128 * scaled_att_vel, 128))
                # clamp

        perturbed_input = torch.clamp(perturbed_input, min=0, max=128)
        return perturbed_input

    def chord_attack(self, data, data_grad, dur, vel=40):
        # gpu tensor to cpu numpy
        data1 = data.detach().cpu().clone().numpy()
        data_grad1 = data_grad.detach().cpu().clone().numpy()

        chords = Detector(data1, dur).run()
        signs = np.sign(data_grad1)
        pos_signs = np.where(signs < 0.0, 0.0, signs)
        perturbed_input = data1 + np.multiply(chords, pos_signs * vel)

        # cpu numpy to gpu tensor
        perturbed_input = torch.tensor(perturbed_input, dtype=torch.float).to(self.device)
        return torch.clamp(perturbed_input, min=0, max=128)

    def last_nonzero(self, arr, axis, invalid_val=-1):
        mask = arr != 0
        val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
        return np.where(mask.any(axis=axis), val, invalid_val)

    def melody_no_change(self, data, data_grad, dur, vel = 40):
        data1 = data.detach().cpu().clone().numpy()
        data_grad1 = data_grad.detach().cpu().clone().numpy()

        melody_np = self.last_nonzero(data1[0][1], axis=1)
        melody_np = melody_np.squeeze()
        chords = Detector(data1, dur).run()
        for time,melody_note in enumerate(melody_np):
            if melody_note == -1:
                continue
            chords[0,1,time,melody_note+1:] = 0
        signs = np.sign(data_grad1)
        pos_signs = np.where(signs < 0.0, 0.0, signs)
        perturbed_input = data1 + np.multiply(chords, pos_signs * vel)

        # cpu numpy to gpu tensor
        perturbed_input = torch.tensor(perturbed_input, dtype=torch.float).to(self.device)
        return torch.clamp(perturbed_input, min=0, max=128)

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


    def save_attack(self, orig, attack, idx, path, eps):
        save_dir = (
            self.config.save_path + self.config.attack_type + "/" + self.date + "/"
        )  # attacks/fgsm/[date]/ep0.1/
        if self.config.attack_type == "fgsm":
            if eps == 0.0:
                return
            save_dir += "ep" + str(eps) + "/"

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_name = path.replace("/", "_").replace(".npy", "") + "_seg" + str(idx)
        # save orig
        np.save(
            save_dir + "orig_" + save_name, orig.cpu().detach().numpy(),
        )
        # save attack
        np.save(
            save_dir + "atk_" + save_name, attack.cpu().detach().numpy(),
        )
        return

    def draw_plot(self, acc, type):
        if type == "fgsm":
            plt.figure(figsize=(5, 5))
            plt.plot(self.epsilons, acc, "*-")
            plt.yticks(np.arange(0, 1.1, step=0.1))
            plt.xticks(np.arange(0, 70, step=5))
            plt.title("Accuracy vs Epsilon")
            plt.xlabel("Epsilon")
            plt.ylabel("Accuracy")
            plt.show()
        return

    def draw_confusion_matrix(self, true, pred):
        temp = visualization.Visualization(sort=True, normalize=True)
        temp.generate_matrix(true, pred)
        # print("confusion matrix saved at: {}".format())
        return


if __name__ == "__main__":
    # Testing
    config, unparsed = get_config()
    # for arg in vars(config):
    #     argname = arg
    #     contents = str(getattr(config, arg))
    #     print(argname + " = " + contents)
    temp = Attacker(config)
    temp.run()
