import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import matplotlib.pyplot as plt

# from torchaudio import transforms
from torch import utils
import numpy as np
import random

# import torchsummary
from torch.optim import lr_scheduler
from os.path import *
from os import listdir
from tqdm import tqdm

from tools.data_loader import MIDIDataset
from models.resnet import resnet50
from models.convnet import CustomCNN

import copy
from sklearn.preprocessing import normalize


torch.manual_seed(123)
import torch.nn as nn


# for GPU use
import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


####################################################
genres = ["Classical", "Rock", "Country", "GameMusic"]  # best
num_genres = len(genres)
batch_size = 1  # attack

data_dir = "/data/drum/bestmodel/"  # orig path
# data_dir = "/data/drum/attack_bestmodel/" #attacked path

# vloader = torch.load('/data/drum/bestmodel/dataset/train/train_loader.pt') #orig train
vloader = torch.load("/data/drum/bestmodel/dataset/test/valid_loader.pt")  # orig valid

only_file = "scn15_11_format0.mid"

input_total = []
output_total = []
fname_total = []
for v in vloader:
    for i in range(len(v[0])):  # 20
        input_total.append(torch.unsqueeze(v[0][i], 0))  # torch [1,129,400,128]
        output_total.append(torch.unsqueeze(v[1][i], 0))  # tensor [(#)]
    fname_total.extend(v[2])


for i, e in enumerate(fname_total):
    if only_file in e:
        input_total = [input_total[i]]
        output_total = [output_total[i]]
        fname_total = [fname_total[i]]

print("==> DATA LOADED")
########################################################
###load model

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss_function = nn.CrossEntropyLoss()

model = resnet50(129, num_genres)
model.eval()
checkpoint = torch.load(
    data_dir + "Res50_valloss_0.8801_acc_81.25.pt"
)  # original model
# checkpoint = torch.load(data_dir + 'deepfool/Res50_valloss_0.7089444994926453_acc_81.13636363636364.pt') #adv training (model2)
# checkpoint = torch.load(data_dir + 'fgsm/Res50_val_TandAT_loss_0.8046790957450867_acc_81.5.pt')
model.load_state_dict(checkpoint["model.state_dict"])
print("==> MODEL LOADED")

# use GPU
# model = model.to(device)
# criterion = loss_function.to(device)
# print("==> MODEL ON GPU")
########################################################
####attack functions

#
def fgsm_attack(input, epsilon, data_grad):
    # collect element-wise "sign" of the data gradient
    sign_data_grad = data_grad.sign()
    perturbed_input = input + epsilon * sign_data_grad
    perturbed_input = torch.clamp(perturbed_input, 0, 127)
    return perturbed_input


#
def vel_attack(input, epsilon, data_grad, random):  # input -> tensor
    # FOR ZERO ATTACK - 모든 셀을 공격
    # sign_data_grad = data_grad.sign()
    # ep_mat = torch.zeros(input.shape)
    # rng = ep_mat + epsilon #all cells are epsilon
    # if(random):
    #     rn = torch.rand(1,129,400,128)
    #     rng = rn*epsilon
    #
    # perturbed_input = input + rng*sign_data_grad
    # perturbed_input = perturbed_input.round()
    # perturbed_input = torch.clamp(perturbed_input, 0, 127)
    # print(perturbed_input)

    # FOR NONZERO ATTACK - 이미 값이 있는 셀만 공격
    sign_data_grad = data_grad.sign()
    indices = torch.nonzero(input)  # get all the attack points
    perturbed_input = input + 0 * sign_data_grad
    for index in indices:
        i, j, k, l = index[0], index[1], index[2], index[3]
        orig_vel = int(input[i][j][k][l].item())  # int
        att_sign = int(sign_data_grad[i][j][k][l].item())
        if att_sign != 0:  # meaningless -> almost all nonzero
            max_vel = 127
            min_vel = 0
            rng = epsilon
            if random:
                rn = np.random.rand()
                rng = rn * epsilon  # ex) ep = 20
            perturbed_input[i][j][k][l] = orig_vel + att_sign * int(round(rng))
            if perturbed_input[i][j][k][l].item() > max_vel:
                perturbed_input[i][j][k][l] = max_vel
            if perturbed_input[i][j][k][l].item() < min_vel:
                perturbed_input[i][j][k][l] = min_vel

    return perturbed_input


def deepfool(input, out_init, max_iter, nzero, overshoot=5):
    indices = torch.nonzero(input)

    # model output (probability)
    f_out = out_init.detach().numpy().flatten()
    I = (np.array(f_out)).argsort()[::-1]  # index of greatest->least  ex:[2, 0, 1, 3]
    label = I[0]  # true class index

    # initialize variables
    input_shape = input.numpy().shape
    w = np.zeros(input_shape)  # (1, 129, 400, 128)
    r_tot = np.zeros(input_shape)
    loop_i = 0
    k_i = label  # initialize as true class

    perturbed_input = copy.deepcopy(input)  # copy entire tensor object
    x = perturbed_input.clone().requires_grad_(True)
    fs = model(x)  # forward
    # fs_list = [fs[0,I[k]] for k in range(num_genres)] #greatest -> least
    print("loop", end=": ")
    while k_i == label and loop_i < max_iter:  # repeat until misclassify
        print("{}".format(loop_i), end=" ")

        pert = np.inf  # find min perturb (comparison)
        # get true class gradient -> used in calculations
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.numpy().copy()

        for k in range(1, num_genres):  # find distance to closest class(hyperplane)

            # x.zero_grad()

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

        # FINALLY we have min w & pert (= distance to closest hyperplane)
        # now compute r_i and r_tot
        r_i = (pert + 1e-4) * w / np.linalg.norm(w)

        # manual implementation
        r_i_scaled = r_i  # initialize
        r_i_valid = np.zeros(input_shape)
        if not nzero:  # non-empty cells
            for index in indices:
                i, j, k = index[1], index[2], index[3]
                if r_i[0][i][j][k] != 0:
                    r_i_valid[0][i][j][k] = r_i[0][i][j][k]  # copy cell

            # 어택 후보4
            # r_i_sign = np.sign(r_i_valid)
            # r_i_scaled = r_i_sign*overshoot

            # 어택 후보3
            # r_i_sign = np.sign(r_i_valid)
            # # normalize abs
            # r_i_abs = np.abs(r_i_valid)
            # r_i_norm = r_i_abs - np.min(r_i_abs) / np.ptp(r_i_abs)
            # r_i_scaled = r_i_norm * overshoot * r_i_sign

            # 어택 후보2
            # #normalize to [-1,1]
            # r_i_norm = 2*(r_i_valid - np.min(r_i_valid)) / np.ptp(r_i_valid) -1  # np / (max - min)
            # #scale to [min, max]
            # r_i_scaled = r_i_norm * overshoot

            # 어택 후보1
            # scale = 10
            # count = 0
            # while(len(np.where(np.abs(r_i_valid) > 1)[0]) < ((len(np.nonzero(r_i_valid)[0]))/3)): #threshold : half
            #     print("{}th: one digits :{} nonzero :{}".format(count,len(np.where(np.abs(r_i_valid) > 1)[0]),len(np.nonzero(r_i_valid)[0])))
            #     count += 1
            #     scale *= 10
            #     r_i_valid = r_i_valid * scale
            # r_i_scaled = np.int_(r_i_valid)  # goal: 1-2digit integer

            # FINAL...
            r_i_scaled = np.int_(r_i_valid * 1e4)  # 1-2digit inte
            # r_i_scaled = r_i_valid*1e+4

        r_tot = np.float32(r_tot + r_i_scaled)  # r_tot += r_i

        # reset perturbed_input
        perturbed_input = input + torch.from_numpy(r_tot)
        perturbed_input = torch.clamp(perturbed_input, 0, 127)

        x = perturbed_input.clone().requires_grad_(True)
        fs = model(x)
        k_i = np.argmax(fs.data.numpy().flatten())  # new pred

        loop_i += 1

    print("")
    # double check
    # perturbed_input = torch.clamp(perturbed_input, 0, 127)
    r_tot = np.clip(np.abs(r_tot), 0, 127)
    return r_tot, loop_i, k_i, perturbed_input


def tempo_edge_attack(input, eps, data_grad):
    sign_data_grad = data_grad.sign()

    # indices = torch.nonzero(input, as_tuple=True)  # get all the attack points
    indices = torch.nonzero(input, as_tuple=False)
    indices_np = np.array(indices)
    cur_ch = indices_np[0][1]
    perturbed_input = input.numpy().copy()

    save_pair = []
    # for i, index in enumerate(indices_np):
    # for i in range(1, len(indices_np)-1):
    for i in range(len(indices_np)):  # 0 - len-1

        while i < len(indices_np) and indices_np[i][1] == cur_ch:

            start, end = indices_np[i], indices_np[i]
            pitch = indices_np[i][3]
            i += 1
            while i < len(indices_np) and indices_np[i][3] == pitch:
                end = indices_np[i]  # cur pos
                i += 1  # update i -> next pos

            # check is len > 2
            if start[2] < end[2] and start[3] == end[3]:
                save_pair.append([start, end])  # save pair
                # print("{}: {}".format(i,[start, end]))

            if i < (len(indices_np) - 1):
                cur_ch = indices_np[i][1]
            else:
                cur_ch = -1

    ########## manipulate!! ###########
    for pair in save_pair:
        begin, until = pair[0], pair[1]
        b_vel = perturbed_input[0][begin[1]][begin[2]][begin[3]]
        u_vel = perturbed_input[0][until[1]][until[2]][until[3]]
        rng = eps
        if begin[2] > rng and until[2] < (400 - rng):
            for i in range(rng):
                perturbed_input[0][begin[1]][begin[2] - i][begin[3]] = b_vel
                perturbed_input[0][until[1]][until[2] + i][until[3]] = u_vel

    perturbed_input = torch.from_numpy(perturbed_input)
    return perturbed_input


##########################################################################


def test(model, epsilon):

    adv_rounded = []
    adv_fname = []
    correct = 0
    orig_wrong = 0

    # for i,val in enumerate(tqdm(val_loader)):
    for i, (v_in, v_out, v_fn) in enumerate(
        zip(input_total, output_total, fname_total)
    ):
        model.eval()
        data, target, name = v_in, v_out, v_fn.replace("/", "_")

        # data, target = data.to(device), target.to(device)
        data = data.detach()
        data.requires_grad = True  # for attack
        init_out = model(data)
        init_pred = torch.max(init_out, 1)[1].view(target.size()).data
        # _, init_pred = torch.max(init_out.data, 1)

        # if correct, skip
        if init_pred.item() != target.item():
            orig_wrong += 1
            # print("{}: wrong! --- pred:{} orig:{}]".format(i,init_pred.item(),target.item()))
            continue
        # print("{}: correct!--- pred:{} orig:{}]".format(i,init_pred.item(),target.item()))

        # if wrong, ATTACK
        loss = loss_function(init_out, target)  # compute loss
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        # VEL ATTACKS
        # perturbed_data = vel_attack(data, epsilon, data_grad, True) # vel --> random attack
        # perturbed_data = vel_attack(data, epsilon, data_grad, False) # vel --> sign grad attack
        # perturbed_data = tempo_edge_attack(data.detach(), epsilon, data_grad) #tempo edge attack
        # RERUN MODEL
        # new_out = model(perturbed_data)
        # confidence = torch.softmax(new_out[0], dim=0)
        # target_confidence = torch.max(confidence).item() * 100
        # new_pred = torch.max(new_out, 1)[1].view(target.size()).data

        # get orig data
        # orig_data = data.squeeze().numpy()
        # np_vel = perturbed_data.squeeze().detach().numpy() #(129 400 128)

        # check for success
        # if new_pred.item() == target.item():
        #     correct += 1
        # else:
        #     pass
        # np.save("/data/attack_test/pitch_" + each_file, np_pitch)  # save as .npy
        #     np.save("/data/attack_test/time_" + each_file, np_time)  # save as .npy
        #     np.save("/data/simulation/vel_" +  name, np_vel)  # save as .npy | "_[" + str(epsilon) +"]"+
        #     np.save("/data/attack_test/vel_" + each_file + "_[0]" , orig_data)  # save as .npy

        # DEEPFOOL(LAST)
        nonzero = False
        data_nograd = data.detach()  # lose gradient
        r_tot, iter_num, wrong_pred, perturbed_data = deepfool(
            data_nograd, init_out, 10, nonzero
        )
        perturbed_data = np.int_(perturbed_data.squeeze().numpy())  # .astype(np.int8)
        r_tot = np.int_(r_tot.squeeze())  # .astype(np.int8)
        if wrong_pred != target:  # check for success
            # pass #success
            np.save("/data/sim_df/vel_" + name, perturbed_data)  # save as .npy
            # np.save("/data/attacks/vel_deepfool2/valid/vel_" + str(name), perturbed_data) #save as .npy
            # np.save("/data/attacks/vel_deepfool2/valid_noise/noise_" + str(name), r_tot)
        else:  # failed to attack
            correct += 1

        # print("{}th data: {}".format(i+1, name))
        # print("{}".format(i+1))
    # print("files saved!")
    return adv_rounded, adv_fname, orig_wrong, correct


##########################################################
# run attack

accuracies = []
# epsilons = [0,2,4,6,8,10,12,14,16,18,20, 22, 24, 26, 28, 30,32,34,36,38,40] #for vel attacks
# epsilons = [5,10,15,20,25,30,35,40,45,50,55,60,65,70]
epsilons = [20]
# epsilons = [3,4,5,6] #for tempo
# epsilons = [0] #for deepfool
for ep in tqdm(epsilons):
    #
    rounded, fname, orig_wrong, correct = test(model, ep)
    denom = len(input_total)
    orig_acc = (denom - orig_wrong) / denom
    final_acc = correct / float(denom)
    print("Epsilon: {}".format(ep))
    print("Before: {} / {} = {}".format(denom - orig_wrong, denom, orig_acc))
    print("After: {} / {} = {}".format(correct, denom, final_acc))
    #
    #     #for plt
    accuracies.append(final_acc)
    # examples.append(ex)


# Draw Results
# plt.figure(figsize=(5,5))
# plt.plot(epsilons, accuracies, "*-")
# plt.yticks(np.arange(0, 1.1, step=0.1))
# plt.xticks(np.arange(0, 7, step=1))
# plt.title("Accuracy vs +- Cell Range")
# plt.xlabel("Cell Range")
# plt.ylabel("Accuracy")
# plt.show()

# Draw Results
# plt.figure(figsize=(5,5))
# plt.plot(epsilons, accuracies, "*-")
# plt.yticks(np.arange(0, 1.1, step=0.1))
# plt.xticks(np.arange(0, 70, step=5))
# plt.title("Accuracy vs Epsilon")
# plt.xlabel("Epsilon")
# plt.ylabel("Accuracy")
# plt.show()
