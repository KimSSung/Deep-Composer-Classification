import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import sklearn
import torchaudio
import torch
import matplotlib.pyplot as plt
from torchaudio import transforms
from torch import utils
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
# import torchsummary
from torch.optim import lr_scheduler
from os.path import *
from os import *

from ResNet import resnet18


#torch.manual_seed(123)
import torch.nn as nn


#for GPU use
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


##########################################################
genres = ['Classical','Rock', 'Country'] #best
num_genres = 3
# min_shape= 820
batch_size = 1 #attack

###load input & target

num_files = 50
input_total=[]
output_total=[]
filename_total = []
for genre in genres:
    genre_dir = "/data/temp/genres/" + genre
    count = 0
    for f in listdir(genre_dir):
        load_saved = np.load(genre_dir+"/"+f, allow_pickle=True)
        input_total.append(load_saved)
        filename_total.append(genre+"/"+f)
        output_total.append(genres.index(genre))
        count += 1
        if(count >= num_files): break


X_np = np.array(input_total)
Y_np = np.array(output_total)
F_np = np.array(filename_total)

##shuffle
data = list(zip(X_np, Y_np, F_np)) #zip data structure
random.shuffle(data)
X,Y, FN = zip(*data)
X,Y = np.asarray(X), np.asarray(Y)

X_data = torch.from_numpy(X).type(torch.Tensor)
Y_data = torch.from_numpy(Y).type(torch.LongTensor)

# tensorDataset
t = TensorDataset(X_data, Y_data)

# create batch (= 1)
data_loader = DataLoader(t, batch_size=batch_size)

print("==> DATA LOADED")
print(X.shape)
print(Y.shape)

########################################################
####load model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = '/data/midi820_128ch/model/'

loss_function = nn.CrossEntropyLoss()

model = resnet18(128, 3)
model.eval()
checkpoint = torch.load('/data/midi820_128ch/model/bestmodel/ResNet18_valloss_0.6343_acc_83.33.pt')
model.load_state_dict(checkpoint['model.state_dict'])
print("==> MODEL LOADED")

# use GPU
model = model.to(device)
criterion = loss_function.to(device)
print("==> MODEL ON GPU")
########################################################
####attack functions


def fgsm_attack(input, epsilon, data_grad):
    #collect element-wise "sign" of the data gradient
    sign_data_grad = data_grad.sign()
    # print(torch.nonzero(sign_data_grad, as_tuple=True))
    indices = torch.nonzero(sign_data_grad, as_tuple=True)
    # i,j,k = indices[1], indices[2], indices[3]
    for count, (i,j,k) in enumerate(zip(indices[1], indices[2], indices[3])):
        print(input[0][i.item()][j.item()][k.item()].item())


    perturbed_input = input + epsilon*sign_data_grad
    return perturbed_input

def pitch_attack(input, epsilon, data_grad):
    sign_data_grad = data_grad.sign()

    perturbed_input =

    return perturbed_input

def vel_attack(input, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_input =

    return perturbed_input

def time_attack(input, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_input =

    return perturbed_input

def tempo_attack(input, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_input =

    return perturbed_input


def test(model, data_loader, files, epsilon):

    adv_rounded = []
    adv_fname = []
    correct = 0
    orig_wrong = 0

    for i, ((data, target), each_file) in enumerate(zip(data_loader, files)):
        model.eval()

        data, target = data.to(device), target.to(device)
        data.requires_grad = True #for attack
        init_out = model(data)
        init_pred = torch.max(init_out, 1)[1].view(target.size()).data
        # _, init_pred = torch.max(init_out.data, 1)

        #if correct, skip
        if(init_pred.item() != target.item()):
            orig_wrong += 1
            # print("{}: correct! --- pred:{} orig:{}]".format(counter,init_pred.item(),target.item()))
            continue
        # print("{}: wrong!--- pred:{} orig:{}]".format(counter,init_pred.item(),target.item()))

        #if wrong, ATTACK

        loss = loss_function(init_out, target)  # compute loss
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        #generate perturbed input
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        new_out = model(perturbed_data)
        new_pred = torch.max(new_out, 1)[1].view(target.size()).data

        # adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
        # orig_data = data.detach().cpu().numpy()

        #check for success
        if new_pred.item() == target.item():
            correct += 1
        # else:
        #
        #     np_pitch = pitch_attack(data, epsilon, data_grad)
        #     np_time = time_attack(data, epsilon, data_grad)
        #     np_vel = vel_attack(data, epsilon, data_grad)
        #     # save attacks
        #     print("\nsaving...\n")
        #     np.save("/data/midi820_128att/ep_" + str(ep) + "_pitch_" + fname, np_pitch)  # save as .npy
        #     np.save("/data/midi820_128att/ep_" + str(ep) + "_time_" + fname, np_time)  # save as .npy
        #     np.save("/data/midi820_128att/ep_" + str(ep) + "_vel_" + fname, np_vel)  # save as .npy
        #     np.save("/data/midi820_128att/ep_" + str(ep) + "_orig_" + fname, data)  # save as .npy

            #recheck accuracy

    return adv_rounded, adv_fname, orig_wrong, correct


##########################################################
#run attack

accuracies = []
examples = []
# epsilons = [.005, .01, .015, .020, .025, .03, 0.035, 0.04, 0.045, 0.05, 0.055]
epsilons = [.03]

for ep in epsilons:

    rounded, fname, orig_wrong, correct = test(model, data_loader, FN, ep)
    orig_acc = (len(data_loader) - orig_wrong) / len(data_loader)
    final_acc = correct / float(len(data_loader))
    print("Epsilon: {}".format(ep))
    print("Before: {} / {} = {}".format(len(data_loader) - orig_wrong, len(data_loader), orig_acc))
    print("After: {} / {} = {}".format(correct, len(data_loader), final_acc))

    #for plt
    # accuracies.append(final_acc)
    # examples.append(ex)

    #save attacks
    # print("\nsaving...\n")
    # np.save("/data/midi820_128att/ep_"+str(ep)+"_pitch_"+fname, np_pitch)  # save as .npy
    # np.save("/data/midi820_128att/ep_"+str(ep)+"_time_"+fname, np_time)  # save as .npy
    # np.save("/data/midi820_128att/ep_"+str(ep)+"_vel_"+fname, np_vel)  # save as .npy
    # np.save("/data/midi820_128att/ep_"+str(ep)+"_orig_"+fname, np_orig)  # save as .npy


#Draw Results
# plt.figure(figsize=(5,5))
# plt.plot(epsilons, accuracies, "*-")
# plt.yticks(np.arange(0, 1.1, step=0.1))
# plt.xticks(np.arange(0, .055, step=0.005))
# plt.title("Accuracy vs Epsilon")
# plt.xlabel("Epsilon")
# plt.ylabel("Accuracy")
# plt.show()

