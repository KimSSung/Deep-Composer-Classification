import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch
import matplotlib.pyplot as plt
from torchaudio import transforms
from torch import utils
import numpy as np
import random
# import torchsummary
from torch.optim import lr_scheduler
from os.path import *
from os import listdir
from tqdm import tqdm
from MIDIDataset import MIDIDataset
from MyResNet import resnet50
# from conv import CustomCNN
import copy


torch.manual_seed(123)
import torch.nn as nn


#for GPU use
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"



####################################################
genres = ['Classical','Rock', 'Country', 'GameMusic'] #best
num_genres = len(genres)
batch_size = 1 #attack

data_dir = "/data/drum/bestmodel/"
each_num = 300
v = MIDIDataset('/data/midi820_400/', genres, each_num * 0.8, each_num)
val_loader = DataLoader(v, batch_size=batch_size, shuffle=False)

print("==> DATA LOADED")

########################################################
###load model

device = torch.device("cuda")

loss_function = nn.CrossEntropyLoss()

model = resnet50(129, num_genres)
model.eval()
checkpoint = torch.load(data_dir + 'Conv_valloss_0.8801_acc_81.25.pt')
model.load_state_dict(checkpoint['model.state_dict'])
print("==> MODEL LOADED")

# use GPU
model = model.to(device)
criterion = loss_function.to(device)
print("==> MODEL ON GPU")
########################################################
####attack functions

#
def fgsm_attack(input, epsilon, data_grad):
    #collect element-wise "sign" of the data gradient
    sign_data_grad = data_grad.sign()
    perturbed_input = input + sign_data_grad
    return perturbed_input

#
def vel_attack(input, epsilon, data_grad):
    # get all the non-zero indices
    sign_data_grad = data_grad.sign()
    indices = torch.nonzero(input) #get all the attack points
    perturbed_input = input + 0*sign_data_grad
    for index in indices:
        i, j, k, l = index[0], index[1], index[2], index[3]
        orig_vel = int(input[i][j][k][l].item()) #int
        att_sign = int(sign_data_grad[i][j][k][l].item())
        if(att_sign != 0): #meaningless -> almost all nonzero
            # rn = np.random.rand()
            max_vel = 127
            min_vel = 0
            # rng = rn*epsilon #20
            rng = 20
            perturbed_input[i][j][k][l] = orig_vel + att_sign * int(round(rng))
            if(perturbed_input[i][j][k][l].item() > max_vel): perturbed_input[i][j][k][l] = max_vel
            if(perturbed_input[i][j][k][l].item() < min_vel): perturbed_input[i][j][k][l] = min_vel

    return perturbed_input

#선종아 여기함수를 채우면 된단다!
def tempo_attack(input, epsilon, data_grad):
    MAX_SIZE = 400
    MAX_NUMPY_SIZE = 400
    sign_data_grad = data_grad.sign()
    # input : 4-dimension array
    # epsilon : Random Pertubation
    #data_grad: Apply data gradients
    perturbed_input = input.copy()
    for song_num in range(0,len(input)):

        current_time = 0
        extend_time_fraction = []
        multiplier_list = []
        frac_time_list = []

        # Find what section will be changed
        while current_time < MAX_SIZE:
            pass_time = random.randint(10, 20)
            if ((current_time + pass_time) > MAX_SIZE):
                break
            current_time += pass_time
            multiplier = random.randint(3, 7)
            frac_time = random.randint(1, 5)
            if ((current_time + multiplier * frac_time) > MAX_SIZE):
                break
            extend_time_fraction.append([current_time, current_time + multiplier * frac_time])
            multiplier_list.append(multiplier)
            frac_time_list.append(frac_time)
            current_time += (multiplier * frac_time)
        total_frac_time =0
        for total in frac_time_list:
            total_frac_time+=total

        total_extend_length = 400
        total_shorten_length = 400
        #Update extend_time_fraction index that delayed
        # temp2 = copy.deepcopy(extend_time_fraction)
        # t_fraction = []
        for manipul in range(0,len(extend_time_fraction)):
            if manipul == 0:
                continue
            else:
                for fib in range(1,manipul+1):
                    extend_time_fraction[manipul][0] += (frac_time_list[fib]* (multiplier_list[fib]-1))
                    extend_time_fraction[manipul][1] += (frac_time_list[fib]* (multiplier_list[fib]-1))
                    # if manipul== len(extend_time_fraction)-1:
                    #     total_extend_length += (extend_time_fraction[fib-1][1] - extend_time_fraction[fib-1][0])
                    #     if fib ==manipul:
                    #         total_extend_length += (extend_time_fraction[fib][1] - extend_time_fraction[fib][0])
        #Update shorten_time_fraction index that delayed
        random_list =[i for i in range(1,400-total_frac_time-10)]
        gathering_index_list = random.sample(random_list,len(extend_time_fraction))
        gathering_index_list.sort()
        shorten_time_fraction = []
        for dec_idx in range(0,len(extend_time_fraction)):
            shorten_time_fraction.append([gathering_index_list[dec_idx],gathering_index_list[dec_idx]+frac_time_list[dec_idx]*multiplier_list[dec_idx]])

        for manipul in range(0,len(extend_time_fraction)):
            if manipul ==0:
                continue
            else:
                for fib in range(1,manipul+1):
                    shorten_time_fraction[manipul][0] += (shorten_time_fraction[fib-1][1] - shorten_time_fraction[fib-1][0])
                    shorten_time_fraction[manipul][1] += (shorten_time_fraction[fib-1][1] - shorten_time_fraction[fib-1][0])
                    # if manipul == len(extend_time_fraction)-1:
                    #     total_shorten_length +=(shorten_time_fraction[fib-1][1] - shorten_time_fraction[fib-1][0])
                    #     if fib == manipul:
                    #         total_shorten_length +=(shorten_time_fraction[fib][1] - shorten_time_fraction[fib][0])
        origin_shorten_time = copy.deepcopy(shorten_time_fraction)
        # print('Shorten time fraction lnegth: ' , origin_shorten_time, 'Extend time fraction length: ',extend_time_fraction)
        # print('Shorten Time:',total_shorten_length,'Extend time:',total_extend_length)


        #for channel, gather all for the same  time series
        for channel in range(0,len(input[song_num])):
            checking = []
            checking_extend = []
            checking_new_numpy_length = []
            new_extended_numpy = perturbed_input[song_num][channel].copy()
            # Extend length of the song notes for same time series
            for idx,time_index in enumerate(extend_time_fraction):
                temp  = new_extended_numpy.copy()
                before_numpy = temp[:time_index[0]][:]
                append_numpy = temp[time_index[0]:time_index[0]+frac_time_list[idx]][:]
                after_numpy = temp[time_index[0]+frac_time_list[idx]:][:]

                # Concatenate three parts or numpy
                new_numpy = before_numpy
                for extend_index in range(0, frac_time_list[idx]):
                    for mul in range(0, multiplier_list[idx]):
                        new_numpy = np.vstack([new_numpy, append_numpy[extend_index][:]])
                        # print(len(new_numpy))
                    # print(len(new_numpy))
                new_numpy = np.vstack([new_numpy, after_numpy])
                # print(len(after_numpy))
                # print(len(new_numpy))
                checking_extend.append([len(before_numpy),len(append_numpy),len(after_numpy),len(new_numpy)])
                new_extended_numpy = new_numpy.copy()
            # print('New extended numpy: ',len(new_extended_numpy))


            #new_decreased_numpy : init 400 length
            shorten_time_fraction = copy.deepcopy(origin_shorten_time)
            for idx,time_index in enumerate(shorten_time_fraction):

                #for shorten_time_fraction index , consider shortened part
                if idx==0:
                    # print('Original time_index: ',shorten_time_fraction)
                    temp = new_extended_numpy.copy()
                    before_numpy = temp[:time_index[0]][:]
                    gathered_numpy = temp[time_index[0]:time_index[1]][:]
                    # print('Original gathered numpy len: ',len(gathered_numpy))
                    after_numpy = temp[time_index[1]:][:]
                    checking.append([len(before_numpy), len(gathered_numpy), len(after_numpy)])
                else:
                    # print('Original time_index: ',shorten_time_fraction)

                    for decrease in range(1,idx+1):
                        time_index[0] -= (frac_time_list[decrease-1] * (multiplier_list[decrease-1] - 1))
                        time_index[1] -= (frac_time_list[decrease-1] * (multiplier_list[decrease-1] - 1))
                    # print('Shorten_time_difference: ', shorten_time_fraction)

                    # print('Current idx :',idx)
                    # print('index 0: ',time_index[0],'index 1: ',time_index[1])
                    temp = new_extended_numpy.copy()
                    before_numpy = temp[:time_index[0]][:]
                    gathered_numpy = temp[time_index[0]:time_index[1]][:]
                    # print('Original gathered numpy len: ',len(gathered_numpy))
                    after_numpy = temp[time_index[1]:][:]
                    checking.append([len(before_numpy),len(gathered_numpy),len(after_numpy)])

                # Concatenate three parts or numpy
                new_numpy = before_numpy
                for append_index in range(0, frac_time_list[idx]):
                    gathering_temp = gathered_numpy[append_index * multiplier_list[idx]].copy()
                    for mul in range(1, multiplier_list[idx]):
                        # gathering_temp = gathering_temp + gathered_numpy[append_index * multiplier_list[idx] + mul].copy()
                        gathering_temp = np.maximum(gathering_temp ,gathered_numpy[append_index * multiplier_list[idx] + mul].copy())
                    # gathering_temp = gathering_temp/multiplier_list[idx]
                    new_numpy = np.vstack([new_numpy,gathering_temp])
                # print(len(new_numpy))

                # Update indexes that array becomde shorten

                # print(len(new_numpy))
                new_numpy = np.vstack([new_numpy, after_numpy])
                checking_new_numpy_length.append(len(new_numpy))
                # print('After Attatched numpy: ', len(new_numpy))
                # print(len(new_numpy))
                new_extended_numpy = new_numpy.copy()
            # print(len(new_extended_numpy))
            perturbed_input[song_num][channel] = new_extended_numpy.copy()

            # perturbed_input[song_num][channel].append(perturbed_input[song_num][channel], new_numpy[len(perturbed_input[song_num][channel]):][:],axis=0)

    # print(perturbed_input)
    return perturbed_input


def test(model, epsilon):

    adv_rounded = []
    adv_fname = []
    correct = 0
    orig_wrong = 0
    global current_perturbed_tempo

    for i,val in enumerate(val_loader):
        print(i)
        data = val[0]
        target = val[1]
        name = val[2]

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

        #generate manual attacks
        # perturbed_data = vel_attack(data, epsilon, data_grad)
        perturbed_data2 = tempo_attack(data.detach().cpu().numpy(), epsilon, data_grad)
        perturbed_data2_tensor = torch.from_numpy(perturbed_data2).to(device)
        new_out = model(perturbed_data2_tensor)
        current_perturbed_tempo[i] = perturbed_data2.copy()
        new_pred = torch.max(new_out, 1)[1].view(target.size()).data

        #get orig data
        orig_data = data.squeeze().detach().cpu().numpy()
        # np_vel = perturbed_data.squeeze().detach().cpu().numpy() #(128 400 128)
        np_tempo = perturbed_data2.copy()
        #check for success
        if new_pred.item() == target.item():
            correct += 1
        else:
            pass
        # np.save("/data/tempo_temp2/" + str(name)[2:-3] +'[2]' , np_tempo)  # save as .npy
        # np.save("/data/tempo_temp/" + str(name)[2:-3] , orig_data)  # save as .npy
        #np.save("/data/attack_test/pitch_" + each_file, np_pitch)  # save as .npy
        #     np.save("/data/attack_test/time_" + each_file, np_time)  # save as .npy
        #     np.save("/data/attack_test/vel_" + each_file + "_[" + str(epsilon) +"]", np_vel)  # save as .npy
        #     np.save("/data/attack_test/vel_" + each_file + "_[0]" , orig_data)  # save as .npy

    # print("files saved!")
    return adv_rounded, adv_fname, orig_wrong, correct


##########################################################
#run attack

accuracies = []
# epsilons = [0,2,4,6,8,10,12,14,16,18,20, 22, 24, 26, 28, 30,32,34,36,38,40]
epsilons = [2]
#

save_max_perturbed_tempo = np.ones(240*129*400*128).reshape((240,1,129,400,128))
current_perturbed_tempo = np.ones(240*129*400*128).reshape((240,1,129,400,128))
max_attack_acc = 100
for iter in range(0,1):
    #
    print("Current iteration: ",iter)
    rounded, fname, orig_wrong, correct = test(model, iter)
    denom = len(val_loader)
    orig_acc = (denom - orig_wrong) / denom
    final_acc = correct / float(denom)
    if final_acc < max_attack_acc:
        max_attack_acc = final_acc
        save_max_perturbed_tempo = current_perturbed_tempo.copy()
        print("Intermediate max_attack_accuracy: ", max_attack_acc)
    print("Epsilon: {}".format(iter))
    print("Before: {} / {} = {}".format(denom - orig_wrong, denom, orig_acc))
    print("After: {} / {} = {}".format(correct, denom, final_acc))

print('Attack Worked Well Accuracy: ', max_attack_acc)

for i, val in enumerate(val_loader):
    print(i)
    data = val[0]
    target = val[1]
    name = val[2]
    np.save("/data/tempo_temp3/" + str(name)[2:-3] +'[2]' ,save_max_perturbed_tempo[i] )
#
#     #for plt
#     accuracies.append(final_acc)
# examples.append(ex)


#Draw Results
# plt.figure(figsize=(5,5))
# plt.plot(epsilons, accuracies, "*-")
# plt.yticks(np.arange(0, 1.1, step=0.1))
# plt.xticks(np.arange(0, 40, step=2))
# plt.title("Accuracy vs Velocity Range")
# plt.xlabel("Velocity Range")
# plt.ylabel("Accuracy")
# plt.show()
