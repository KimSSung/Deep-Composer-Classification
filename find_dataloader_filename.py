import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
from tqdm import tqdm

# from ResNet import resnet18

# # load:
# the_model = resnet18(128, 3)
# the_model.eval()
# the_model.load_state_dict(torch.load('/data/midi820_128ch/model/ResNet18_valloss_0.5151_acc_85.0.pt'))

TRAIN_LOADER_SAVE_PATH = "/data/midi820_128ch/dataset/train/"
VALID_LOADER_SAVE_PATH = "/data/midi820_128ch/dataset/valid/"

# classical_input = np.load("/data/midi820_128ch/Classical_input.npy", allow_pickle=True)
# classical_file = np.load("/data/midi820_128ch/Classical_filename.npy", allow_pickle=True)
# rock_input = np.load("/data/midi820_128ch/Rock_input.npy", allow_pickle=True)
# rock_file = np.load("/data/midi820_128ch/Rock_filename.npy", allow_pickle=True)
country_input = np.load("/data/midi820_128ch/Country_input.npy", allow_pickle=True)
country_file = np.load("/data/midi820_128ch/Country_filename.npy", allow_pickle=True)

train_loader = torch.load(TRAIN_LOADER_SAVE_PATH + "train_loader.pt")
print("train_loader loaded!")
val_loader = torch.load(VALID_LOADER_SAVE_PATH + "valid_loader.pt")
print("valid_loader loaded!")

# clf = open('/data/midi820_128ch/find_filename/Classical/find_classical_filename.txt', 'w')
# rof = open('/data/midi820_128ch/find_filename/Rock/find_rock_filename.txt', 'w')
cof = open("/data/midi820_128ch/find_filename/Country/find_country_filename.txt", "w")

# clf.write('###################  Train_loader ###################\n')
# rof.write('###################  Train_loader ###################\n')
cof.write("###################  Train_loader ###################\n")

# tot_sum = 0.
num_of_zero = 0
new_classical_trainset = []
new_rock_trainset = []
new_country_trainset = []

new_classical_validset = []
new_rock_validset = []
new_country_validset = []
for trainset in train_loader:  # trainset 안에 20개씩
    train_in, _ = trainset
    for j in train_in:  # 개별 (128, 400, 128) 파일
        jsum = torch.sum(j)
        if float(jsum) == 0.0:
            num_of_zero += 1
            continue

        # # classical matching
        # print('matching train classical......')
        # for idx, arr in tqdm(enumerate(classical_input)):
        # 	if float(np.sum(arr)) == float(jsum):
        # 		# print(np.sum(arr))
        # 		# print(jsum)
        # 		clf.write(str(idx) + '.' + classical_file[idx] + '\n')
        # 		clf.write('\n')
        # 		new_classical_trainset.append(arr)

        # # rock matching
        # print('matching train rock......')
        # for idx, arr in tqdm(enumerate(rock_input)):
        # 	if float(np.sum(arr)) == float(jsum):
        # 		# print(np.sum(arr))
        # 		# print(jsum)
        # 		rof.write(str(idx) + '.' + rock_file[idx] + '\n')
        # 		rof.write('\n')
        # 		new_rock_trainset.append(arr)

        # country matching
        print("matching train country......")
        for idx, arr in tqdm(enumerate(country_input)):
            if float(np.sum(arr)) == float(jsum):
                # print(np.sum(arr))
                # print(jsum)
                cof.write(str(idx) + "." + country_file[idx] + "\n")
                cof.write("\n")
                new_country_trainset.append(arr)

        # print(torch.sum(j))
        # tot_sum += torch.sum(j)


# clf.write('###################  Valid_loader ###################\n')
# rof.write('###################  Valid_loader ###################\n')
cof.write("###################  Valid_loader ###################\n")
for validset in val_loader:  # trainset 안에 20개씩
    valid_in, _ = validset
    for j in valid_in:  # 개별 (128, 400, 128) 파일
        jsum = torch.sum(j)
        if float(jsum) == 0.0:
            num_of_zero += 1
            continue

        # # classical matching
        # print('matching valid classical......')
        # for idx, arr in tqdm(enumerate(classical_input)):
        # 	if float(np.sum(arr)) == float(jsum):
        # 		# print(np.sum(arr))
        # 		# print(jsum)
        # 		clf.write(str(idx) + '.' + classical_file[idx] + '\n')
        # 		clf.write('\n')
        # 		new_classical_validset.append(arr)

        # # rock matching
        # print('matching valid rock......')
        # for idx, arr in tqdm(enumerate(rock_input)):
        # 	if float(np.sum(arr)) == float(jsum):
        # 		# print(np.sum(arr))
        # 		# print(jsum)
        # 		rof.write(str(idx) + '.' + rock_file[idx] + '\n')
        # 		rof.write('\n')
        # 		new_rock_validset.append(arr)

        # country matching
        print("matching valid country......")
        for idx, arr in tqdm(enumerate(country_input)):
            if float(np.sum(arr)) == float(jsum):
                # print(np.sum(arr))
                # print(jsum)
                cof.write(str(idx) + "." + country_file[idx] + "\n")
                cof.write("\n")
                new_country_validset.append(arr)

        # print(torch.sum(j))
        # tot_sum += torch.sum(j)


# # print(tot_sum)
print("total 0:", num_of_zero)

# np.save('/data/midi820_128ch/find_filename/Classical/new_classical_trainset.npy', new_classical_trainset)
# print('new classical trainset saved!')

# np.save('/data/midi820_128ch/find_filename/Classical/new_classical_validset.npy', new_classical_validset)
# print('new classical valid saved!')

# np.save('/data/midi820_128ch/find_filename/Rock/new_rock_trainset.npy', new_rock_trainset)
# print('new rock trainset saved!')

# np.save('/data/midi820_128ch/find_filename/Rock/new_rock_validset.npy', new_rock_validset)
# print('new rock valid saved!')

np.save(
    "/data/midi820_128ch/find_filename/Country/new_country_trainset.npy",
    new_country_trainset,
)
print("new country trainset saved!")

np.save(
    "/data/midi820_128ch/find_filename/Country/new_country_validset.npy",
    new_country_validset,
)
print("new country valid saved!")


# clf.close()
# rof.close()
cof.close()
