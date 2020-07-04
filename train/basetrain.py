import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import os
import torch
import matplotlib.pyplot as plt
from torch import utils
import numpy as np
# from tqdm import tqdm
from torch.optim import lr_scheduler

from models.resnet import resnet18, resnet101, resnet152, resnet50
# from CustomCNN import CustomCNN

# dataloader
from data_loader import MIDIDataset

torch.manual_seed(123)


#for GPU use
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_PATH = '/data/drum/model/'
TRAIN_LOADER_SAVE_PATH = '/data/drum/dataset/train/' # /data/midi820_drum/dataset/train/'
VALID_LOADER_SAVE_PATH = '/data/drum/dataset/test/'
VALID_FILENAME_PATH = '/data/drum/dataset/val_filename/'

##############################################################
# genres = ['Classical', 'Jazz','Rock','Country','Pop', 'HipHopRap', 'NewAge','Blues'] #total
genres = ['Classical', 'Rock', 'Country', 'GameMusic'] #best
num_genres = len(genres)
min_shape= 820
batch_size = 20

each_num = 300

# Loader for origin training
t = MIDIDataset('/data/midi820_400/train/', 0, each_num * 0.8, genres, 'folder')
v = MIDIDataset('/data/midi820_400/valid/', 0, each_num * 0.2, genres, 'folder')

# create batch
train_loader = DataLoader(t, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(v, batch_size=batch_size, shuffle=True)

#############################################################################
################### Loader for adversarial training #########################
#############################################################################

# t_list = []
# t_list.append(MIDIDataset('/data/attacks/vel_deepfool/train/', -1, -1, genres, 'flat')) # not use start, end index for 'flat'
# t_list.append(MIDIDataset('/data/midi820_400/train/', 0, each_num * 0.8, genres, 'folder'))
# t = ConcatDataset(t_list)


# v2 = MIDIDataset('/data/midi820_400/valid/', 0, each_num * 0.2, genres, 'folder')
# v_list = []
# v_list.append(MIDIDataset('/data/attacks/vel_deepfool/valid/', -1, -1, genres, 'flat'))
# v_list.append(v2)
# v1 = ConcatDataset(v_list) # test + attack test 


# # train + attack train
# train_loader = DataLoader(t, batch_size=batch_size, shuffle=True)
# # test + attack test = TandAT
# val_loader_1 = DataLoader(v1, batch_size=batch_size, shuffle=True)
# # Only Test = T
# val_loader_2 = DataLoader(v2, batch_size=batch_size, shuffle=True)



#############################################################################
###################### Loader for base training #############################
#############################################################################

# save train_loader & valid_loader
torch.save(train_loader, TRAIN_LOADER_SAVE_PATH + 'train_loader.pt')
print("train_loader saved!")
torch.save(val_loader, VALID_LOADER_SAVE_PATH + 'valid_loader.pt')
print("valid_loader saved!")



train_loader = torch.load(TRAIN_LOADER_SAVE_PATH + 'train_loader.pt')
print("train_loader loaded!")
val_loader_1 = torch.load(VALID_LOADER_SAVE_PATH + 'valid_loader.pt')
print("valid_loader loaded!")

#############################################################################


# Define model
model = resnet50(129, num_genres)
# model = CustomCNN(129, num_genres)
# print(model)



#hyper params
num_epochs = 100
num_batches = len(train_loader)
# num_dev_batches = len(val_loader)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-6) # 0.00005
# optimizer = optim.SGD(model.parameters(),lr=0.0001)
# optimizer = optim.ASGD(model.parameters(), lr=0.00005, weight_decay=1e-6)
# optimizer = optim.SparseAdam(model.parameters(), lr=0.00005, betas=(0.9, 0.999), eps=1e-08)
print("optimizer:",optimizer)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.5,patience=10,verbose=True) #0.5 best for midi370
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


# use GPU
model = model.to(device)
criterion = criterion.to(device)


#for plot
trn_loss_list = []
val_loss_list = []
trn_acc_list = []
val_acc_list= []


def Test(val_loader, model, save_filename=False):

   val_file_names = [] # file name list
   with torch.no_grad(): # important!!! for validation
      # validate mode
      model.eval()

      #average the acc of each batch
      val_loss, val_acc = 0.0, 0.0
      # val_correct = 0
      # val_total = 0

      for j, valset in enumerate(val_loader):
         val_in, val_out, file_name = valset

         # save valid file name only at first validation
         
         if save_filename: # only when epoch = val_term(10)
            for fname in file_name:
               val_file_names.append(fname)

         # to GPU
         val_in = val_in.to(device)
         val_out = val_out.to(device)

         # forward
         val_pred = model(val_in)
         v_loss = criterion(val_pred, val_out)
         val_loss += v_loss

         # scheduler.step(v_loss)  # for reduceonplateau
         scheduler.step()       #for cos

         # accuracy
         _, val_label_pred = torch.max(val_pred.data, 1)
         val_total = val_out.size(0)
         val_correct = (val_label_pred == val_out).sum().item()
         val_acc += val_correct / val_total * 100
         print("correct: {}, total: {}, acc: {}".format(val_correct, val_total, val_correct/val_total*100))

      avg_valloss = val_loss / len(val_loader)
      avg_valacc = val_acc / len(val_loader)

   return avg_valloss, avg_valacc, val_file_names



min_valloss = 10000.0
for epoch in range(num_epochs):

   trn_running_loss, trn_acc = 0.0, 0.0
   # trn_correct = 0
   # trn_total = 0
   for i, trainset in enumerate(train_loader):
      #train_mode
      model.train()
      #unpack
      train_in, train_out, file_name = trainset
      # print(train_in.shape)
      # print(train_out.shape)
      #use GPU
      train_in = train_in.to(device)
      train_out = train_out.to(device)
      #grad init
      optimizer.zero_grad()

      #forward pass
      # print(train_in.shape)
      train_pred = model(train_in)
      #calculate acc
      _, label_pred = torch.max(train_pred.data, 1)
      trn_total = train_out.size(0)
      trn_correct = (label_pred == train_out).sum().item()
      trn_acc += (trn_correct / trn_total * 100)
      #calculate loss
      t_loss = criterion(train_pred, train_out)
      #back prop
      t_loss.backward()
      #weight update
      optimizer.step()

      trn_running_loss += t_loss.item()


   #print learning process
   print(
      "Epoch:  %d | Train Loss: %.4f | Train Accuracy: %.2f"
      % (epoch, trn_running_loss / num_batches,
         # (trn_correct/trn_total *100))
         trn_acc / num_batches)
   )


   ####### VALIDATION #######
   val_term = 10
   if epoch % val_term == 0:

      if epoch == 0:
         avg_valloss, avg_valacc, val_file_names = Test(val_loader, model, save_filename=True)
         # save val file names
         torch.save(val_file_names, VALID_FILENAME_PATH + 'val_file_names.pt')
         filenames = torch.load(VALID_FILENAME_PATH + 'val_file_names.pt')
         print('val file names len:',len(filenames))

      else:
         avg_valloss, avg_valacc, _ = Test(val_loader, model, save_filename=False)


      lr = optimizer.param_groups[0]['lr']
      print('''epoch: {}/{} | trn loss: {:.4f} | trn acc: {:.2f}%| lr: {:.6f} |
val loss: {:.4f} | val acc: {:.2f}%'''
            .format(epoch + 1, num_epochs,
               trn_running_loss / num_batches, trn_acc / num_batches, lr,
               avg_valloss, avg_valacc
               ))

      # save model
      if avg_valloss < min_valloss:
         min_valloss = avg_valloss
         torch.save({'epoch':epoch,
                  'model.state_dict':model.state_dict(),
                  'loss':avg_valloss,
                  'acc':avg_valacc}, MODEL_SAVE_PATH + 'Res50_valloss_' + str(float(avg_valloss)) + '_acc_' + str(float(avg_valacc)) + '.pt')
         print('model saved!')





'''
# test for loading model
#hyper params
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-6)
# optimizer = optim.SGD(model.parameters(),lr=0.0001)
# print("optimizer:",optimizer)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.5,patience=10,verbose=True) #0.5 best for midi370
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# load:
model = resnet50(129, num_genres)
model.eval()
checkpoint = torch.load('/data/drum/bestmodel/Conv_valloss_0.8801_acc_81.25.pt')
model.load_state_dict(checkpoint['model.state_dict'])

val_term = 10

with torch.no_grad():  # important!!! for validation
   # validate mode
   model.eval()

   print('testing pretrained model.......')
   #average the acc of each batch
   val_loss, val_acc = 0.0, 0.0
   # val_correct = 0
   # val_total = 0
   for j, valset in enumerate(val_loader):
      val_in, val_out, val_filename = valset
      # to GPU
      # val_in = val_in.to(device)
      # val_out = val_out.to(device)

      # forward
      val_pred = model(val_in)
      v_loss = criterion(val_pred, val_out)
      val_loss += v_loss

      # # scheduler.step(v_loss)  # for reduceonplateau
      # scheduler.step()       #for cos
      # lr = optimizer.param_groups[0]['lr']

      # accuracy
      _, val_label_pred = torch.max(val_pred.data, 1)
      val_total = val_out.size(0)
      val_correct = (val_label_pred == val_out).sum().item()
      val_acc += val_correct / val_total * 100
      print("correct: {}, total: {}, acc: {}".format(val_correct, val_total, val_correct/val_total*100))


print("val acc: {:.2f}%".format(val_acc/len(val_loader)))
'''