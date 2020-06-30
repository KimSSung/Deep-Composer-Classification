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
import torchsummary
from torch.optim import lr_scheduler




torch.manual_seed(123)
import torch.nn as nn


#for GPU use
device = torch.device('cuda:0')


##############################################################
# genres = ['Classical', 'Jazz','Rock','Country','Pop', 'HipHopRap', 'NewAge','Blues'] #total
genres = ['Classical','Rock','Country'] #best
num_genres = 3
min_shape= 820
batch_size = 40

input_total=[]
output_total=[]
for genre in genres:

    load_saved = np.load("/data/midi820_cnn/" + genre + "_input.npy", allow_pickle=True)
    if(load_saved.shape[0] < min_shape):
        min_shape = load_saved.shape[0] # num of data in genre
    output_temp = [genres.index(genre)]*load_saved.shape[0]
    output_total.append(output_temp)
    input_total.append(load_saved)

input_list = []
output_list = []
for i in input_total:
    input_list.extend(i[:min_shape,:,:])
for o in output_total:
    output_list.extend(o[:min_shape])
X_np = np.array(input_list)
Y_np = np.array(output_list)

##shuffle
data = list(zip(X_np, Y_np)) #zip data structure
random.shuffle(data)

##partition
X,Y = zip(*data)
train_len = int(len(X) * 8 / 10)  # train : valid = 8 : 2
X,Y = np.asarray(X), np.asarray(Y)
train_X, train_Y = X[:train_len], Y[:train_len]
dev_X, dev_Y = X[train_len:], Y[train_len:]

##for batch calc
t_keep = len(train_X) - len(train_X) % batch_size
v_keep = len(dev_X) - len(dev_X) % batch_size
trn_X, trn_Y, val_X, val_Y = train_X[:t_keep], train_Y[:t_keep], dev_X[:v_keep], dev_Y[:v_keep]


trn_X = torch.from_numpy(trn_X).type(torch.Tensor)
val_X = torch.from_numpy(val_X).type(torch.Tensor)
trn_Y = torch.from_numpy(trn_Y).type(torch.LongTensor)
val_Y = torch.from_numpy(val_Y).type(torch.LongTensor)

# tensorDataset
t = TensorDataset(trn_X, trn_Y)
v = TensorDataset(val_X, val_Y)

# create batch
train_loader = DataLoader(t, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(v, batch_size=batch_size, shuffle=True)

print("Training X shape: " + str(trn_X.shape))
print("Training Y shape: " + str(trn_Y.shape))
print("Validation X shape: " + str(val_X.shape))
print("Validation Y shape: " + str(val_Y.shape))

##############################################################

k = 3 # kernel size
p = 3 # pool size
class CustomCNN(nn.Module):

    def __init__(self, input_size, num_genres):
        super(CustomCNN, self).__init__()
        self.input_size = input_size
        self.output_size = num_genres

        # padding = (n - ((n-k)+1)) / 2 = (k-1)/2 = 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=k, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=p, stride=p),
            nn.Dropout(p=0.25)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=k, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=p, stride=p),
            nn.Dropout(p=0.25)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=k, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=p, stride=p),
            nn.Dropout(p=0.25)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=k, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=p, stride=p),
            nn.Dropout(p=0.25)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=k, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=p, stride=p),
            nn.Dropout(p=0.25)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            # nn.Linear(4096, 512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            # nn.Linear(512, num_genres),
            nn.Linear(128, num_genres),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out = self.fc(out5)

        return out

###################################

# Define model
model = CustomCNN(num_genres=num_genres, input_size=2)
# torchsummary.summary(model, (2, 400, 128), 35)

#hyper params
num_epochs = 200
num_batches = len(train_loader)
num_dev_batches = len(val_loader)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-6)
# optimizer = optim.SGD(model.parameters(),lr=learning_rate)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.5,patience=10,verbose=True) #0.5 best for midi370
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


#use GPU
model = model.to(device)
criterion = criterion.to(device)


#for plot
trn_loss_list = []
val_loss_list = []
trn_acc_list = []
val_acc_list= []

for epoch in range(num_epochs):

    trn_running_loss, trn_acc = 0.0, 0.0
    # trn_correct = 0
    # trn_total = 0
    for i, trainset in enumerate(train_loader):
        #train_mode
        model.train()
        #unpack
        train_in, train_out = trainset
        #use GPU
        train_in = train_in.to(device)
        train_out = train_out.to(device)
        #grad init
        optimizer.zero_grad()

        #forward pass
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

      with torch.no_grad():  # important!!! for validation
        # validate mode
        model.eval()

		#average the acc of each batch
        val_loss, val_acc = 0.0, 0.0
        # val_correct = 0
        # val_total = 0
        for j, valset in enumerate(val_loader):
            val_in, val_out = valset
            # to GPU
            val_in = val_in.to(device)
            val_out = val_out.to(device)

            # forward
            val_pred = model(val_in)
            v_loss = criterion(val_pred, val_out)
            val_loss += v_loss

            # scheduler.step(v_loss)  # for reduceonplateau
            scheduler.step()       #for cos
            lr = optimizer.param_groups[0]['lr']

		# accuracy
            _, val_label_pred = torch.max(val_pred.data, 1)
            val_total = val_out.size(0)
            val_correct = (val_label_pred == val_out).sum().item()
            val_acc += val_correct / val_total * 100
            print("correct: {}, total: {}, acc: {}".format(val_correct, val_total, val_correct/val_total*100))

        print("epoch: {}/{} | trn loss: {:.4f} | trn acc: {:.2f}%| val loss: {:.4f} | val acc: {:.2f}% | lr: {:.6f}"
			  .format(epoch + 1, num_epochs,
                    trn_running_loss / num_batches,
					trn_acc / num_batches,
				    val_loss / num_dev_batches,
                    val_acc / num_dev_batches,
			  		lr))


        trn_loss_list.append(trn_running_loss / num_batches)
        val_loss_list.append(val_loss / num_dev_batches)
        trn_acc_list.append(trn_acc / num_batches)
        val_acc_list.append(val_acc / num_dev_batches)

        # reinit to 0
        # trn_running_loss = 0.0
        # trn_total = 0
        # trn_correct = 0




# Summarize history for accuracy
xi = [i*val_term for i in range(int(num_epochs/val_term))]
plt.plot(xi, trn_acc_list)
plt.plot(xi, val_acc_list)
plt.xticks()
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(xi, trn_loss_list)
plt.plot(xi, val_loss_list)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()