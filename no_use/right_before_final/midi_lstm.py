
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import lr_scheduler

#local python class that retrieves input data

################################### INPUT DATA #####################################

# genres = ['Classical', 'Jazz','Rock','Country','Pop', 'HipHopRap', 'NewAge','Blues'] #total
genres = ['Classical','Rock','Country'] #best
# genres = ['Classical', 'Rock','Country','HipHopRap','NewAge','Jazz']
min_shape= 820


input_total=[]
output_total=[]
for genre in genres:

    load_saved = np.load("/data/midi820_instr/" + genre + "_input.npy", allow_pickle=True)
    # print(load_saved.shape)
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

##one-hot for Y
y_one_hot = np.zeros((Y_np.shape[0], len(genres)))
for i, index in enumerate(Y_np):
    y_one_hot[i, index] = 1
Y_np = y_one_hot

##shuffle
data = list(zip(X_np, Y_np)) #zip data structure
random.shuffle(data)


##partition
X,Y = zip(*data)
# X,Y= torch.tensor(X, dtype=torch.tensor()), torch.tensor(Y, dtype=torch.LongTensor)
train_len = int(len(X) * 8 / 10)  # train : valid = 8 : 2
# val_len = int(len(X)*15/100)

X,Y = np.asarray(X), np.asarray(Y)
train_X, train_Y = X[:train_len], Y[:train_len]
dev_X, dev_Y = X[train_len:], Y[train_len:]
# dev_X, dev_Y = X[train_len:train_len+val_len], Y[train_len:train_len+val_len]
# test_X, test_Y = X[train_len:train_len+val_len:], Y[train_len:train_len+val_len:]


train_X = torch.from_numpy(train_X).type(torch.Tensor)
dev_X = torch.from_numpy(dev_X).type(torch.Tensor)
# test_X = torch.from_numpy(test_X).type(torch.Tensor)

# Targets is a long tensor of size (N,) which tells the true class of the sample.
train_Y = torch.from_numpy(train_Y).type(torch.LongTensor)
dev_Y = torch.from_numpy(dev_Y).type(torch.LongTensor)
# test_Y = torch.from_numpy(test_Y).type(torch.LongTensor)


# tensorDataset
t = TensorDataset(train_X, train_Y)
v = TensorDataset(dev_X, dev_Y)

# create batch
batch_size = 35
train_loader = DataLoader(t, batch_size=batch_size, shuffle=True) # 45(data for each batch) x 128(batches) = 640 data
val_loader = DataLoader(v, batch_size=batch_size, shuffle=True) # 45(data for each batch) x 32(batches) = 160 data


# Convert {training, test} torch.Tensors
print("Training X shape: " + str(train_X.shape))
print("Training Y shape: " + str(train_Y.shape))
print("Validation X shape: " + str(dev_X.shape))
print("Validation Y shape: " + str(dev_Y.shape))
# print("Test X shape: " + str(test_X.shape))
# print("Test Y shape: " + str(test_Y.shape))



########################################### MODEL ##############################################


# class definition
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # setup LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # setup output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        return (
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
        )

    def forward(self, input):
        # lstm step => then ONLY take the sequence's final timestep to pass into the linear/dense layer
        # Note: lstm_out contains outputs for every step of the sequence we are looping over (for BPTT)
        # but we just need the output of the last step of the sequence, aka lstm_out[-1]
        lstm_out, hidden = self.lstm(input)
        logits = self.linear(lstm_out[-1])
        genre_scores = F.log_softmax(logits, dim=1)
        return genre_scores

    def get_accuracy(self, logits, target, length):
        """ compute accuracy for training round """
        corrects = (
            torch.max(logits, 1)[1].view(target.size()).data == target.data
        ).sum()
        accuracy = 100.0 * corrects / length
        # print("correct:{}/{} = acc:{}".format(corrects,length, accuracy))
        # if(accuracy > 95):
        #     print("correct:{}/{} = acc:{}".format(corrects, length, accuracy))
        #     print(torch.max(logits, 1)[1].view(target.size()).data)
        #     print(target.data)
        return accuracy.item()

    # def get_corrects(self, logits, target):
    #     corrects = (
    #             torch.max(logits, 1)[1].view(target.size()).data == target.data
    #     ).sum()
    #     return corrects.item()

    def get_genre_accuracy(self,logits,target):
        return


########################################## TRAIN ############################################


# batch_size = 35  # num of training examples per minibatch
num_epochs = 400

# Define model
print("Build LSTM RNN model ...")
model = LSTM( #hidden_dim = 80 best #use input dim = 128 for instr_input
    input_dim=128, hidden_dim=90, batch_size=batch_size, output_dim=len(genres), num_layers=2
)
#check model parameters
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())
loss_function = nn.NLLLoss()  # expects outputs from LogSoftmax
optimizer = optim.Adam(model.parameters(), lr=0.001) #0.0005 best for midi370 #0.001 best for midi820
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.5,patience=10,verbose=True) #0.5 best for midi370
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


##for GPU use
train_on_gpu = torch.cuda.is_available()
device = None
if train_on_gpu:
    device = torch.device('cuda:0')
    print("\nTraining on GPU:{}".format(device))
else:
    print("\nNo GPU, training on CPU")

#use GPU
model = model.to(device)
loss_function = loss_function.to(device)

# all training data (1910) / batch_size(35) => num_batches (54)
# all val data (478) / batch_size(35) => num_batches (14) 13.67..
num_batches = int(train_X.shape[0] / batch_size)
num_dev_batches = int(dev_X.shape[0] / batch_size)

val_loss_list, val_accuracy_list, epoch_list = [], [], []

stop_epochs = 5
best_val_loss = 10000.0
model_count = 0
######## TRAINING START ########
print("Training ...")
for epoch in range(num_epochs):

    train_running_loss, train_acc = 0.0, 0.0

    # Init hidden state - if you don't want a stateful LSTM (between epochs)
    model.hidden = model.init_hidden()
    # for i in range(num_batches):
    for i, trainset in enumerate(train_loader):
        # unpack
        X_local_minibatch, y_local_minibatch = trainset

        # zero out gradient, so they don't accumulate btw epochs
        model.zero_grad()
        # zero out gradient from previous epoch
        optimizer.zero_grad()

        # train_X shape: (total # of training examples, sequence_length, input_dim)
        # train_Y shape: (total # of training examples, # output classes)
        #
        # Slice out local minibatches & labels => Note that we *permute* the local minibatch to
        # match the PyTorch expected input tensor format of (sequence_length, batch size, input_dim)
        # X_local_minibatch, y_local_minibatch = (
        #     train_X[i * batch_size : (i + 1) * batch_size,],
        #     train_Y[i * batch_size : (i + 1) * batch_size,],
        # )

        # Reshape input & targets to "match" what the loss_function wants
        X_local_minibatch = X_local_minibatch.permute(1, 0, 2)

        # NLLLoss does not expect a one-hot encoded vector as the target, but class indices
        y_local_minibatch = torch.max(y_local_minibatch, 1)[1]

        # to GPU
        X_local_minibatch = X_local_minibatch.to(device)
        y_local_minibatch = y_local_minibatch.to(device)

        y_pred = model(X_local_minibatch)                # fwd the batch (forward pass)
        loss = loss_function(y_pred, y_local_minibatch)  # compute loss
        loss.backward()                                  # reeeeewind (backward pass)
        optimizer.step()                                 # parameter update

        train_running_loss += loss.detach().item()       # unpacks the tensor into a scalar value
        train_acc += model.get_accuracy(y_pred, y_local_minibatch, len(y_local_minibatch))

    print(
        "Epoch:  %d | NLLoss: %.4f | Train Accuracy: %.2f"
        % (epoch, train_running_loss / num_batches, train_acc / num_batches)
    )

####### VALIDATION #######
    if epoch % 10 == 0:
        print("Validation ...")  # should this be done every N epochs
        val_running_loss, val_acc = 0.0, 0.0

        # Compute validation loss, accuracy. Use torch.no_grad() & model.eval()
        with torch.no_grad():
            model.eval()

            model.hidden = model.init_hidden()
            for j, valset in enumerate(val_loader):
                # X_local_validation_minibatch, y_local_validation_minibatch = (
                #     dev_X[i * batch_size : (i + 1) * batch_size,],
                #     dev_Y[i * batch_size : (i + 1) * batch_size,],
                # )
                #unpack
                X_local_validation_minibatch, y_local_validation_minibatch = valset

                X_local_minibatch = X_local_validation_minibatch.permute(1, 0, 2)
                y_local_minibatch = torch.max(y_local_validation_minibatch, 1)[1]

                # to GPU
                X_local_minibatch = X_local_minibatch.to(device)
                y_local_minibatch = y_local_minibatch.to(device)

                #forward
                y_pred = model(X_local_minibatch)
                val_loss = loss_function(y_pred, y_local_minibatch) #return tensor

                val_running_loss += (
                    val_loss.detach().item()
                )  # unpacks the tensor into a scalar value
                val_acc += model.get_accuracy(y_pred, y_local_minibatch, len(y_local_minibatch))

                #Note that step should be called after validate()
                scheduler.step(val_loss) #for reduceonplateau
                # scheduler.step()       #for cos
                lr = optimizer.param_groups[0]['lr']

            # save the model if validation accuracy improved
            if val_running_loss < best_val_loss:
                state = model.state_dict()
                model_count += 1
                filename = ("/data/lstm_models/instr/{}_model4_valloss_".format(model_count)) + str(val_running_loss)
                print("=> Saving a new best")
                torch.save(state, filename)  # save checkpoint
                best_val_loss = val_running_loss

            model.train()  # reset to train mode after iteration through validation data
            print(
                "Epoch:  %d | NLLoss: %.4f | Train Accuracy: %.2f | Val Loss %.4f  | Val Accuracy: %.2f | lr : %.6f"
                % (
                    epoch,
                    train_running_loss / num_batches,
                    train_acc / num_batches,
                    val_running_loss / num_dev_batches,
                    val_acc / num_dev_batches,
                    lr
                )
            )

        epoch_list.append(epoch)
        val_accuracy_list.append(val_acc / num_dev_batches)
        val_loss_list.append(val_running_loss / num_dev_batches)


########################## PLOT #########################


# visualization loss
plt.plot(epoch_list, val_loss_list)
plt.xlabel("# of epochs")
plt.ylabel("Loss")
plt.title("LSTM: Loss vs # epochs")
plt.show()

# visualization accuracy
plt.plot(epoch_list, val_accuracy_list, color="red")
plt.xlabel("# of epochs")
plt.ylabel("Accuracy")
plt.title("LSTM: Accuracy vs # epochs")
# plt.savefig('graph.png')
plt.show()

# print("Testing ...")
#no test code,,,??

