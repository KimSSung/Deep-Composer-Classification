
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random
#local python class that retrieves input data

################################### INPUT DATA #####################################


genres = ['Classical', 'Jazz', 'Pop', 'Country', 'Rock']
min_shape= 370


input_total=[]
output_total=[]
for genre in genres:

    load_saved = np.load("/data/midi370_input/" + genre + "_input.npy", allow_pickle=True)
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
y_one_hot = np.zeros((Y_np.shape[0], len(genres))) #1805 x 5
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
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=5, num_layers=2):
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

    def get_accuracy(self, logits, target):
        """ compute accuracy for training round """
        corrects = (
            torch.max(logits, 1)[1].view(target.size()).data == target.data
        ).sum()
        accuracy = 100.0 * corrects / self.batch_size
        return accuracy.item()

    def get_genre_accuracy(self,logits,target):
        return


########################################## TRAIN ############################################


batch_size = 35  # num of training examples per minibatch
num_epochs = 100

# Define model
print("Build LSTM RNN model ...")
model = LSTM(
    input_dim=129, hidden_dim=128, batch_size=batch_size, output_dim=8, num_layers=2
)
loss_function = nn.NLLLoss()  # expects ouputs from LogSoftmax
optimizer = optim.Adam(model.parameters(), lr=0.001)

##for GPU use
train_on_gpu = torch.cuda.is_available()
device = None
if train_on_gpu:
    device = torch.device('cuda:0')
    print("\nTraining on GPU")
else:
    print("\nNo GPU, training on CPU")

#use GPU
model = model.to(device)
loss_function = loss_function.to(device)

# all training data (epoch) / batch_size == num_batches (12)
num_batches = int(train_X.shape[0] / batch_size)
num_dev_batches = int(dev_X.shape[0] / batch_size)

val_loss_list, val_accuracy_list, epoch_list = [], [], []


######## TRAINING START ########
print("Training ...")
for epoch in range(num_epochs):

    train_running_loss, train_acc = 0.0, 0.0

    # Init hidden state - if you don't want a stateful LSTM (between epochs)
    model.hidden = model.init_hidden()
    for i in range(num_batches):

        # zero out gradient, so they don't accumulate btw epochs
        model.zero_grad()

        # train_X shape: (total # of training examples, sequence_length, input_dim)
        # train_Y shape: (total # of training examples, # output classes)
        #
        # Slice out local minibatches & labels => Note that we *permute* the local minibatch to
        # match the PyTorch expected input tensor format of (sequence_length, batch size, input_dim)
        X_local_minibatch, y_local_minibatch = (
            train_X[i * batch_size : (i + 1) * batch_size,],
            train_Y[i * batch_size : (i + 1) * batch_size,],
        )

        # Reshape input & targets to "match" what the loss_function wants
        X_local_minibatch = X_local_minibatch.permute(1, 0, 2)

        # NLLLoss does not expect a one-hot encoded vector as the target, but class indices
        y_local_minibatch = torch.max(y_local_minibatch, 1)[1]

        # to GPU
        X_local_minibatch = X_local_minibatch.to(device)
        y_local_minibatch = y_local_minibatch.to(device)

        y_pred = model(X_local_minibatch)                # fwd the bass (forward pass)
        loss = loss_function(y_pred, y_local_minibatch)  # compute loss
        loss.backward()                                  # reeeeewind (backward pass)
        optimizer.step()                                 # parameter update

        train_running_loss += loss.detach().item()       # unpacks the tensor into a scalar value
        train_acc += model.get_accuracy(y_pred, y_local_minibatch)

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
            for i in range(num_dev_batches):
                X_local_validation_minibatch, y_local_validation_minibatch = (
                    dev_X[i * batch_size : (i + 1) * batch_size,],
                    dev_Y[i * batch_size : (i + 1) * batch_size,],
                )

                X_local_minibatch = X_local_validation_minibatch.permute(1, 0, 2)
                y_local_minibatch = torch.max(y_local_validation_minibatch, 1)[1]

                # to GPU
                X_local_minibatch = X_local_minibatch.to(device)
                y_local_minibatch = y_local_minibatch.to(device)

                y_pred = model(X_local_minibatch)
                val_loss = loss_function(y_pred, y_local_minibatch)

                val_running_loss += (
                    val_loss.detach().item()
                )  # unpacks the tensor into a scalar value
                val_acc += model.get_accuracy(y_pred, y_local_minibatch)

            model.train()  # reset to train mode after iterationg through validation data
            print(
                "Epoch:  %d | NLLoss: %.4f | Train Accuracy: %.2f | Val Loss %.4f  | Val Accuracy: %.2f"
                % (
                    epoch,
                    train_running_loss / num_batches,
                    train_acc / num_batches,
                    val_running_loss / num_dev_batches,
                    val_acc / num_dev_batches,
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

