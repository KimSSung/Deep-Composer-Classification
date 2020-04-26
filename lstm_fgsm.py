import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import lr_scheduler
from torchsummary import summary

################################################################################

##model definition
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
        return accuracy.item()


################################################

# genres = ['Classical', 'Jazz','Rock','Country','Pop', 'HipHopRap', 'NewAge','Blues'] #total
genres = ['Classical','Rock','Country'] #best
min_shape= 50 #samples for each genre
# epsilons = [.05, .1, .15, .2, .25, .3]
epsilons = [.005, .01, .015, .020, .025, .03, 0.035, 0.04, 0.045, 0.05, 0.055]

###model
pretrained_model = "/data/lstm_models/2_model2_valloss_12.196326494216919"
print("CUDA Available: ", torch.cuda.is_available())
device = torch.device('cuda:0')

print("Build LSTM RNN model ...")
model = LSTM( #hidden_dim = 80 best
    input_dim=129, hidden_dim=80, batch_size=1, output_dim=len(genres), num_layers=2
)

loss_function = nn.NLLLoss()  # expects outputs from LogSoftmax
model = model.to(device)
#load pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
print("==> MODEL LOADED")
# model.eval()


##load data
input_total=[]
output_total=[]
for genre in genres:
    load_saved = np.load("/data/midi820_input/" + genre + "_input.npy", allow_pickle=True)
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
X,Y = np.asarray(X), np.asarray(Y)
test_X = torch.from_numpy(X).type(torch.Tensor)
test_Y = torch.from_numpy(Y).type(torch.LongTensor)
A = TensorDataset(test_X, test_Y)
test_loader = DataLoader(A, batch_size=1, shuffle=True)


###########################attack functions##########################

def fgsm_attack(input, epsilon, data_grad):
    #collect element-wise "sign" of the data gradient
    sign_data_grad = data_grad.sign()
    #adjust each element of input
    perturbed_input = input + epsilon*sign_data_grad
    perturbed_input = torch.clamp(perturbed_input, 0, 1) #clip to range[0,1]
    return perturbed_input

def test(model, device, test_loader, epsilon):

    # Accuracy counter
    correct = 0
    adv_examples = []
    counter=0
    orig_wrong = 0
    for data, target in test_loader:
        counter +=1
        data =data.permute(1, 0, 2)
        target = torch.max(target, 1)[1]

        data, target = data.to(device), target.to(device)
        data.requires_grad = True #for attack
        init_output = model(data)
        init_pred = torch.max(init_output, 1)[1].view(target.size()).data
        # print(init_pred)
        #if correct, skip
        if(init_pred.item() != target.item()):
            orig_wrong += 1
            # print("{}: correct! --- pred:{} orig:{}]".format(counter,init_pred.item(),target.item()))
            continue
        # print("{}: wrong!--- pred:{} orig:{}]".format(counter,init_pred.item(),target.item()))
        #if wrong, attack
        loss = loss_function(init_output, target)  # compute loss
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        #generate perturbed input
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        new_output = model(perturbed_data)
        new_pred = torch.max(new_output, 1)[1].view(target.size()).data

        #check for success
        if new_pred.item() == target.item():
            correct += 1
        else:
            if len(adv_examples) < 5: #save for later
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), new_pred.item(), adv_ex))

    #calculate final accuracy of attack
    final_acc = correct / float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    return final_acc, adv_examples, orig_wrong, correct

##########################################################
#run attack

accuracies = []
examples = []
# ep = 0.01

for ep in epsilons:

    acc, ex, orig_wrong, correct = test(model, device, test_loader, ep)
    accuracies.append(acc)
    examples.append(ex)
    print("{} were originally predicted wrong, out of {} total data".format(orig_wrong, min_shape*len(genres)))
    print("{} examples were still classified correctly, out of {} attempts".format(correct, min_shape*len(genres)-orig_wrong))
print("Some adversarial attacks generated:")
print(examples)

#Draw Results
plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .055, step=0.005))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()
