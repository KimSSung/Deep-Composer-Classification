import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
<<<<<<< HEAD
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
=======
from torch.utils.data import DataLoader, TensorDataset
>>>>>>> ab7c03260d44501b2b7eab366a6063fb8d1bf6d6
import os
# import sklearn
# import torchaudio
import torch
import matplotlib.pyplot as plt
from torch import utils
import numpy as np
from sklearn.model_selection import train_test_split
# from tqdm import tqdm
import random
# import torchsummary
from torch.optim import lr_scheduler

from ResNet import resnet18, resnet101, resnet152, resnet50
# from CustomCNN import CustomCNN

# dataloader
from MIDIDataset import MIDIDataset

torch.manual_seed(123)
import torch.nn as nn


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

'''
input_total=[]
output_total=[]
for genre in genres:

<<<<<<< HEAD
	load_saved = np.load("/data/midi820_drum/" + genre + "_input.npy", allow_pickle=True)[:200]
	if(load_saved.shape[0] < min_shape):
		min_shape = load_saved.shape[0] # num of data in genre
	output_temp = [genres.index(genre)]*load_saved.shape[0]
	output_total.append(output_temp)
	input_total.append(load_saved)
=======
    load_saved = np.load("/data/midi820_drum/" + genre + "_input.npy", allow_pickle=True)[:200]
    if(load_saved.shape[0] < min_shape):
        min_shape = load_saved.shape[0] # num of data in genre
    output_temp = [genres.index(genre)]*load_saved.shape[0]
    output_total.append(output_temp)
    input_total.append(load_saved)
>>>>>>> ab7c03260d44501b2b7eab366a6063fb8d1bf6d6

input_list = []
output_list = []
for i in input_total:
<<<<<<< HEAD
	input_list.extend(i[:min_shape,:,:])
for o in output_total:
	output_list.extend(o[:min_shape])
=======
    input_list.extend(i[:min_shape,:,:])
for o in output_total:
    output_list.extend(o[:min_shape])
>>>>>>> ab7c03260d44501b2b7eab366a6063fb8d1bf6d6
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
'''

<<<<<<< HEAD
each_num = 300

# Loader for origin training
# t = MIDIDataset('/data/midi820_400/', genres, 0, each_num * 0.8)
# v = MIDIDataset('/data/midi820_400/', genres, each_num * 0.8, each_num)

# # create batch
# train_loader = DataLoader(t, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(v, batch_size=batch_size, shuffle=True)

##############################################################################
##############################################################################
# Loader for adversarial training

t_list = []
t_list.append(MIDIDataset('/data/attacks/vel_deepfool/train/', -1, -1, genres, 'flat')) # not use start, end index for 'flat'
t_list.append(MIDIDataset('/data/midi820_400/train/', 0, each_num * 0.8, genres, 'folder'))
t = ConcatDataset(t_list)


v2 = MIDIDataset('/data/midi820_400/valid/', 0, each_num * 0.2, genres, 'folder')
v_list = []
v_list.append(MIDIDataset('/data/attacks/vel_deepfool/valid/', -1, -1, genres, 'flat'))
v_list.append(v2)
v1 = ConcatDataset(v_list) # test + attack test 


# train + attack train
train_loader = DataLoader(t, batch_size=batch_size, shuffle=True)
# test + attack test = TandAT
val_loader_1 = DataLoader(v1, batch_size=batch_size, shuffle=True)
# Only Test = T
val_loader_2 = DataLoader(v2, batch_size=batch_size, shuffle=True)



# print('###############################################')
=======


'''
each_num = 300
t = MIDIDataset('/data/midi820_400/', genres, 0, each_num * 0.8)
v = MIDIDataset('/data/midi820_400/', genres, each_num * 0.8, each_num)

# create batch
train_loader = DataLoader(t, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(v, batch_size=batch_size, shuffle=True)


# print("Training X shape: " + str(trn_X.shape))
# print("Training Y shape: " + str(trn_Y.shape))
# print("Validation X shape: " + str(val_X.shape))
# print("Validation Y shape: " + str(val_Y.shape))

print('###############################################')
>>>>>>> ab7c03260d44501b2b7eab366a6063fb8d1bf6d6
# print('train_loader:',train_loader)
# print('train_loader_len:', len(train_loader))

# save train_loader & valid_loader
torch.save(train_loader, TRAIN_LOADER_SAVE_PATH + 'train_loader.pt')
print("train_loader saved!")
<<<<<<< HEAD
torch.save(val_loader_1, VALID_LOADER_SAVE_PATH + 'valid_loader_TandAT.pt')
print("valid_loader_TandAT saved!")
torch.save(val_loader_2, VALID_LOADER_SAVE_PATH + 'valid_loader_T.pt')
print("valid_loader_T saved!")
=======
torch.save(val_loader, VALID_LOADER_SAVE_PATH + 'valid_loader.pt')
print("valid_loader saved!")
'''

>>>>>>> ab7c03260d44501b2b7eab366a6063fb8d1bf6d6



train_loader = torch.load(TRAIN_LOADER_SAVE_PATH + 'train_loader.pt')
print("train_loader loaded!")
<<<<<<< HEAD
val_loader_1 = torch.load(VALID_LOADER_SAVE_PATH + 'valid_loader_TandAT.pt')
print("valid_loader_TandAT loaded!")
val_loader_2 = torch.load(VALID_LOADER_SAVE_PATH + 'valid_loader_T.pt')
print("valid_loader_T loaded!")
=======
val_loader = torch.load(VALID_LOADER_SAVE_PATH + 'valid_loader.pt')
print("valid_loader loaded!")
>>>>>>> ab7c03260d44501b2b7eab366a6063fb8d1bf6d6

##############################################################


# Define model
model = resnet50(129, num_genres)
# model = CustomCNN(129, num_genres)
# print(model)



<<<<<<< HEAD
#hyper params
num_epochs = 40
num_batches = len(train_loader)
# num_dev_batches = len(val_loader)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-6) # 0.00005
=======
'''
#hyper params
num_epochs = 200
num_batches = len(train_loader)
num_dev_batches = len(val_loader)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-6)
>>>>>>> ab7c03260d44501b2b7eab366a6063fb8d1bf6d6
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
<<<<<<< HEAD


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


=======
val_file_names = [] # file name list
>>>>>>> ab7c03260d44501b2b7eab366a6063fb8d1bf6d6

min_valloss = 10000.0
for epoch in range(num_epochs):

<<<<<<< HEAD
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


		# 1. Test + Attack Test -> val_loader_1
		if epoch == val_term:
			avg_valloss_1, avg_valacc_1, val_file_names_1 = Test(val_loader_1, model, save_filename=True)
			# save val file names
			torch.save(val_file_names_1, VALID_FILENAME_PATH + 'val_file_names_TandAT.pt')
			filenames = torch.load(VALID_FILENAME_PATH + 'val_file_names_TandAT.pt')
			print('Test and Attack test val file names len:',len(filenames))

		else:
			avg_valloss_1, avg_valacc_1, _ = Test(val_loader_1, model, save_filename=False)

		# 2. Only Test
		if epoch == val_term:
			avg_valloss_2, avg_valacc_2, val_file_names_2 = Test(val_loader_2, model, save_filename=True)
			# save val file names
			torch.save(val_file_names_2, VALID_FILENAME_PATH + 'val_file_names_T.pt')
			filenames = torch.load(VALID_FILENAME_PATH + 'val_file_names_T.pt')
			print('Only Test val file names len:',len(filenames))

		else:
			avg_valloss_2, avg_valacc_2, _ = Test(val_loader_2, model, save_filename=False)


		lr = optimizer.param_groups[0]['lr']
		print('''epoch: {}/{} | trn loss: {:.4f} | trn acc: {:.2f}%| lr: {:.6f} |
val_TandAT loss: {:.4f} | val_TandAT acc: {:.2f}% |
val_T loss: {:.4f} | val_T acc: {:.2f}% | '''
			  	.format(epoch + 1, num_epochs,
					trn_running_loss / num_batches, trn_acc / num_batches, lr,
					avg_valloss_1, avg_valacc_1,
					avg_valloss_2, avg_valacc_2
					))

		# save model
		if True: # avg_valloss_1 < min_valloss
			min_valloss = avg_valloss_1
			torch.save({'epoch':epoch,
						'model.state_dict':model.state_dict(),
						'loss':avg_valloss_1,
						'acc':avg_valacc_1}, MODEL_SAVE_PATH + 'Res50_val_TandAT_loss_' + str(float(avg_valloss_1)) + '_acc_' + str(float(avg_valacc_1)) + '.pt')
			print('model saved!')

			# load:
			# the_model = TheModelClass(*args, **kwargs)
			# the_model.eval()
			# the_model.load_state_dict(torch.load(PATH))


		# trn_loss_list.append(trn_running_loss / num_batches)
		# val_loss_list.append(val_loss / num_dev_batches)
		# trn_acc_list.append(trn_acc / num_batches)
		# val_acc_list.append(val_acc / num_dev_batches)

		# reinit to 0
		# trn_running_loss = 0.0
		# trn_total = 0
		# trn_correct = 0
	



# # save val file names
# torch.save(val_file_names, VALID_FILENAME_PATH + 'val_file_names.pt')
# filenames = torch.load(VALID_FILENAME_PATH + 'val_file_names.pt')
# print('val file names len:',len(filenames))

=======
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
                
                if epoch == val_term: # only when epoch = val_term(10)
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
                lr = optimizer.param_groups[0]['lr']

                # accuracy
                _, val_label_pred = torch.max(val_pred.data, 1)
                val_total = val_out.size(0)
                val_correct = (val_label_pred == val_out).sum().item()
                val_acc += val_correct / val_total * 100
                print("correct: {}, total: {}, acc: {}".format(val_correct, val_total, val_correct/val_total*100))

            avg_valloss = val_loss / num_dev_batches
            avg_valacc = val_acc / num_dev_batches

            print("epoch: {}/{} | trn loss: {:.4f} | trn acc: {:.2f}%| val loss: {:.4f} | val acc: {:.2f}% | lr: {:.6f}"
                  .format(epoch + 1, num_epochs,
                        trn_running_loss / num_batches,
                        trn_acc / num_batches,
                        avg_valloss,
                        avg_valacc,
                        lr))

            # save model
            if avg_valloss < min_valloss:
                min_valloss = avg_valloss
                torch.save({'epoch':epoch,
                            'model.state_dict':model.state_dict(),
                            'loss':avg_valloss,
                            'acc':avg_valacc}, MODEL_SAVE_PATH + 'Conv_valloss_' + str(avg_valloss) + '_acc_' + str(avg_valacc) + '.pt')
                print('model saved!')

                # load:
                # the_model = TheModelClass(*args, **kwargs)
                # the_model.eval()
                # the_model.load_state_dict(torch.load(PATH))


            trn_loss_list.append(trn_running_loss / num_batches)
            val_loss_list.append(val_loss / num_dev_batches)
            trn_acc_list.append(trn_acc / num_batches)
            val_acc_list.append(val_acc / num_dev_batches)

            # reinit to 0
            # trn_running_loss = 0.0
            # trn_total = 0
            # trn_correct = 0
    



# save val file names
torch.save(val_file_names, VALID_FILENAME_PATH + 'val_file_names.pt')
filenames = torch.load(VALID_FILENAME_PATH + 'val_file_names.pt')
print('val file names len:',len(filenames))
'''
>>>>>>> ab7c03260d44501b2b7eab366a6063fb8d1bf6d6





'''
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

'''

<<<<<<< HEAD
'''
=======
>>>>>>> ab7c03260d44501b2b7eab366a6063fb8d1bf6d6
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
<<<<<<< HEAD
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
=======
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
>>>>>>> ab7c03260d44501b2b7eab366a6063fb8d1bf6d6
