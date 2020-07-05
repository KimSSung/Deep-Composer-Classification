from config import get_config

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

import os
import sys 

# to import from sibling folders
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from models.resnet import resnet18, resnet101, resnet152, resnet50
# from model.convnet import CustomCNN

# dataloader
from tools.data_loader import MIDIDataset

torch.manual_seed(123)



def Test(test_loader, model, criterion, save_filename=False):
	#############################
	######## Test function ######
	#############################

	val_file_names = [] # file name list
	with torch.no_grad(): # important!!! for validation
		# validate mode
		model.eval()

		#average the acc of each batch
		val_loss, val_acc = 0.0, 0.0
		# val_correct = 0
		# val_total = 0

		for j, valset in enumerate(test_loader):
			val_in, val_out, file_name = valset

			# save valid file name only at first validation
			
			if save_filename: # only when epoch = val_term(10)
				for fname in file_name:
					val_file_names.append(fname)

			# to GPU
			val_in = val_in.to(device)
			val_out = val_out.to(device)
			criterion = criterion.to(device)

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

		avg_valloss = val_loss / len(test_loader)
		avg_valacc = val_acc / len(test_loader)

	return avg_valloss, avg_valacc, val_file_names


def main(config):

	##################
	####  Setting ####
	##################

	################### Loader for adversarial training #########################
	t_list = []
	t_list.append(MIDIDataset(config.attacked_train_input_path, -1, -1, config.genres, 'flat')) # not use start, end index for 'flat'
	t_list.append(MIDIDataset(config.train_input_path, 0, config.genre_datanum * 0.8, config.genres, 'folder'))
	t = ConcatDataset(t_list)


	# Caution: attacked_train_input_path & valid_path must be checked !!!!!!!!
	v2 = MIDIDataset(config.valid_input_path, 0, config.genre_datanum * 0.2, config.genres, 'folder')
	v_list = []
	v_list.append(MIDIDataset(config.attacked_valid_input_path, -1, -1, config.genres, 'flat'))
	v_list.append(v2)
	v1 = ConcatDataset(v_list) # test + attack test 


	# train + attack train
	adv_train_loader = DataLoader(t, batch_size=config.train_batch, shuffle=True)
	# test + attack test = TandAT
	adv_valid_loader_1 = DataLoader(v1, batch_size=config.valid_batch, shuffle=True)
	# Only Test = T
	adv_valid_loader_2 = DataLoader(v2, batch_size=config.valid_batch, shuffle=True)


	# save adv_train_loader & valid_loader
	torch.save(adv_train_loader, config.trainloader_save_path + 'adv_train_loader.pt')
	print("adv_train_loader saved!")
	torch.save(adv_valid_loader_1, config.validloader_save_path + 'adv_valid_loader_TandAT.pt')
	print("adv_valid_loader_TandAT saved!")
	torch.save(adv_valid_loader_2, config.validloader_save_path + 'adv_valid_loader_T.pt')
	print("adv_valid_loader_T saved!")



	adv_train_loader = torch.load(config.trainloader_save_path + 'adv_train_loader.pt')
	print("adv_train_loader loaded!")
	adv_valid_loader_1 = torch.load(config.validloader_save_path + 'adv_valid_loader_TandAT.pt')
	print("adv_valid_loader_TandAT loaded!")
	adv_valid_loader_2 = torch.load(config.validloader_save_path + 'adv_valid_loader_T.pt')
	print("adv_valid_loader_T loaded!")

	#############################################################################


	num_batches = len(adv_train_loader)
	# num_dev_batches = len(valid_loader)


	# train
	min_valloss = 10000.0
	for epoch in range(config.epochs):

		trn_running_loss, trn_acc = 0.0, 0.0
		# trn_correct = 0
		# trn_total = 0
		for i, trainset in enumerate(adv_train_loader):
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


			# 1. Test + Attack Test -> adv_valid_loader_1
			if epoch == val_term:
				avg_valloss_1, avg_valacc_1, val_file_names_1 = Test(adv_valid_loader_1, model, criterion, save_filename=True)
				# save val file names
				torch.save(val_file_names_1, VALID_FILENAME_PATH + 'adv_val_file_names_TandAT.pt')
				filenames = torch.load(VALID_FILENAME_PATH + 'adv_val_file_names_TandAT.pt')
				print('Test and Attack test val file names len:',len(filenames))

			else:
				avg_valloss_1, avg_valacc_1, _ = Test(adv_valid_loader_1, model, criterion, save_filename=False)

			# 2. Only Test
			if epoch == val_term:
				avg_valloss_2, avg_valacc_2, val_file_names_2 = Test(adv_valid_loader_2, model, criterion, save_filename=True)
				# save val file names
				torch.save(val_file_names_2, VALID_FILENAME_PATH + 'adv_val_file_names_T.pt')
				filenames = torch.load(VALID_FILENAME_PATH + 'adv_val_file_names_T.pt')
				print('Only Test val file names len:',len(filenames))

			else:
				avg_valloss_2, avg_valacc_2, _ = Test(adv_valid_loader_2, model, criterion, save_filename=False)


			lr = optimizer.param_groups[0]['lr']
			print('''epoch: {}/{} | trn loss: {:.4f} | trn acc: {:.2f}%| lr: {:.6f} |
	val_TandAT loss: {:.4f} | val_TandAT acc: {:.2f}% |
		 val_T loss: {:.4f} | val_T acc: {:.2f}% '''
					.format(epoch + 1, config.epochs,
						trn_running_loss / num_batches, trn_acc / num_batches, lr,
						avg_valloss_1, avg_valacc_1,
						avg_valloss_2, avg_valacc_2
						))

			# save model
			if True: # avg_valloss_1 < min_valloss:
				min_valloss = avg_valloss_1
				torch.save({'epoch':epoch,
							'model.state_dict':model.state_dict(),
							'loss':avg_valloss_1,
							'acc':avg_valacc_1}, config.model_save_path + config.model_name + '_val_TandAT_loss_' + str(float(avg_valloss_1)) + '_acc_' + str(float(avg_valacc_1)) + '.pt')
				print('model saved!')



if __name__ == '__main__':
	config, unparsed = get_config()
	with open("adv_config.txt", "w") as f: # execute on /train/ folder
		f.write("Parameters for adversarial training:\n\n")
		for arg in vars(config):
			argname = arg
			contents = str(getattr(config, arg))
			#print(argname + ' = ' + contents)
			f.write(argname + ' = ' + contents + '\n')

	
	#for GPU use
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Define model
	num_genres = len(config.genres)
	model = resnet50(config.input_shape[0], num_genres)
	# model = convnet(config.input_shape[0], num_genres)
	model = model.to(device)

	criterion = nn.CrossEntropyLoss()
	criterion = criterion.to(device)

	optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-6) # 0.00005
	# optimizer = optim.SGD(model.parameters(),lr=0.0001)
	# optimizer = optim.ASGD(model.parameters(), lr=0.00005, weight_decay=1e-6)
	# optimizer = optim.SparseAdam(model.parameters(), lr=0.00005, betas=(0.9, 0.999), eps=1e-08)
	scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
	# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.5,patience=10,verbose=True) #0.5 best for midi370

	main(config)
	


