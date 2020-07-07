# Trainer class (config.mode = 'basetrain' -> base training / 'advtrain' -> adversarial training)

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



class Trainer:

	def __init__(self, args):
		self.config = args

		#for GPU use
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
		os.environ["CUDA_VISIBLE_DEVICES"]=self.config.gpu
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.data_load(self.config.mode)
		self.num_batches = len(self.train_loader)

		# Define model
		num_genres = len(self.config.genres)
		self.model = resnet50(self.config.input_shape[0], num_genres)
		# model = convnet(self.config.input_shape[0], num_genres)
		self.model = self.model.to(self.device)

		self.criterion = nn.CrossEntropyLoss()
		self.criterion = self.criterion.to(self.device)

		self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-6) # 0.00005
		# self.optimizer = optim.SGD(self.model.parameters(),lr=self.config.learning_rate)
		# self.optimizer = optim.ASGD(self.model.parameters(), lr=self.config.learning_rate, weight_decay=1e-6)
		# self.optimizer = optim.SparseAdam(self.model.parameters(), lr=self.config.learning_rate, betas=(0.9, 0.999), eps=1e-08)
		self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.epochs)
		# self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',factor=0.5,patience=10,verbose=True) #0.5 best for midi370



	def data_load(self, mode):
		if mode == 'basetrain':
			print('>>>>>> Base Training <<<<<<\n')

			# Loader for base training
			t = MIDIDataset(self.config.train_input_path, 0, self.config.genre_datanum * 0.8, self.config.genres, 'folder')
			v = MIDIDataset(self.config.valid_input_path, 0, self.config.genre_datanum * 0.2, self.config.genres, 'folder')

			# create batch
			self.train_loader = DataLoader(t, batch_size=self.config.train_batch, shuffle=True)
			self.valid_loader = DataLoader(v, batch_size=self.config.valid_batch, shuffle=True)


			###################### Loader for base training #############################
			
			# save train_loader & valid_loader
			torch.save(self.train_loader, self.config.trainloader_save_path + 'train_loader.pt')
			print("train_loader saved!")
			torch.save(self.valid_loader, self.config.validloader_save_path + 'valid_loader.pt')
			print("valid_loader saved!")

			# load train_loader & valid_loader (to check whether well saved)
			self.train_loader = torch.load(self.config.trainloader_save_path + 'train_loader.pt')
			print("train_loader loaded!")
			self.valid_loader = torch.load(self.config.validloader_save_path + 'valid_loader.pt')
			print("valid_loader loaded!")

			#############################################################################

		elif mode == 'advtrain':
			print('>>>>>> Adversarial Training <<<<<<\n')

			################### Loader for adversarial training #########################
			t_list = []
			t_list.append(MIDIDataset(self.config.attacked_train_input_path, -1, -1, self.config.genres, 'flat')) # not use start, end index for 'flat'
			t_list.append(MIDIDataset(self.config.train_input_path, 0, self.config.genre_datanum * 0.8, self.config.genres, 'folder'))
			t = ConcatDataset(t_list)


			# Caution: attacked_train_input_path & valid_path must be checked !!!!!!!!
			v2 = MIDIDataset(self.config.valid_input_path, 0, self.config.genre_datanum * 0.2, self.config.genres, 'folder')
			v_list = []
			v_list.append(MIDIDataset(self.config.attacked_valid_input_path, -1, -1, self.config.genres, 'flat'))
			v_list.append(v2)
			v1 = ConcatDataset(v_list) # test + attack test 


			# train + attack train
			self.train_loader = DataLoader(t, batch_size=self.config.train_batch, shuffle=True)
			# test + attack test = TandAT
			self.valid_loader_1 = DataLoader(v1, batch_size=self.config.valid_batch, shuffle=True)
			# Only Test = T
			self.valid_loader_2 = DataLoader(v2, batch_size=self.config.valid_batch, shuffle=True)


			# save adv_train_loader & valid_loader (to check whether well saved)
			torch.save(self.train_loader, self.config.trainloader_save_path + 'adv_train_loader.pt')
			print("adv_train_loader saved!")
			torch.save(self.valid_loader_1, self.config.validloader_save_path + 'adv_valid_loader_TandAT.pt')
			print("adv_valid_loader_TandAT saved!")
			torch.save(self.valid_loader_2, self.config.validloader_save_path + 'adv_valid_loader_T.pt')
			print("adv_valid_loader_T saved!")



			self.train_loader = torch.load(self.config.trainloader_save_path + 'adv_train_loader.pt')
			print("adv_train_loader loaded!")
			self.valid_loader_1 = torch.load(self.config.validloader_save_path + 'adv_valid_loader_TandAT.pt')
			print("adv_valid_loader_TandAT loaded!")
			self.valid_loader_2 = torch.load(self.config.validloader_save_path + 'adv_valid_loader_T.pt')
			print("adv_valid_loader_T loaded!")

			#############################################################################



	def set_mode(self, mode='train'):
		if mode == 'train':
			self.model.train()
		elif mode == 'eval':
			self.model.eval()
		else:
			raise('Mode error. It should be either train or eval')


	def train(self, mode):

		self.set_mode('train') # model.train()

		# train
		for epoch in range(self.config.epochs):

			trn_running_loss, trn_acc = 0.0, 0.0
			# trn_correct = 0
			# trn_total = 0
			for i, trainset in enumerate(self.train_loader):
				#train_mode
				
				#unpack
				train_in, train_out, file_name = trainset
				# print(train_in.shape)
				# print(train_out.shape)
				#use GPU
				train_in = train_in.to(self.device)
				train_out = train_out.to(self.device)
				#grad init
				self.optimizer.zero_grad()

				#forward pass
				# print(train_in.shape)
				train_pred = self.model(train_in)
				#calculate acc
				_, label_pred = torch.max(train_pred.data, 1)
				trn_total = train_out.size(0)
				trn_correct = (label_pred == train_out).sum().item()
				trn_acc += (trn_correct / trn_total * 100)
				#calculate loss
				t_loss = self.criterion(train_pred, train_out)
				#back prop
				t_loss.backward()
				#weight update
				self.optimizer.step()

				trn_running_loss += t_loss.item()


			#print learning process
			print(
				"Epoch:  %d | Train Loss: %.4f | Train Accuracy: %.2f"
				% (epoch, trn_running_loss / self.num_batches,
					# (trn_correct/trn_total *100))
					trn_acc / self.num_batches)
			)


			################## TEST ####################
			val_term = 10
			min_valloss = 10000.0

			if epoch % val_term == 0:

				if epoch == 0:

					if mode == 'basetrain':
						avg_valloss, avg_valacc, val_file_names = self.test(self.valid_loader, self.model, save_filename=True)
						# save val file names
						torch.save(val_file_names,self.config.valid_filename_save_path + 'val_file_names.pt')
						filenames = torch.load(self.config.valid_filename_save_path + 'val_file_names.pt')
						print('val file names len:',len(filenames))

					elif mode == 'advtrain':
						# 1. Test + Attack Test -> adv_valid_loader_1
						avg_valloss_1, avg_valacc_1, val_file_names_1 = self.test(self.valid_loader_1, self.model, save_filename=True)
						torch.save(val_file_names_1, self.config.valid_filename_save_path + 'adv_val_file_names_TandAT.pt')
						filenames = torch.load(self.config.valid_filename_save_path + 'adv_val_file_names_TandAT.pt')
						print('Test and Attack test val file names len:',len(filenames))

						# 2. Only Test
						avg_valloss_2, avg_valacc_2, val_file_names_2 = self.test(self.valid_loader_2, self.model, save_filename=True)
						torch.save(val_file_names_2, self.config.valid_filename_save_path + 'adv_val_file_names_T.pt')
						filenames = torch.load(self.config.valid_filename_save_path + 'adv_val_file_names_T.pt')
						print('Only Test val file names len:',len(filenames))


				else:

					if mode == 'basetrain':
						avg_valloss, avg_valacc, _ = self.test(self.valid_loader, self.model, save_filename=False)

					elif mode == 'advtrain':
						avg_valloss_1, avg_valacc_1, _ = self.test(self.valid_loader_1, self.model, save_filename=False)
						avg_valloss_2, avg_valacc_2, _ = self.test(self.valid_loader_2, self.model, save_filename=False)
	



				lr = self.optimizer.param_groups[0]['lr']

				if mode == 'basetrain':
					print('''epoch: {}/{} | trn loss: {:.4f} | trn acc: {:.2f}%| lr: {:.6f} |
		val loss: {:.4f} | val acc: {:.2f}%'''
						.format(epoch + 1, self.config.epochs,
						trn_running_loss / self.num_batches, trn_acc / self.num_batches, lr,
						avg_valloss, avg_valacc
						))

					# save model
					if avg_valloss < min_valloss:
						min_valloss = avg_valloss
						torch.save({'epoch':epoch,
									'model.state_dict':self.model.state_dict(),
									'loss':avg_valloss,
									'acc':avg_valacc}, self.config.model_save_path + self.config.model_name + '_valloss_' + str(float(avg_valloss)) + '_acc_' + str(float(avg_valacc)) + '.pt')
						print('model saved!')


				elif mode == 'advtrain':
					print('''epoch: {}/{} | trn loss: {:.4f} | trn acc: {:.2f}%| lr: {:.6f} |
	val_TandAT loss: {:.4f} | val_TandAT acc: {:.2f}% |
	val_T loss: {:.4f} | val_T acc: {:.2f}% '''
						.format(epoch + 1, self.config.epochs,
						trn_running_loss / self.num_batches, trn_acc / self.num_batches, lr,
						avg_valloss_1, avg_valacc_1,
						avg_valloss_2, avg_valacc_2
						))

					# save model
					if True: # avg_valloss_1 < min_valloss:
						min_valloss = avg_valloss_1
						torch.save({'epoch':epoch,
									'model.state_dict':self.model.state_dict(),
									'loss':avg_valloss_1,
									'acc':avg_valacc_1}, self.config.model_save_path + self.config.model_name + '_val_TandAT_loss_' + str(float(avg_valloss_1)) + '_acc_' + str(float(avg_valacc_1)) + '.pt')
						print('model saved!')




	def test(self, test_loader, model, save_filename=False):
		#############################
		######## Test function ######
		#############################


		val_file_names = [] # file name list
		with torch.no_grad(): # important!!! for validation
			# validate mode
			self.set_mode('eval') # model.eval()

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
				val_in = val_in.to(self.device)
				val_out = val_out.to(self.device)

				# forward
				val_pred = self.model(val_in)
				v_loss = self.criterion(val_pred, val_out)
				val_loss += v_loss

				# scheduler.step(v_loss)  # for reduceonplateau
				self.scheduler.step()       #for cos

				# accuracy
				_, val_label_pred = torch.max(val_pred.data, 1)
				val_total = val_out.size(0)
				val_correct = (val_label_pred == val_out).sum().item()
				val_acc += val_correct / val_total * 100
				print("correct: {}, total: {}, acc: {}".format(val_correct, val_total, val_correct/val_total*100))

			avg_valloss = val_loss / len(test_loader)
			avg_valacc = val_acc / len(test_loader)


		self.set_mode('train') # model.train()

		return avg_valloss, avg_valacc, val_file_names



# # Testing
# config, unparsed = get_config()
# with open("config.txt", "w") as f: # execute on /train/ folder
# 	f.write('Parameters for ' + config.mode + ':\n\n')
# 	for arg in vars(config):
# 		argname = arg
# 		contents = str(getattr(config, arg))
# 		#print(argname + ' = ' + contents)
# 		f.write(argname + ' = ' + contents + '\n')

# temp = Trainer(config)
# temp.train(config.mode)