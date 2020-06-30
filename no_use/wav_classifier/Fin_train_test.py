'''
train_test.py
A file for training model for genre classification.
Please check the device in hparams.py before you run this code.
https://github.com/Dohppak/Music_Genre_Classification_Pytorch
audio_augmentation.py -> feature_extraction.py -> custom_cnn_2d_pytorch.py
'''
import torch
torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(0)

from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.optim.lr_scheduler import CosineAnnealingLR

import Fin_data_manager as data_manager

# model
# from custom_upchannel32 import * # acc: 0.8172 for lr: 1e-2/2
# from custom_rcnn import * # acc: 0.8254 for lr: 1e-2/2
# from DenseNet import *
from ResNet2 import * # 0.7
# import torchvision.models as models # for resnet50 # not yet

from Fin_hparams import hparams
# Wrapper class to run PyTorch model
class Runner(object):
	def __init__(self, hparams):
		# self.model = upchannel(hparams)
		# self.model = RCNN(hparams)
		self.model = resnet50(1,5)
		# self.model = densenet121()

		self.criterion = torch.nn.CrossEntropyLoss()
		# self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hparams.learning_rate)
		self.optimizer = torch.optim.SGD(self.model.parameters(), lr=hparams.learning_rate, momentum=hparams.momentum, weight_decay=1e-6, nesterov=True)
		
		self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=hparams.factor, patience=hparams.patience, verbose=True)
		# self.scheduler = CosineAnnealingLR(self.optimizer, 10, eta_min=0, last_epoch=-1)

		self.learning_rate = hparams.learning_rate
		self.stopping_rate = hparams.stopping_rate
		self.device = torch.device("cpu")

		if hparams.device > 0:
			torch.cuda.set_device(hparams.device - 1)
			self.model.cuda(hparams.device - 1)
			self.criterion.cuda(hparams.device - 1)
			self.device = torch.device("cuda:" + str(hparams.device - 1))

	# Accuracy function works like loss function in PyTorch
	def accuracy(self, source, target):
		source = source.max(1)[1].long().cpu()
		target = target.long().cpu()
		correct = (source == target).sum().item()

		return correct/float(source.size(0))

	# Running model for train, test and validation. mode: 'train' for training, 'eval' for validation and test
	def run(self, dataloader, mode='train'):
		self.model.train() if mode is 'train' else self.model.eval()

		epoch_loss = 0
		epoch_acc = 0
		for batch, (x, y) in enumerate(dataloader):
			x = x.to(self.device)
			y = y.to(self.device)

			# reshape for resnet
			x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

			prediction = self.model(x)
			# if mode == 'eval': print('prediction: ', prediction.argmax(dim=1), "  |  label: ", y.long())
			loss = self.criterion(prediction, y.long())
			acc = self.accuracy(prediction, y.long())

			if mode is 'train':
				loss.backward()
				self.optimizer.step()
				self.optimizer.zero_grad()

			epoch_loss += prediction.size(0)*loss.item()
			epoch_acc += prediction.size(0)*acc

		epoch_loss = epoch_loss/len(dataloader.dataset)
		epoch_acc = epoch_acc/len(dataloader.dataset)

		return epoch_loss, epoch_acc

	# Early stopping function for given validation loss
	def early_stop(self, loss, epoch):
		self.scheduler.step(loss, epoch) # ReduceLROnPlateau
		# self.scheduler.step(loss) # Consine

		self.learning_rate = self.optimizer.param_groups[0]['lr']
		stop = self.learning_rate < self.stopping_rate

		return stop

def device_name(device):
	if device == 0:
		device_name = 'CPU'
	else:
		device_name = 'GPU:' + str(device - 1)

	return device_name

def main():
	train_loader, valid_loader, test_loader = data_manager.get_dataloader(hparams)
	# print("train:", len(train_loader))
	# print("valid:", len(train_loader))
	# print("test:", len(train_loader))
	

	runner = Runner(hparams)

	print('Training on ' + device_name(hparams.device))
	for epoch in range(hparams.num_epochs):
		train_loss, train_acc = runner.run(train_loader, 'train')
		valid_loss, valid_acc = runner.run(valid_loader, 'eval')

		print("[Epoch %d/%d] [Train Loss: %.4f] [Train Acc: %.4f] [Valid Loss: %.4f] [Valid Acc: %.4f]" %
			  (epoch + 1, hparams.num_epochs, train_loss, train_acc, valid_loss, valid_acc))

		if runner.early_stop(valid_loss, epoch + 1):
			break

	test_loss, test_acc = runner.run(test_loader, 'eval')
	print("Training Finished")
	print("Test Accuracy: %.2f%%" % (100*test_acc))

if __name__ == '__main__':
	main()