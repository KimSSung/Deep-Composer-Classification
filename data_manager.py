import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

class GTZANDataset(Dataset):
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __getitem__(self, index):
		return self.x[index], self.y[index]

	def __len__(self):
		return self.x.shape[0]

# Function to get genre index for the give file
def get_label(file_name, hparams):
	genre = file_name.split('.')[0]
	label = hparams.genres.index(genre)
	return label

def load_dataset(set_name, hparams):
	x = []
	y = []

	dataset_path = os.path.join(hparams.feature_path, set_name)
	for root,dirs,files in os.walk(dataset_path):
		for file in files:
			# print('file name:', file)
			data = np.load(os.path.join(root,file))
			label = get_label(file, hparams)
			x.append(data)
			y.append(label)

			print('data shape:',data.shape)

	x = np.stack(x)
	y = np.stack(y)

	print("x shape:",x.shape) # (8,128)
	print("y shape:",y.shape)
	print('--------------------------')

	return x,y

def get_dataloader(hparams):
	x_train, y_train = load_dataset('train', hparams)
	x_valid, y_valid = load_dataset('valid', hparams)
	x_test, y_test = load_dataset('test', hparams)


	# print("train:",x_train.shape)
	# print("valid:",x_valid.shape)
	# print("test:",x_test.shape)

	mean = np.mean(x_train)
	std = np.std(x_train)
	x_train = (x_train - mean)/std
	x_valid = (x_valid - mean)/std
	x_test = (x_test - mean)/std

	train_set = GTZANDataset(x_train, y_train)
	vaild_set = GTZANDataset(x_valid, y_valid)
	test_set = GTZANDataset(x_test, y_test)

	train_loader = DataLoader(train_set, batch_size=hparams.batch_size, shuffle=True, drop_last=False)
	valid_loader = DataLoader(vaild_set, batch_size=hparams.batch_size, shuffle=False, drop_last=False)
	test_loader = DataLoader(test_set, batch_size=hparams.batch_size, shuffle=False, drop_last=False)



	return train_loader, valid_loader, test_loader