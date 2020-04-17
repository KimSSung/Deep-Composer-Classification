import torch
import os
import numpy as np
import torchaudio

from Fin_hparams import hparams

TENSOR_PATH = '../../../../data/lmd/tensors/genre7/'
GENRES = ['RnB', 'Pop_Rock', 'Country', 'New_Age', 'Jazz', 'Folk', 'Latin']

# print 
print('###################################################')
for f in os.listdir(TENSOR_PATH):

	file_path = TENSOR_PATH + f
	tensor_list = torch.load(file_path) # len: 300
	print('▶ ',f, 'len : ', len(tensor_list))

print('###################################################')


# original melspectogram
# def melspectrogram(file_name, hparams):
	
# 	####### original librosa version #######

# 	y, sr = librosa.load(os.path.join(hparams.dataset_path, file_name), hparams.sample_rate)
# 	wave_tensor = torch.tensor(y)
	
# 	'''
	
# 	# separate it to time windows, and apply the Fourier Transform on each time window
# 	S = librosa.stft(y, n_fft=hparams.fft_size, hop_length=hparams.hop_size, win_length=hparams.win_size)
# 	print(S.shape)
# 	# non linear transformation matrix
# 	# partitions the Hz scale into bins, and transforms each bin into a corresponding bin in the Mel Scale, using a overlapping triangular filters
# 	# low freq: high energy, high freq: low energy (not considerable)
# 	mel_basis = librosa.filters.mel(hparams.sample_rate, n_fft=hparams.fft_size, n_mels=hparams.num_mels)
# 	# the amplitude of one time window, compute the dot product with mel to perform the transformation
# 	mel_S = np.dot(mel_basis, np.abs(S))
# 	mel_S = np.log10(1+10*mel_S)
# 	mel_S = mel_S.T

# 	# print(mel_S.shape)

# 	'''

# 	####### torch audio version (MelSpectogram ver) #######
# 	# file already pickle of tensor
# 	# with open(os.path.join(hparams.dataset_path, file_name), 'rb') as f:
# 	# 	wav_tensor = pickle.load(f)
# 	mel_S = torchaudio.transforms.MelSpectrogram(sample_rate=hparams.sample_rate, n_fft=hparams.fft_size,
# 													win_length=hparams.win_size, hop_length=hparams.hop_size, n_mels=hparams.num_mels)(wave_tensor)

# 	mel_S = torch.log10(1+10*mel_S)

# 	####### torch audio version (stft * men_basis -> log -> transpose) #######
# 	# separate it to time windows, and apply the Fourier Transform on each time window
	
# 	# S = torch.stft(wave_tensor, n_fft=hparams.fft_size, hop_length=hparams.hop_size, win_length=hparams.win_size)
# 	# print(S.shape)
# 	# # non linear transformation matrix
# 	# # partitions the Hz scale into bins, and transforms each bin into a corresponding bin in the Mel Scale, using a overlapping triangular filters
# 	# # low freq: high energy, high freq: low energy (not considerable)
# 	# mel_basis = torchaudio.transforms.MelScale(n_mels=hparams.num_mels, sample_rate=hparams.sample_rate, n_stft=hparams.fft_size)
# 	# print(mel_basis.fb.shape)
# 	# # the amplitude of one time window, compute the dot product with mel to perform the transformation
# 	# mel_S = mel_basis(torch.abs(S))
# 	# mel_S = torch.log10(1+10*mel_S)
# 	# mel_S = torch.t(mel_S)



# 	print(mel_S.shape)

# 	return mel_S

# Function to get genre index for the give file
def get_label(file_name, hparams):
	genre = file_name.split('.')[0]
	label = GENRES.index(genre)
	return label


def tensor2mel(wave_tensor, hparams):

	mel_S = torchaudio.transforms.MelSpectrogram(sample_rate=hparams.sample_rate, n_fft=hparams.fft_size,
													win_length=hparams.win_size, hop_length=hparams.hop_size, n_mels=hparams.num_mels)(wave_tensor)

	mel_S = torch.log10(1+10*mel_S)

	# print(mel_S.shape)

	return mel_S


def resize_array(array, length):
	array_T = array.t() # transpose
	# print("transpose shape:",array_T.shape)
	resize_array = torch.zeros((length, array_T.shape[1]))
	if array_T.shape[0] >= length:
		resize_array = array_T[:length]
	else:
		resize_array[:array_T.shape[0]] = array_T

	return resize_array

	# resize_array = np.zeros((length, array.shape[1]))
	# if array.shape[0] >= length:
	# 	resize_array = array[:length]
	# else:-------
	# 	resize_array[:array.shape[0]] = array


	# return resize_array

def main():
	print("Extracting Feature")
	# list_names = ['train_list.txt', 'valid_list.txt', 'test_list.txt']
	err = 0

	# train: 70 * 5 genre -> 70%
	# valid: 15 * 5 genre -> 15%
	# test: 15 * 5 genre -> 15%
	for f in os.listdir(TENSOR_PATH):

		num = 0 # if num < 70: train, elif num < 85: test, elif num < 100: valid
		label = get_label(f, hparams)
		print('Label:', GENRES[label]) # load from 'GENRES' -> because of Pop_Rock & New_Age ('_')

		file_path = TENSOR_PATH + f
		tensor_list = torch.load(file_path) # len: 100
		for i, tensor in enumerate(tensor_list):
			tensor_list[i] = tensor.float()


		shape_list = []
		shape_avg = 0 # to see avg of shape : 1024 can be too large
		for k, wav_tensor in enumerate(tensor_list):
			num += 1
			
			if num < 70: # 100 * 0.7
				set_name = 'train'
			elif num < 85: # 100 * 0.15
				set_name = 'test'
			elif num < 100:
				set_name = 'valid'
			else: # only take 100 for each genre
				break

			feature = tensor2mel(wav_tensor, hparams) # [1,128, something]
			# print('before feature size:', feature.shape)
			feature = feature.reshape(feature.shape[1], feature.shape[2]) # [128, something]
			# print('feature size:',feature.shape)

			# to see avg of shape
			shape_list.append(feature.shape[1])

			feature = resize_array(feature, hparams.feature_length)
			print('after resize:', feature.shape) # (1024, 128)

			# Data Arguments
			num_chunks = feature.shape[0]//hparams.num_mels # 1024 / 128 = 8, 672 / 224 = 3
			# print(num_chunks)
			# print('###################################################')

			# data_chuncks = np.split(feature, num_chunks)
			data_chuncks = torch.chunk(feature, num_chunks, dim=0) # (1024, 128) => 8 x (128,128) # not use torch.split !!!
			
			for idx, i in enumerate(data_chuncks):

				save_name = hparams.genres[label] + '_' + str(k).zfill(4) + str(idx) + '.npy' # Classical_00011.npy
				# print(save_name)
				save_path = os.path.join(hparams.feature_path, set_name, hparams.genres[label])
				# print('save_path: ', save_path)
				
				if not os.path.exists(save_path):
					os.makedirs(save_path)

				if i.shape[0] == 128 and i.shape[1] == 128 and len(data_chuncks) == 8: # num_mels = 128
				

					np.save(os.path.join(save_path, save_name), i.type(torch.FloatTensor))
					print(os.path.join(save_path, save_name))
					

				else:
					print('error on file: ', file_name)
					err += 1


		# print(hparams.genres[label], ' - shape_list: ', shape_list)
		print(hparams.genres[label], 'shape avg: ', sum(shape_list) / len(shape_list))
		print('###################################################')

			
	print('total error:', err)

	print('finished')

if __name__ == '__main__':
	main()
