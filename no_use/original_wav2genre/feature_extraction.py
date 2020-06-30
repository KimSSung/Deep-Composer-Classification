import os
import numpy as np
import torch
import librosa
import torchaudio
import pickle


from hparams import hparams

# def load_list(list_name, hparams):
# 	with open(os.path.join(hparams.dataset_path, list_name)) as f:
# 		file_names = f.read().splitlines()
# 		return file_names

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

def melspectogram2(wav_tensor, hparams):

	mel_S = torchaudio.transforms.MelSpectrogram(sample_rate=hparams.sample_rate, n_fft=hparams.fft_size,
													win_length=hparams.win_size, hop_length=hparams.hop_size, n_mels=hparams.num_mels)(wave_tensor)

	mel_S = torch.log10(1+10*mel_S)

	print(mel_S.shape)

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
	list_names = ['train_list.txt', 'valid_list.txt', 'test_list.txt']
	err = 0

	for list_name in list_names:


		set_name = list_name.replace('_list.txt', '')
		file_names = load_list(list_name, hparams)

		for file_name in file_names:


			feature = melspectrogram(file_name, hparams)
			print('feature size:',feature.shape)
			feature = resize_array(feature, hparams.feature_length)
			print('after resize:', feature.shape) # (1024, 128)
			

			# Data Arguments
			num_chunks = feature.shape[0]//hparams.num_mels # 1024 / 128 = 8, 672 / 224 = 3
			print(num_chunks)

			# print('-------------------')
			# print(num_chunks)
			# print(feature)
			print('-------------------')

			# data_chuncks = np.split(feature, num_chunks)
			data_chuncks = torch.chunk(feature, num_chunks, dim=0) # (1024, 128) => 8 x (128,128) # not use torch.split !!!
			
			# print(len(data_chuncks)) # 8 / if torch.split : this becomes 128

			

			for idx, i in enumerate(data_chuncks):

				save_path = os.path.join(hparams.feature_path, set_name, file_name.split('/')[0])
				save_name = file_name.split('/')[1].split('.wav')[0]+str(idx)+".npy"
				
				if not os.path.exists(save_path):
					os.makedirs(save_path)

				if i.shape[0] == 128 and i.shape[1] == 128 and len(data_chuncks) == 8: # num_mels = 128
				

					np.save(os.path.join(save_path, save_name), i.type(torch.FloatTensor))
					print(os.path.join(save_path, save_name))
					

				else:
					print('error on file: ', file_name)
					err += 1
								

			
	print('total error:', err)

	print('finished')

if __name__ == '__main__':
	main()