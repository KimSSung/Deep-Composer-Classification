import torchaudio
from matplotlib import pyplot as plt
from hparams import hparams
import librosa
import numpy as np

filename = "./classical.00099.wav"
waveform, sample_rate = torchaudio.load(filename)

print("Shape of waveform: {}".format(waveform.size()))
print(type(waveform))
print(waveform.shape[1])
print("Sample rate of waveform: {}".format(sample_rate))


plt.figure()
plt.plot(waveform.t().numpy())


def melspectrogram(file_name, hparams):
	y, sr = librosa.load(file_name)
	print("y:", type(y), "    |     ", y)
	print("sr:", type(sr), "    |     ", sr)
	
	
	# separate it to time windows, and apply the Fourier Transform on each time window
	S = librosa.stft(y, n_fft=hparams.fft_size, hop_length=hparams.hop_size, win_length=hparams.win_size)
	print("S:", type(S), "    |     ", S)

	# non linear transformation matrix
	# partitions the Hz scale into bins, and transforms each bin into a corresponding bin in the Mel Scale, using a overlapping triangular filters
	# low freq: high energy, high freq: low energy (not considerable)
	mel_basis = librosa.filters.mel(hparams.sample_rate, n_fft=hparams.fft_size, n_mels=hparams.num_mels)
	print("mel_basis:", type(mel_basis), "    |     ", mel_basis)

	# the amplitude of one time window, compute the dot product with mel to perform the transformation
	mel_S = np.dot(mel_basis, np.abs(S))
	mel_S = np.log10(1+10*mel_S)
	mel_S = mel_S.T

	return mel_S

melspectrogram(filename, hparams)