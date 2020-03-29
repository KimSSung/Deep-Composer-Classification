import torchaudio

ten, sr = torchaudio.load('./classical.00001.wav', sampling_rate = 10000)
print(ten.dtype)