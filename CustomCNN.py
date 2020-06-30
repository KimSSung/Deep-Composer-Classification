import torch
torch.manual_seed(123)
import torch.nn as nn

k = 3 # kernel size
p = 3 # pool size
class CustomCNN(nn.Module):

	def __init__(self, input_size, num_genres):
		super(CustomCNN, self).__init__()
		self.input_size = input_size
		self.output_size = num_genres

		# padding = (n - ((n-k)+1)) / 2 = (k-1)/2 = 1
		self.layer1 = nn.Sequential(
			nn.Conv2d(input_size, 256, kernel_size=k, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=p, stride=p),
			nn.Dropout(p=0.25)
		)
		self.layer2 = nn.Sequential(
			nn.Conv2d(256, 512, kernel_size=k, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=p, stride=p),
			nn.Dropout(p=0.25)
		)
		self.layer3 = nn.Sequential(
			nn.Conv2d(512, 1024, kernel_size=k, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=p, stride=p),
			nn.Dropout(p=0.25)
		)
		self.layer4 = nn.Sequential(
			nn.Conv2d(1024, 2048, kernel_size=k, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=p, stride=p),
			nn.Dropout(p=0.25)
		)

		self.fc = nn.Sequential(
			nn.Flatten(),
			nn.Dropout(p=0.5),
			nn.Linear(8192, 1024),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(1024, 128),
			nn.ReLU(),			
			nn.Dropout(0.25),
			nn.Linear(128, num_genres),
			nn.Softmax(dim=1)
		)

	def forward(self, x):
		out1 = self.layer1(x)
		out2 = self.layer2(out1)
		out3 = self.layer3(out2)
		out4 = self.layer4(out3)
		out = self.fc(out4)

		return out