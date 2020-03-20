import torch
torch.manual_seed(123)
import torch.nn as nn

class upchannel(nn.Module):
	def __init__(self, hparams):
		super(upchannel, self).__init__()

		self._extractor = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),

			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),

			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2),

			nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=4),

			nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=4)
		)

		self._classifier = nn.Sequential(
										 nn.Linear(in_features=512, out_features=256),
										 nn.ReLU(),
										 nn.Dropout(),
										 nn.Linear(in_features=256, out_features=128),
										 nn.ReLU(),
										 nn.Dropout(),
										 nn.Linear(in_features=128, out_features=len(hparams.genres)))
		# self.apply(self._init_weights)

	def forward(self, x):
		x = torch.unsqueeze(x,1)
		x = self._extractor(x)
		x = x.view(x.size(0), -1)
		score = self._classifier(x)
		return score
	#
	# def _init_weights(self, layer) -> None:
	#     if isinstance(layer, nn.Conv2d):
	#         nn.init.kaiming_uniform_(layer.weight)
	#     elif isinstance(layer, nn.Linear):
	#         nn.init.xavier_uniform_(layer.weight)