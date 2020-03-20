import torch
torch.manual_seed(123)
import torch.nn as nn

class RCNN(nn.Module):
	def __init__(self, num_genres):
		super(RCNN, self).__init__()

		self._extractor = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
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
			nn.MaxPool2d(kernel_size=8)
		)

		self._rnnModule = nn.Sequential(
				 nn.GRU(512, 512, batch_first=False,bidirectional=True),
				#nn.LSTM(512, 512, batch_first=False, bidirectional=True),
				)

		self._classifier = nn.Sequential(nn.Linear(in_features=1024, out_features=512),
										 nn.ReLU(),
										 nn.Dropout(),
										 nn.Linear(in_features=512, out_features=256),
										 nn.ReLU(),
										 nn.Dropout(),
										 nn.Linear(in_features=256, out_features=num_genres))
		self.apply(self._init_weights)

	def forward(self, x):
		x = torch.unsqueeze(x,1)
		x = self._extractor(x)
		x = x.permute(3,0,1,2)
		x = x.view(x.size(0), x.size(1), -1)
		x, hn = self._rnnModule(x)
		x = x.permute(1, 2, 0)
		x = x.view(x.size(0), -1)
		score = self._classifier(x)
		return score

	def _init_weights(self, layer) -> None:
		if isinstance(layer, nn.Conv1d):
			nn.init.kaiming_uniform_(layer.weight)
		elif isinstance(layer, nn.Linear):
			nn.init.xavier_uniform_(layer.weight)