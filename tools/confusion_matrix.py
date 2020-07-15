from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from matplotlib import pyplot
import sys
import os
import torch

import torch.nn as nn
import numpy as np

from data_loader import MIDIDataset


# to import from sibling folders
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# print(sys.path)

from models.resnet import resnet18, resnet101, resnet152, resnet50

genres = ["Classical", "Rock", "Country", "GameMusic"]
model = resnet50(129, len(genres))

model_fname = os.listdir('/data/drum/bestmodel/model/')
checkpoint = torch.load('/data/drum/bestmodel/model/' + model_fname[0])
model.load_state_dict(checkpoint["model.state_dict"])
print(">>> MODEL LOADED.")

test_loader = torch.load('/data/drum/bestmodel/dataset/test/valid_loader.pt')
print(">>> TEST LOADER LOADED.")


y_pred = []
y_true = []
with torch.no_grad():  # important!!! for validation
	# validate mode
	model.eval()

	for j, valset in enumerate(test_loader):
		val_in, val_out, file_name = valset

		# predict
		val_pred = model(val_in)
		_, val_label_pred = torch.max(val_pred.data, 1)

		y_pred.extend(list(map(int, val_label_pred)))


		# true_label
		y_true.extend(list(map(int, val_out)))



# confusion matrix
conf = confusion_matrix(y_true, y_pred)
print(">>> CONFUSION MATRIX:")
print(conf)
