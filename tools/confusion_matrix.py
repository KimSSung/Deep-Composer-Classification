from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os
import torch

import torch.nn as nn
import numpy as np

# score metric
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

# to import from sibling folders
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data_loader import MIDIDataset


from models.wresnet import resnet18, resnet34, resnet101, resnet152, resnet50

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = resnet50(2, 13)
# model = nn.DataParallel(model)
model.cuda()
model_fname = os.listdir("/data/drum/bestmodel/model/")
checkpoint = torch.load("/data/drum/bestmodel/model/" + model_fname[0])
model.load_state_dict(checkpoint["model.state_dict"])

print(">>> MODEL LOADED.")

valid_loader = torch.load("/data/drum/bestmodel/dataset/valid/valid_loader.pt")
print(">>> TEST LOADER LOADED.")

label_num = 13
seg_num = 90

val_preds = []
val_ground_truths = []
with torch.no_grad():  # important!!! for validation
	# validate mode
	model.eval()

	# average the acc of each batch
	val_acc = 0.0

	# val_total = 0 # = len(valid_loader)
	val_correct = 0

	cur_midi_preds = []
	cur_midi_truths = []
	pred_labels = [-1] * label_num
	cur_true_label = -1
	cur_pred_label = -1  # majority label
	for j, valset in enumerate(valid_loader):

		# val_in, val_out = valset
		val_in = valset["X"]
		val_out = valset["Y"]

		cur_true_label = int(val_out[0])
		cur_midi_truths.append(cur_true_label)
		if cur_true_label != int(val_out[-1]):
			print("Error!! => Diff label in same batch.")
			break

		# to GPU
		val_in = val_in.to(device)
		val_out = val_out.to(device)

		# forward
		val_pred = model(val_in)  # probability
		val_softmax = torch.softmax(val_pred, dim=1)
		batch_confidence = torch.sum(val_softmax, dim=0)  # =1
		batch_confidence = torch.div(
			batch_confidence, seg_num
		)  # avg value

		# accuracy
		_, val_label_pred = torch.max(val_pred.data, 1)

		# val_total += val_out.size(0)
		# val_correct += (val_label_pred == val_out).sum().item()

		# changed accuracy metric
		# acc for each batch (=> one batch = one midi)
		val_label_pred = val_label_pred.tolist()

		occ = [val_label_pred.count(x) for x in range(label_num)]
		max_vote = max(occ)
		occ = np.array(occ)
		dup_list = np.where(max_vote == occ)[0]
		# returns indices of same max occ
		if len(dup_list) > 1:
			max_confidence = -1.0
			for dup in dup_list:
				if batch_confidence[dup] > max_confidence:
					cur_pred_label = dup
		else:
			cur_pred_label = max(val_label_pred, key=val_label_pred.count)


		if cur_true_label == cur_pred_label:
			val_correct += 1

		# f1 score
		val_preds.append(cur_pred_label)
		val_ground_truths.append(cur_true_label)

		# reset for next midi
		cur_midi_preds = []
		cur_midi_truths = []
		pred_labels = [-1] * label_num
		cur_true_label = -1
		cur_pred_label = -1  # majority label


	val_acc = val_correct / len(valid_loader)

	# 2. weighted f1-score
	w_f1score = f1_score(val_ground_truths, val_preds, average="weighted")

	precision, recall, f1, supports = precision_recall_fscore_support(
		val_ground_truths,
		val_preds,
		average=None,
		labels=list(range(label_num)),
		warn_for=tuple(),
	)

	# print learning process
	print("\n######## Valid #########")
	print("Accuracy: {:.4f}".format(val_acc))
	print("F1-score: {:.4f}".format(w_f1score))
	print("{:<30}{:<}".format("Precision", "Recall"))
	for p, r in zip(precision, recall):
		print("{:<30}{:<}".format(p, r))
	print()



# confusion matrix
conf = confusion_matrix(val_ground_truths, val_preds)
print(">>> CONFUSION MATRIX:")
print(conf)
# print(type(conf))

# Sorting by age
# ['Scriab','Debus','Scarl','Liszt','Schube','Chop','Bach',
#  'Brahm','Haydn','Beethov','Schum','Rach','Moza']
# 1. Baroque: Scarlatti / Bach => [2, 6]
# 2. Classical: Haydn / Mozart / Beethoven / Schubert => [4, 8, 9, 12]
# 3. Romanticism: Schumann / Chopin / Liszt / Brahms / Debussy
#                 / Rachmaninoff / Scriabin => [0, 1, 3, 5, 7, 10, 11]
axis_labels = np.array(
	[
		"Scriab",
		"Debus",
		"Scarl",
		"Liszt",
		"Schube",
		"Chop",
		"Bach",
		"Brahm",
		"Haydn",
		"Beethov",
		"Schum",
		"Rach",
		"Moza",
	]
)

want_order = [2, 6, 4, 8, 9, 12, 0, 1, 3, 5, 7, 10, 11]


sort = True
if sort:
	conf = conf[want_order, :][:, want_order]
	axis_labels = axis_labels[want_order]


normalize = True
val_format = ""  # heatmap print value format
if normalize:
	conf = np.round(conf.astype("float") / conf.sum(axis=1)[:, np.newaxis], 2)
	print(">>> Normalized confusion matrix:")
	print(conf)
	val_format = ".2f"

else:
	print("Confusion matrix, without normalization")
	val_format = "d"


# fig, ax = plt.subplots()
# im = ax.imshow(conf)
# ax.set_title("Confusion matrix [13 composers]")
# fig.tight_layout()
# plt.savefig('confmat.png', dpi=300)


sns.heatmap(
	conf,
	annot=True,
	annot_kws={"size": 7},
	fmt=val_format,
	xticklabels=axis_labels,
	yticklabels=axis_labels,
	cmap=plt.cm.bone,
)

plt.title("Confusion Matrix => [x : Pred, y : True]")
plt.savefig("confmat.png", dpi=700)
