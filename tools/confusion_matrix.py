from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import os
import torch

import torch.nn as nn
import numpy as np


from data_loader import MIDIDataset


# to import from sibling folders
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
# print(sys.path)

from models.resnet import resnet18, resnet34, resnet101, resnet152, resnet50

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

model = resnet50(2, 13)
model = nn.DataParallel(model).cuda()
model_fname = os.listdir("/data/drum/bestmodel_noaug/model/")
checkpoint = torch.load("/data/drum/bestmodel_noaug/model/" + model_fname[0])
model.load_state_dict(checkpoint["model.state_dict"])

print(">>> MODEL LOADED.")

test_loader = torch.load("/data/drum/bestmodel_noaug/dataset/test/valid_loader.pt")
print(">>> TEST LOADER LOADED.")


y_pred = []
y_true = []
with torch.no_grad():  # important!!! for validation
    # validate mode
    model.eval()

    for j, valset in enumerate(test_loader):
        val_in, val_out = valset

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
