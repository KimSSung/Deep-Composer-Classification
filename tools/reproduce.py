import sys
import os
import torch

import torch.nn as nn
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from models.wresnet import resnet18, resnet34, resnet101, resnet152, resnet50

import random

torch.manual_seed(333)
random.seed(333)

# score metric
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet50(2, 13)
# model = nn.DataParallel(model)
model = model.cuda()
model_fname = os.listdir("/data/drum/bestmodel/model/")
checkpoint = torch.load("/data/drum/bestmodel/model/" + model_fname[0])
model.load_state_dict(checkpoint["model.state_dict"])

print(">>> MODEL LOADED.")

valid_loader = torch.load("/data/drum/bestmodel/dataset/valid/valid_loader.pt")
print(">>> TEST LOADER LOADED.")

criterion = nn.CrossEntropyLoss().to(device)

seg_num = 90
label_num = 13
input_shape = (2, 400, 128)

with torch.no_grad():  # important!!! for validation
    # validate mode
    model.eval()

    # average the acc of each batch
    val_loss, val_acc = 0.0, 0.0

    val_preds = []
    val_ground_truths = []

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


        ##### Optional: Remove onset channel = [0]
        ##### Run here when --input_shape 1,400,128
        if int(input_shape[0]) == 1:
            # if torch.sum(train_in[:,1:,:,:]) < torch.sum(train_in[:,:1,:,:]): print("1 is onset")
            val_in = val_in[:, 1:, :, :]  # note channel
            # print(val_in.shape)
            # print(train_out.shape)

        ################################################################

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
        # print("confidence: ")
        # print(batch_confidence)
        
        v_loss = criterion(val_pred, val_out)
        val_loss += v_loss

        # scheduler.step(v_loss)  # for reduceonplateau
        # scheduler.step()  # for cos

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
        # print(dup_list)
        # print(cur_pred_label)
        # print("cur preds:", val_label_pred)
        # print("cur outs:", val_out)
        # print("cur pred label:",cur_pred_label)
        # print("cur true label:", cur_true_label)
        # print("===========================================")
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

    avg_valloss = val_loss / len(valid_loader)

    # score
    # 1. accuracy
    # print("len valid_loader:", len(valid_loader))
    # print("len val_preds:", len(val_preds))
    # print("len val_ground_truths:", len(val_ground_truths))
    print("============================================")

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
    print("Accuracy: {:.4f} | Loss: {:.4f}" "".format(val_acc, avg_valloss))
    print("F1-score: %.4f" % (w_f1score))
    print("{:<30}{:<}".format("Precision", "Recall"))
    for p, r in zip(precision, recall):
        print("{:<30}{:<}".format(p, r))
    print()

