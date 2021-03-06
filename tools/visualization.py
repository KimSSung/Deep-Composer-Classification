from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from scipy.stats import spearmanr,pointbiserialr
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
from sklearn.metrics import classification_report

# to import from sibling folders
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from tools.data_loader import MIDIDataset


from models.resnet import resnet18, resnet34, resnet101, resnet152, resnet50

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


LOAD_PATH = "/data/drum/bestmodel/"
SAVE_PATH = "./visual/"

class Visualization:
    def __init__(self, label_num=13, seg_num=90, normalize=True, bar=True, mode=None):

        self.label_num = label_num
        self.seg_num = seg_num

        self.val_preds = []
        self.val_ground_truths = []


        self.mode = mode # 'age' or 'data' or 'birth' or None


        self.normalize = normalize
        self.bar = bar # T of F

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet50(2, self.label_num)
        
        # data parallel
        # self.model = nn.DataParallel(self.model)
        
        self.model.cuda()

        model_fname = os.listdir(LOAD_PATH + "model/")
        checkpoint = torch.load(LOAD_PATH + "model/" + model_fname[0])
        self.model.load_state_dict(checkpoint["model.state_dict"])
        print(">>> MODEL LOADED.")

        self.valid_loader = torch.load(
            LOAD_PATH + "dataset/valid/valid_loader.pt"
        )
        print(">>> TEST LOADER LOADED.")

        # for saving pdf
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)


        self.axis_labels = np.array(
            [
                "Scri",
                "Debu",
                "Scar",
                "Lisz",
                "F.Sch",
                "Chop",
                "Bach",
                "Brah",
                "Hayd",
                "Beet",
                "R.Sch",
                "Rach",
                "Moza",
            ]
        )

    def run(self):
        self.validation()

    def validation(self):
        with torch.no_grad():
            self.model.eval()

            # average the acc of each batch
            val_acc = 0.0
            # val_total = 0 # = len(self.valid_loader)
            val_correct = 0

            cur_midi_preds = []
            cur_midi_truths = []
            pred_labels = [-1] * self.label_num
            cur_true_label = -1
            cur_pred_label = -1  # majority label
            for j, valset in enumerate(self.valid_loader):

                # val_in, val_out = valset
                val_in = valset["X"]
                val_out = valset["Y"]

                cur_true_label = int(val_out[0])
                cur_midi_truths.append(cur_true_label)
                if cur_true_label != int(val_out[-1]):
                    print("Error!! => Diff label in same batch.")
                    return

                # to GPU
                val_in = val_in.to(self.device)
                val_out = val_out.to(self.device)

                # forward
                val_pred = self.model(val_in)  # probability
                val_softmax = torch.softmax(val_pred, dim=1)
                batch_confidence = torch.sum(val_softmax, dim=0)  # =1
                batch_confidence = torch.div(
                    batch_confidence, self.seg_num
                )  # avg value

                # accuracy
                _, val_label_pred = torch.max(val_pred.data, 1)

                # val_total += val_out.size(0)
                # val_correct += (val_label_pred == val_out).sum().item()

                # changed accuracy metric
                # acc for each batch (=> one batch = one midi)
                val_label_pred = val_label_pred.tolist()

                occ = [val_label_pred.count(x) for x in range(self.label_num)]
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
                self.val_preds.append(cur_pred_label)
                self.val_ground_truths.append(cur_true_label)

                # reset for next midi
                cur_midi_preds = []
                cur_midi_truths = []
                pred_labels = [-1] * self.label_num
                cur_true_label = -1
                cur_pred_label = -1  # majority label

            val_acc = val_correct / len(self.valid_loader)

            # 2. weighted f1-score
            w_f1score = f1_score(
                self.val_ground_truths, self.val_preds, average="weighted"
            )

            precision, recall, f1, supports = precision_recall_fscore_support(
                self.val_ground_truths,
                self.val_preds,
                average=None,
                labels=list(range(self.label_num)),
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

            # generate & save confusion matrix
            if self.bar:
                self.draw_bar(self.val_ground_truths, self.val_preds)
            else:
                self.generate_matrix(self.val_ground_truths, self.val_preds)

    def generate_matrix(self, true, pred):
        # confusion matrix
        conf = confusion_matrix(true, pred)
        print(">>> CONFUSION MATRIX:")
        print(conf)

        # sorting by age
        # ['Scriab','Debus','Scarl','Liszt','Schube','Chop','Bach',
        #  'Brahm','Haydn','Beethov','Schum','Rach','Moza']
        # 1. Baroque: Scarlatti / Bach => [2, 6]
        # 2. Classical: Haydn / Mozart / Beethoven / Schubert => [4, 8, 9, 12]
        # 3. Romanticism: Schumann / Chopin / Liszt / Brahms / Debussy
        #                 / Rachmaninoff / Scriabin => [0, 1, 3, 5, 7, 10, 11]

        if self.mode:
            if self.mode == 'age':
                want_order = [2, 6, 4, 8, 9, 12, 0, 1, 3, 5, 7, 10, 11]

            elif self.mode == 'birth':
                want_order = [2, 6, 8, 12, 9, 4, 5, 10, 3, 7, 1, 0, 11]

            print(">>> Sorting.....")
            conf = conf[want_order, :][:, want_order]
            conf_axis_labels = self.axis_labels[want_order]

        else: # mode == None
            conf_axis_labels = self.axis_labels

        val_format = ""  # heatmap print value format
        if self.normalize:
            conf = np.round(conf.astype("float") / conf.sum(axis=1)[:, np.newaxis], 2)
            print(">>> Normalized confusion matrix:")
            print(conf)
            val_format = ".2f"

        else:
            print(">>> Confusion matrix, without normalization")
            val_format = "d"

        sns.set(font_scale=0.7)
        plt.figure(figsize=(6,6))
        ax = sns.heatmap(
            conf,
            annot=True,
            annot_kws={"size": 8},
            fmt=val_format,
            xticklabels=conf_axis_labels,
            yticklabels=conf_axis_labels,
            cmap=plt.cm.bone,
            cbar=True,
            linecolor='white',
            linewidths=0.01
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7.5)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=7.5, rotation_mode='anchor', ha='right')

        plt.title("Confusion Matrix, with normalization", fontsize=12)
        plt.xlabel("Predicted label", fontsize=10)
        plt.ylabel("True label", fontsize=10)
        plt.savefig(SAVE_PATH + "confmat.pdf", dpi=1000)

        
    def draw_bar(self, true, pred):

        print(">>> Drawing bar graph....")
        # no sorting for bar graph
        dic = classification_report(true, pred, target_names=self.axis_labels, output_dict=True)
        print(dic)
        bar_values = []
        for i in range(self.label_num):
            # print(dic[self.axis_labels[i]]['f1-score'])
            bar_values.append(dic[self.axis_labels[i]]['f1-score'])
            print(self.axis_labels[i] + " -> " + str(dic[self.axis_labels[i]]['f1-score']))


        # sorting by age or # of data
        if self.mode in ['age', 'data', 'birth']:

            if self.mode == 'age':
                want_order = [2, 6, 4, 8, 9, 12, 0, 1, 3, 5, 7, 10, 11]
            elif self.mode == 'data':
                want_order = [9, 4, 3, 5, 1, 12, 11, 8, 10, 7, 6, 0, 2]
            else: # self.mode == 'birth'
                want_order = [2, 6, 8, 12, 9, 4, 5, 10, 3, 7, 1, 0, 11]
            
            b = np.array(bar_values)
            bar_values = b[want_order]
            bar_axis_labels = self.axis_labels[want_order]
            self.SpearmanCorr(bar_values, bar_axis_labels)
            
        else: # self.mode == None
            self.mode = '' # for save pdf name
            bar_axis_labels = self.axis_labels

        # print('bar vals:', bar_values)
        # print('labels:', bar_axis_labels)

        indices = range(len(bar_values))

        fig, ax = plt.subplots(figsize=(21,11))
        ax.bar(indices, bar_values, width=0.5, align="center")
        for i in range(len(bar_values)):
            plt.annotate('{0:03.3f}'.format(bar_values[i]).lstrip('0'), (-0.47 + i, bar_values[i] + 0.01), fontsize=30)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(False)
        ax.spines['left'].set_linewidth(False)
        plt.xticks(indices, bar_axis_labels, fontsize=30, rotation=45)
        plt.setp(ax.get_yticklabels(), fontsize=30)
        plt.ylim([0.6, 1.01])

        # plt.title("Bar chart, with normalization", fontsize=12)
        # plt.xlabel("composers", fontsize=15)
        plt.ylabel("F1 score", fontsize=40)
        plt.grid(b=True, linewidth=0.1, axis='y')
        # plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)
        plt.savefig(SAVE_PATH + "barchart_" + self.mode + ".pdf", dpi=1000)
        print("\n>>> barchart_" + self.mode + ".pdf saved.")


    def SpearmanCorr(self, values, labels):

        if self.mode == 'birth' or self.mode == 'data':
            if self.mode == 'birth':
                print('--------  Organized with birth date --------')
            elif self.mode == 'data':
                print('--------  Organized with # of data --------')

            coef = spearmanr(values, np.arange(0,13,step=1))[0]
            p_value = spearmanr(values, np.arange(0,13,step=1))[1]
            print('Spearman Rank: ', round(coef,4))
            print('P_Value: ', round(p_value,4))

            # Calculating Point Biserial Rank
            coef = pointbiserialr(values, np.arange(0,13,step=1))[0]
            p_value = pointbiserialr(values, np.arange(0,13, step=1))[1]
            print('Point Biserial Rank: ', round(coef,4))
            print('P_value: ', round(p_value,4))

            return

        # elif self.mode == 'age' :

        #     print('--------  Organized with Age (3 class) --------')

        #     coef = spearmanr(values, np.array([1,1,2,2,2,2,3,3,3,3,3,3,3]))[0]
        #     p_value = spearmanr(values, np.array([1,1,2,2,2,2,3,3,3,3,3,3,3]))[1]
        #     print('Spearman Rank: ', round(coef, 4))
        #     print('P_Value: ', round(p_value, 4))

        #     # Calculating Point Biserial Rank
        #     coef = pointbiserialr(values, np.array([1,1,2,2,2,2,3,3,3,3,3,3,3]))[0]
        #     p_value = pointbiserialr(values, np.array([1,1,2,2,2,2,3,3,3,3,3,3,3]))[1]
        #     print('Point Biserial Rank: ', round(coef, 4))
        #     print('P_value: ', round(p_value, 4))


# Testing
if __name__ == "__main__":

    # for base train
    # # 0. not sorted confusion matrix
    # temp0 = Visualization(label_num=13, seg_num=90, normalize=True, bar=False, mode=None)
    # temp0.run()

    # # 1. age confusion matrix
    # temp1 = Visualization(label_num=13, seg_num=90, normalize=True, bar=False, mode='age')
    # temp1.run()

    # 2. birth confusion matrix
    temp1 = Visualization(label_num=13, seg_num=90, normalize=True, bar=False, mode='birth')
    temp1.run()

    # # 3. birth bar chart
    # temp2 = Visualization(label_num=13, seg_num=90, normalize=True, bar=True, mode='birth')
    # temp2.run()

