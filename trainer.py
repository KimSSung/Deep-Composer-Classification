# Trainer class (config.mode = 'basetrain' -> base training / 'advtrain' -> adversarial training)

# from config import get_config

import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torch

# from tqdm import tqdm
from torch.optim import lr_scheduler

# to import from sibling folders
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.convnet import convnet

# dataloader
from tools.data_loader import MIDIDataset

# score metric
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support

# transform
import torchvision
from tools.transformation import ToTensor, Segmentation, Transpose

# torch.manual_seed(333)


class Trainer:
    def __init__(self, args):
        self.config = args

        # 0 : acc / 1: loss / 2: f1 / 3: precision / 4: recall
        self.best_valid = [-1.0, 30000.0, -1.0, [], []]
        self.seg_num = 20  # change this

        # for GPU use
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # moved to main
        # os.environ["CUDA_VISIBLE_DEVICES"] = self.config.gpu # moved to main
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.omitlist = []
        if self.config.omit:
            self.omitlist = self.config.omit.split(",")  # ['2', '5']. str list.

        self.label_num = self.config.composers - len(self.omitlist)
        print("==> Total label # :", self.label_num)
        print()
        # if age == True ==> label: 0, 1, 2
        if self.config.age:
            self.label_num = 3

        self.data_load(self.config.mode)
        self.num_batches = len(self.train_loader)

        # Define model
        self.model = self.model_selection()
        self.model = nn.DataParallel(self.model)
        self.model.cuda()

        self.criterion = nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(self.device)

        # Define optimizer
        self.optimizer = self.optim_selection()

        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.epochs
        )
        # self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',factor=0.5,patience=10,verbose=True) #0.5 best for midi370

    def model_selection(self):
        if self.config.model_name == "resnet18":
            return resnet18(int(self.config.input_shape[0]), self.label_num)
        elif self.config.model_name == "resnet34":
            return resnet34(int(self.config.input_shape[0]), self.label_num)
        elif self.config.model_name == "resnet50":
            return resnet50(int(self.config.input_shape[0]), self.label_num)
        elif self.config.model_name == "resnet101":
            return resnet101(int(self.config.input_shape[0]), self.label_num)
        elif self.config.model_name == "resnet152":
            return resnet152(int(self.config.input_shape[0]), self.label_num)
        elif self.config.model_name == "convnet":
            return convnet(int(self.config.input_shape[0]), self.label_num)

    def optim_selection(self):
        if self.config.optim == "Nesterov":
            return optim.SGD(
                self.model.parameters(), lr=self.config.lr, momentum=0.9, nesterov=True
            )
        elif self.config.optim == "SGD":
            return optim.SGD(
                self.model.parameters(), lr=self.config.lr, momentum=0.9, nesterov=False
            )
        elif self.config.optim == "Adadelta":  # default lr = 1.0
            return optim.Adadelta(
                self.model.parameters(),
                lr=self.config.lr,
                rho=0.9,
                eps=1e-06,
                weight_decay=1e-6,
            )
        elif self.config.optim == "Adagrad":  # default lr = 0.01
            return optim.Adagrad(
                self.model.parameters(),
                lr=self.config.lr,
                lr_decay=0,
                weight_decay=1e-6,
                initial_accumulator_value=0,
                eps=1e-10,
            )
        elif self.config.optim == "Adam":  # default lr=0.001
            return optim.Adam(
                self.model.parameters(), lr=self.config.lr, weight_decay=1e-6
            )
        elif self.config.optim == "AdamW":  # default lr=0.001
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.01,
                amsgrad=False,
            )
        elif self.config.optim == "SparseAdam":  # default lr = 0.001
            return optim.SparseAdam(
                self.model.parameters(),
                lr=self.config.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
            )
        elif self.config.optim == "Adamax":  # default lr=0.002
            return optim.Adamax(
                self.model.parameters(),
                lr=self.config.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=1e-6,
            )
        elif self.config.optim == "ASGD":
            return optim.ASGD(
                self.model.parameters(),
                lr=self.config.lr,
                lambd=0.0001,
                alpha=0.75,
                t0=1000000.0,
                weight_decay=1e-6,
            )
        elif self.config.optim == "RMSprop":  # default lr=0.01
            return optim.RMSprop(
                self.model.parameters(),
                lr=self.config.lr,
                alpha=0.99,
                eps=1e-08,
                weight_decay=0,
                momentum=0,
                centered=False,
            )
        elif self.config.optim == "Rprop":  # default lr=0.01
            return optim.Rprop(
                self.model.parameters(),
                lr=self.config.lr,
                etas=(0.5, 1.2),
                step_sizes=(1e-06, 50),
            )

    def data_load(self, mode):
        if mode == "basetrain":
            print(">>>>>> Base Training <<<<<<\n")

            # Loader for base training
            trans = None
            if self.config.transform:
                trans = torchvision.transforms.Compose(
                    [Segmentation(), Transpose(), ToTensor()]
                )
            t = MIDIDataset(
                self.config.train_split_path,
                classes=self.label_num,
                omit=self.config.omit,  # str
                seg_num=self.seg_num,
                age=self.config.age,
                transform=trans,
            )
            v = MIDIDataset(
                self.config.test_split_path,
                classes=self.label_num,
                omit=self.config.omit,
                seg_num=self.seg_num,
                age=self.config.age,
                transform=None,
            )

            # create batch
            self.train_loader = DataLoader(
                t, batch_size=self.config.train_batch, shuffle=True
            )
            self.valid_loader = DataLoader(
                v, batch_size=self.config.valid_batch, shuffle=True
            )

            ###################### Loader for base training #############################

            # save train_loader & valid_loader
            torch.save(
                self.train_loader, self.config.trainloader_save_path + "train_loader.pt"
            )
            print("train_loader saved!")
            torch.save(
                self.valid_loader, self.config.validloader_save_path + "valid_loader.pt"
            )
            print("valid_loader saved!")

            # load train_loader & valid_loader (to check whether well saved)
            self.train_loader = torch.load(
                self.config.trainloader_save_path + "train_loader.pt"
            )
            print("train_loader loaded!")
            self.valid_loader = torch.load(
                self.config.validloader_save_path + "valid_loader.pt"
            )
            print("valid_loader loaded!")

            #############################################################################

        elif mode == "advtrain":
            print(">>>>>> Adversarial Training <<<<<<\n")

            ################### Loader for adversarial training #########################
            t_list = []
            t_list.append(
                MIDIDataset(
                    self.config.attacked_train_input_path,
                    -1,
                    -1,
                    self.config.genres,
                    "flat",
                )
            )  # not use start, end index for 'flat'
            t_list.append(
                MIDIDataset(
                    self.config.train_input_path,
                    0,
                    self.config.genre_datanum * 0.8,
                    self.config.genres,
                    "folder",
                )
            )
            t = ConcatDataset(t_list)

            # Caution: attacked_train_input_path & valid_path must be checked !!!!!!!!
            v2 = MIDIDataset(
                self.config.valid_input_path,
                0,
                self.config.genre_datanum * 0.2,
                self.config.genres,
                "folder",
            )
            v_list = []
            v_list.append(
                MIDIDataset(
                    self.config.attacked_valid_input_path,
                    -1,
                    -1,
                    self.config.genres,
                    "flat",
                )
            )
            v_list.append(v2)
            v1 = ConcatDataset(v_list)  # test + attack test

            # train + attack train
            self.train_loader = DataLoader(
                t, batch_size=self.config.train_batch, shuffle=True
            )
            # test + attack test = TandAT
            self.valid_loader_1 = DataLoader(
                v1, batch_size=self.config.valid_batch, shuffle=True
            )
            # Only Test = T
            self.valid_loader_2 = DataLoader(
                v2, batch_size=self.config.valid_batch, shuffle=True
            )

            # save adv_train_loader & valid_loader (to check whether well saved)
            torch.save(
                self.train_loader,
                self.config.trainloader_save_path + "adv_train_loader.pt",
            )
            print("adv_train_loader saved!")
            torch.save(
                self.valid_loader_1,
                self.config.validloader_save_path + "adv_valid_loader_TandAT.pt",
            )
            print("adv_valid_loader_TandAT saved!")
            torch.save(
                self.valid_loader_2,
                self.config.validloader_save_path + "adv_valid_loader_T.pt",
            )
            print("adv_valid_loader_T saved!")

            self.train_loader = torch.load(
                self.config.trainloader_save_path + "adv_train_loader.pt"
            )
            print("adv_train_loader loaded!")
            self.valid_loader_1 = torch.load(
                self.config.validloader_save_path + "adv_valid_loader_TandAT.pt"
            )
            print("adv_valid_loader_TandAT loaded!")
            self.valid_loader_2 = torch.load(
                self.config.validloader_save_path + "adv_valid_loader_T.pt"
            )
            print("adv_valid_loader_T loaded!")

            #############################################################################

    def set_mode(self, mode="train"):
        if mode == "train":
            self.model.train()
        elif mode == "eval":
            self.model.eval()
        else:
            raise ("Mode error. It should be either train or eval")

    def train(self, mode):

        self.set_mode("train")  # model.train()

        # print input shape
        print("\nInput shape:", self.config.input_shape)
        print()

        # train
        for epoch in range(self.config.epochs + 1):

            trn_running_loss, trn_acc = 0.0, 0.0
            train_preds = []
            ground_truths = []
            trn_correct = 0
            trn_total = 0
            for i, trainset in enumerate(self.train_loader):
                # train_mode

                # unpack
                # train_in, train_out = trainset
                train_in = trainset["X"]
                train_out = trainset["Y"]

                ##### Optional: Remove onset channel = [0]
                ##### Run here when --input_shape 1,400,128
                if int(self.config.input_shape[0]) == 1:
                    # if torch.sum(train_in[:,1:,:,:]) < torch.sum(train_in[:,:1,:,:]): print("1 is onset")
                    train_in = train_in[:, 1:, :, :]  # note channel
                    # print(train_in.shape)
                    # print(train_out.shape)

                ################################################################

                # use GPU
                train_in = train_in.to(self.device)
                train_out = train_out.to(self.device)
                # grad init
                self.optimizer.zero_grad()

                # forward pass
                # print(train_in.shape)
                train_pred = self.model(train_in)
                # calculate acc
                _, label_pred = torch.max(train_pred.data, 1)

                # accuracy
                trn_total += train_out.size(0)
                trn_correct += (label_pred == train_out).sum().item()

                # print('-------------------------')
                # print("pred:",label_pred)
                # print("true:",train_out)
                # print()

                # f1 accuracy
                train_preds.extend(label_pred.tolist())
                ground_truths.extend(train_out.tolist())

                # calculate loss
                t_loss = self.criterion(train_pred, train_out)
                # back prop
                t_loss.backward()
                # weight update
                self.optimizer.step()

                trn_running_loss += t_loss.item()

            # score
            # 1. accuracy
            trn_acc = trn_correct / trn_total

            # 2. weighted f1-score
            w_f1score = f1_score(ground_truths, train_preds, average="weighted")

            precision, recall, f1, supports = precision_recall_fscore_support(
                ground_truths,
                train_preds,
                average=None,
                labels=list(range(self.label_num)),
                warn_for=tuple(),
            )
            # print learning process
            print(
                "Epoch:  %d | Train Loss: %.4f | f1-score: %.4f | accuracy: %.4f"
                % (epoch, trn_running_loss / self.num_batches, w_f1score, trn_acc)
            )
            # print("Train accuracy: %.2f" % (trn_acc))
            # print("Precision:", precision)
            # print("Recall:", recall)

            ################## TEST ####################
            val_term = 10
            min_valloss = 10000.0

            if epoch % val_term == 0:

                if epoch == 0:

                    if mode == "basetrain":
                        avg_valloss, avg_valacc = self.test(
                            self.valid_loader, self.model
                        )

                    elif mode == "advtrain":
                        # 1. Test + Attack Test -> adv_valid_loader_1
                        avg_valloss_1, avg_valacc_1 = self.test(
                            self.valid_loader_1, self.model
                        )

                        # 2. Only Test
                        avg_valloss_2, avg_valacc_2 = self.test(
                            self.valid_loader_2, self.model
                        )

                else:

                    if mode == "basetrain":
                        avg_valloss, avg_valacc = self.test(
                            self.valid_loader, self.model
                        )

                    elif mode == "advtrain":
                        avg_valloss_1, avg_valacc_1 = self.test(
                            self.valid_loader_1, self.model
                        )
                        avg_valloss_2, avg_valacc_2 = self.test(
                            self.valid_loader_2, self.model
                        )

                lr = self.optimizer.param_groups[0]["lr"]

                if mode == "basetrain":
                    print(
                        """epoch: {}/{} | trn loss: {:.4f} | lr: {:.6f}""".format(
                            epoch + 1,
                            self.config.epochs,
                            trn_running_loss / self.num_batches,
                            lr,
                        )
                    )

                    # save model
                    if avg_valloss < min_valloss:
                        min_valloss = avg_valloss
                        torch.save(
                            {
                                "epoch": epoch,
                                "model.state_dict": self.model.state_dict(),
                                "loss": avg_valloss,
                                "acc": avg_valacc,
                            },
                            self.config.model_save_path
                            + self.config.model_name
                            + "_valloss_"
                            + str(float(avg_valloss))
                            + "_acc_"
                            + str(float(avg_valacc))
                            + ".pt",
                        )
                        print("model saved!")

                elif mode == "advtrain":
                    print(
                        """epoch: {}/{} | trn loss: {:.4f} | trn acc: {:.2f}%| lr: {:.6f} |
	val_TandAT loss: {:.4f} | val_TandAT acc: {:.2f}% |
	val_T loss: {:.4f} | val_T acc: {:.2f}% """.format(
                            epoch + 1,
                            self.config.epochs,
                            trn_running_loss / self.num_batches,
                            trn_acc / self.num_batches,
                            lr,
                            avg_valloss_1,
                            avg_valacc_1,
                            avg_valloss_2,
                            avg_valacc_2,
                        )
                    )

                    # save model
                    if True:  # avg_valloss_1 < min_valloss:
                        min_valloss = avg_valloss_1
                        torch.save(
                            {
                                "epoch": epoch,
                                "model.state_dict": self.model.state_dict(),
                                "loss": avg_valloss_1,
                                "acc": avg_valacc_1,
                            },
                            self.config.model_save_path
                            + self.config.model_name
                            + "_val_TandAT_loss_"
                            + str(float(avg_valloss_1))
                            + "_acc_"
                            + str(float(avg_valacc_1))
                            + ".pt",
                        )
                        print("model saved!")

        # print best valid f1 score
        print()
        print("######## Best Valid F1-score environ #########")
        print(
            "Accuracy: {:.4f} | Loss: {:.4f}"
            "".format(self.best_valid[0], self.best_valid[1])
        )
        print("F1-score: {:.4f}".format(self.best_valid[2]))
        print("Precision:", self.best_valid[3])
        print("Recall:", self.best_valid[4])
        print()

    def test(self, test_loader, model):
        #############################
        ######## Test function ######
        #############################

        with torch.no_grad():  # important!!! for validation
            # validate mode
            self.set_mode("eval")  # model.eval()

            # average the acc of each batch
            val_loss, val_acc = 0.0, 0.0
            val_preds = []
            val_ground_truths = []
            val_total = 0
            val_correct = 0

            for j, valset in enumerate(test_loader):
                # val_in, val_out = valset
                val_in = valset["X"]
                val_out = valset["Y"]

                ##### Optional: Remove onset channel = [0]
                ##### Run here when --input_shape 1,400,128
                if int(self.config.input_shape[0]) == 1:
                    # if torch.sum(train_in[:,1:,:,:]) < torch.sum(train_in[:,:1,:,:]): print("1 is onset")
                    val_in = val_in[:, 1:, :, :]  # note channel
                    # print(val_in.shape)
                    # print(train_out.shape)

                ################################################################

                # to GPU
                val_in = val_in.to(self.device)
                val_out = val_out.to(self.device)

                # forward
                val_pred = self.model(val_in)
                v_loss = self.criterion(val_pred, val_out)
                val_loss += v_loss

                # scheduler.step(v_loss)  # for reduceonplateau
                self.scheduler.step()  # for cos

                # accuracy
                _, val_label_pred = torch.max(val_pred.data, 1)

                val_total += val_out.size(0)
                val_correct += (val_label_pred == val_out).sum().item()

                # f1 score
                val_preds.extend(val_label_pred.tolist())
                val_ground_truths.extend(val_out.tolist())

                # print(
                # 	"correct: {}, total: {}, f1-score: {}".format(
                # 		val_correct, val_total,
                # 		f1_score(val_out.tolist(), val_label_pred.tolist(), average='weighted')  * 100
                # 	)
                # )

            avg_valloss = val_loss / len(test_loader)

            # score
            # 1. accuracy
            val_acc = val_correct / val_total

            # 2. weighted f1-score
            w_f1score = f1_score(val_ground_truths, val_preds, average="weighted")

            precision, recall, f1, supports = precision_recall_fscore_support(
                val_ground_truths,
                val_preds,
                average=None,
                labels=list(range(self.label_num)),
                warn_for=tuple(),
            )
            # print learning process
            print()
            print("######## Valid #########")
            print("Accuracy: {:.4f} | Loss: {:.4f}" "".format(val_acc, avg_valloss))
            print("F1-score: %.4f" % (w_f1score))
            print("Precision:", precision)
            print("Recall:", recall)
            print()

            if self.best_valid[2] < w_f1score:
                self.best_valid = [val_acc, avg_valloss, w_f1score, precision, recall]

        self.set_mode("train")  # model.train()

        return avg_valloss, w_f1score


# # Testing
# config, unparsed = get_config()
# with open("config.txt", "w") as f: # execute on /train/ folder
# 	f.write('Parameters for ' + config.mode + ':\n\n')
# 	for arg in vars(config):
# 		argname = arg
# 		contents = str(getattr(config, arg))
# 		#print(argname + ' = ' + contents)
# 		f.write(argname + ' = ' + contents + '\n')

# temp = Trainer(config)
# temp.train(config.mode)
