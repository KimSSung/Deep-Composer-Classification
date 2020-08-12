# Configuration file for (base training / adversarial training / adversarial attack)

import argparse

# Basic Configuration
parser = argparse.ArgumentParser(description="Base Training")

parser.add_argument(
    "--composers", default=13, type=int, help="The number of composers.",
)
parser.add_argument(
    "--model_name",
    type=str,
    default="resnet50",
    help="Prefix of model name (resnet18 / resnet34 / resnet50 / resnet101 / resnet152 / convnet)",
)
parser.add_argument(
    "--train_split_path",
    type=str,
    default="/data/split/train.txt",
    help="Path of train.txt (train split text file)",
)
parser.add_argument(
    "--test_split_path",
    type=str,
    default="/data/split/test.txt",
    help="Path of test.txt (test split text file)",
)
parser.add_argument(
    "--optim",
    type=str,
    default="optimizer",
    help="Optimizer [Adadelta, Adagrad, Adam, AdamW, SparseAdam, Adamax, ASGD, RMSprop, Rprop, SGD, Nesterov]",
)




# attacked input
parser.add_argument(
    "--attacked_train_input_path",
    type=str,
    default="/data/attacks/vel_deepfool/train/",
    help="Attacked Train input directory.",
)
parser.add_argument(
    "--attacked_valid_input_path",
    type=str,
    default="/data/attacks/vel_deepfool/valid/",
    help="Attacked Valid input directory.",
)

parser.add_argument(
    "--epochs",
    default=100,
    type=int,
    help="Total number of epochs to run. Not actual epoch.",
)
parser.add_argument(
    "--train_batch", default=20, type=int, help="Batch size for training"
)
parser.add_argument("--valid_batch", default=20, type=int, help="Batch size for valid.")

parser.add_argument("--gpu", default="0,1,2,3", type=str, help="GPU id to use.")

parser.add_argument("--lr", default=0.0005, type=float, help="Model learning rate.")


##shared parameters (train & attack)
parser.add_argument(
    "--mode",
    default="foo",  # force to input mode ^_^
    type=str,
    help="Mode (basetrain / advtrain / attack / generate / convert / split)",
)
parser.add_argument(
    "--model_save_path", default="/data/drum/model/", type=str, help="Model saving path"
)
parser.add_argument(
    "--input_shape", default=(2, 400, 128), type=tuple, help="Input shape."
)
parser.add_argument(
    "--trainloader_save_path",
    default="/data/drum/dataset/train/",
    type=str,
    help="Train loader saving path",
)
parser.add_argument(
    "--validloader_save_path",
    default="/data/drum/dataset/test/",
    type=str,
    help="Test loader saving path",
)


##attack parameters

# 1. basic configurations
# parser.add_argument("--attack_mode", default="base", type=str, help="Attack Mode (base / trained)") # full name of .pt will tell anyway
parser.add_argument("--targeted", default=False, type=bool, help="is Targeted?")
parser.add_argument(
    "--attack_type", default="fgsm", type=str, help="attack (fgsm / deepfool / random)"
)
parser.add_argument(
    "--atk_path",
    default="/data/drum/bestmodel/",
    type=str,
    help="model & data_loader to be attacked [/model/ & /dataset/]",
)
parser.add_argument(
    "--save_atk",
    default=False,
    type=bool,
    help="save some adversarial examples? (True / False)",
)
parser.add_argument(
    "--save_atk_path",
    default="/data/attacks/",
    type=str,
    help="Dir to save successful attacks",
)

# 2. data related
parser.add_argument(
    "--specific_files",
    default=[],
    type=list,
    help="List file names to attack(default = [])",
)
parser.add_argument(
    "--t_or_v", default="v", type=str, help="dataset to attack (t: train / v: valid)"
)
parser.add_argument(
    "--orig",
    default=True,
    type=bool,
    help="attack on original dataset? (default: True)",
)

# 3. specific attack related
parser.add_argument(
    "--epsilons",
    default=[0.0],
    type=list,
    help="list of epsilons [ep0, ep1, ep2..] (default: [0.0])",
)
parser.add_argument(
    "--max_iter", default=10, type=int, help="max iterations for deepfool attack",
)
parser.add_argument(
    "--overshoot", default=5, type=int, help="overshoot for deepfool attack"
)
parser.add_argument("--plot", default=False, type=bool, help="draw plot?")
parser.add_argument(
    "--tempo_range", default=1, type=int, help="+/-n tempo cells to attack"
)

##input generator

parser.add_argument(
    "--midi_files_path",
    default="/data/MAESTRO/maestro-v2.0.0/",
    type=str,
    help="dir to original files (absolute dir)",
)
parser.add_argument(
    "--input_save_path",
    default="/data/which_dir/",  # to avoid overwriting
    type=str,
    help="save path (absolute dir)",
)

parser.add_argument("--overlap", default=False, type=bool, help="enable 50% overlap?")

parser.add_argument(
    "--segment_num",
    default=10,
    type=int,
    help="how many segments per midi?(discard below)",
)

parser.add_argument(
    "--augmentation",
    default="None",
    type=str,
    help="(default=None / transpose / tempo_stretch)",
)

##converter
parser.add_argument(
    "--to_convert_path",
    default="/data/attacks/",
    type=str,
    help="Path for 'only MIDIs' to convert. Path CAN contain any folder !!",
)


##spliter
# use --input_save_path
parser.add_argument(
    "--train_percentage",
    default=0.7,
    type=float,
    help="Train data percentage (0 ~ 1)",
)
parser.add_argument(
    "--aug_mode",
    default="before",
    type=str,
    help="Augmentation mode (before / after). Defualt: before",
)
parser.add_argument(
    "--age",
    default=False,
    type=bool,
    help="Classification of Age? (True / False)",
)



def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
