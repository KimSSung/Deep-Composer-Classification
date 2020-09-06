# Configuration file for (base training / adversarial training / adversarial attack)

import argparse

# Basic Configuration
parser = argparse.ArgumentParser(description="Base Training")


##########################PATH MERGE##########################
# TODO: merge all the paths as one

# path
parser.add_argument(
    "--load_path",
    type=str,
    default="/load/parent/folder/path/",
    help="Path of load parent folder.",
)
parser.add_argument(
    "--save_path",
    type=str,
    default="/save/parent/folder/path/",
    help="Path of save parent folder.",
)
parser.add_argument(
    "--model_save_path", default="/data/drum/model/", type=str, help="Model saving path"
)
parser.add_argument(
    "--save_atk",
    default="False",
    type=str,
    help="save adversarial examples? (True / False)",
)

## split path
# ====> --load_path '/data/inputs_full/'
# ====> --save_path '/data/split/' => use save_path + [train.txt | valid.txt]

## trainer
# basetrain
# ====> --load_path '/data/split/'

# save loader
# ====> --save_path '/data/drum/dataset/' => use save_path + [train/ | valid/]


##generator
# ====> --load_path '/data/MAESTRO/maestro-v2.0.0/'
# ====> --save_path '/data/which_dir/'

## attacker
# ===> --load_path '/data/drum/bestmodel/' => use --load_path + [model/ | dataset/]
# ===> --save_path '/data/attacks/'

##converter
# ====> --load_path "/data/attacks/08-25-00-00/ep0.6/"

# parser.add_argument(
#     "--to_convert_path",
#     default="/data/attacks/",
#     type=str,
#     help="Path for 'only MIDIs' to convert. Path CAN contain any folder !!",
# )

## attacked input
# ====> --save_path '/data/attacks/vel_deepfool/' => use save_path + [train/ | valid/]

# parser.add_argument(
#     "--attacked_train_input_path",
#     type=str,
#     default="/data/attacks/vel_deepfool/train/",
#     help="Attacked Train input directory.",
# )
# parser.add_argument(
#     "--attacked_valid_input_path",
#     type=str,
#     default="/data/attacks/vel_deepfool/valid/",
#     help="Attacked Valid input directory.",
# )


##########################PATH MERGE##########################

parser.add_argument(
    "--mode",
    default="foo",  # force to input mode ^_^
    type=str,
    help="Mode (basetrain / advtrain / attack / generate / convert / split)",
)
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
    "--optim",
    type=str,
    default="SGD",
    help="Optimizer [Adadelta, Adagrad, Adam, AdamW, SparseAdam, Adamax, ASGD, RMSprop, Rprop, SGD, Nesterov]",
)
parser.add_argument(
    "--transform", type=str, default=None, help="Transform mode [Transpose / Tempo]",
)
parser.add_argument(
    "--epochs",
    default=100,
    type=int,
    help="Total number of epochs to run. Not actual epoch.",
)
parser.add_argument(
    "--seg_num", default=20, type=int, help="segment number for each midi#"
)
parser.add_argument(
    "--train_batch", default=40, type=int, help="Batch size for training"
)
parser.add_argument("--valid_batch", default=40, type=int, help="Batch size for valid.")

parser.add_argument("--gpu", default="0,1,2,3", type=str, help="GPU id to use.")

parser.add_argument("--lr", default=0.01, type=float, help="Model learning rate.")


parser.add_argument(
    "--save_trn", type=bool, default=True, help="Save both model & loader?"
)


##attack parameters

# 1. basic configurations
# parser.add_argument("--attack_mode", default="base", type=str, help="Attack Mode (base / trained)") # full name of .pt will tell anyway
parser.add_argument("--targeted", default=False, type=bool, help="is Targeted?")
parser.add_argument(
    "--attack_type", default="fgsm", type=str, help="attack (fgsm / deepfool / random)"
)
# 2. data related
parser.add_argument(
    "--orig",
    default=True,
    type=bool,
    help="attack on original dataset? (default: True)",
)
# 3. specific attack related
parser.add_argument(
    "--epsilons",
    default="0.0",
    type=str,
    help="list of epsilons 'ep0, ep1, ep2..' seperated by ,",
)
parser.add_argument(
    "--max_iter", default=10, type=int, help="max iterations for deepfool attack",
)
parser.add_argument(
    "--overshoot", default=5, type=int, help="overshoot for deepfool attack"
)
parser.add_argument("--plot", default=False, type=bool, help="draw plot?")
parser.add_argument(
    "--confusion", default=False, type=bool, help="draw confusion matrix?"
)
# parser.add_argument(
#     "--tempo_range", default=1, type=int, help="+/-n tempo cells to attack"
# )


##spliter
# use --input_save_path
parser.add_argument(
    "--train_percentage", default=0.7, type=float, help="Train data percentage (0 ~ 1)",
)
parser.add_argument(
    "--omit", default=None, type=str, help="List of omitted composers' indices.",
)
parser.add_argument(
    "--age", default=False, type=bool, help="Classification of Age? (True / False)",
)


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
