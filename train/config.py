# Configuration file for (base training / adversarial training / adversarial attack)

import argparse

# Configuration
parser = argparse.ArgumentParser(description="Base Training")

parser.add_argument(
    "--genres",
    default=["Classical", "Rock", "Country", "GameMusic"],
    type=list,
    help="Genres",
)
parser.add_argument(
    "--genre_datanum", default=300, type=int, help="The number of data per each genre."
)
parser.add_argument(
    "--model_name", type=str, default="ResNet", help="Prefix of model name."
)

# base input
parser.add_argument(
    "--train_input_path",
    type=str,
    default="/data/midi820_400/train/",
    help="Train input directory.",
)
parser.add_argument(
    "--valid_input_path",
    type=str,
    default="/data/midi820_400/valid/",
    help="Valid input directory.",
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

parser.add_argument("--gpu", default="0", type=str, help="GPU id to use.")

parser.add_argument(
    "--learning_rate", default=0.00005, type=float, help="Model learning rate."
)


##shared parameters (train & attack)
parser.add_argument(
    "--mode",
    default="basetrain",
    type=str,
    help="Mode (basetrain / advtrain / attack)",
)
parser.add_argument(
    "--model_save_path", default="/data/drum/model/", type=str, help="Model saving path"
)
parser.add_argument(
    "--input_shape", default=(129, 400, 128), type=tuple, help="Input shape."
)
parser.add_argument(
    "--trainloader_save_path",
    default="/data/drum/dataset/train/",
    type=str,
    help="Train loader saving path",
)
parser.add_argument(
    "--validloader_save_path",
    default="/data/drum/dataset/valid/",
    type=str,
    help="Valid loader saving path",
)
parser.add_argument(
    "--valid_filename_save_path",
    default="/data/drum/dataset/val_filename/",
    type=str,
    help="Valid loader file name saving path",
)


##attack parameters

# 1. basic configurations
# parser.add_argument("--attack_mode", default="base", type=str, help="Attack Mode (base / trained)") # full name of .pt will tell anyway
parser.add_argument("--targeted", default=False, type=bool, help="is Targeted?")
parser.add_argument(
    "--attack_type", default="fgsm", type=str, help="attack (fgsm / deepfool / random)"
)
parser.add_argument(
    "--model_fname",
    default="Res50_valloss_0.8801_acc_81.25.pt",
    type=str,
    help="model file (ex: Res50_valloss_0.8801_acc_81.25.pt)",
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
parser.add_argument("--plot", default=True, type=bool, help="draw plot?")
parser.add_argument(
    "--tempo_range", default=1, type=int, help="+/-n tempo cells to attack"
)


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
