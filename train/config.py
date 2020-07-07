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

parser.add_argument(
    "--input_shape", default=(129, 400, 128), type=tuple, help="Input shape."
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
    "--model_name", type=str, default="ResNet", help="Prefix of model name."
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
    "--model_save_path", default="/data/drum/model/", type=str, help="Model saving path"
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

parser.add_argument(
    "--learning_rate", default=0.00005, type=float, help="Model learning rate."
)


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
