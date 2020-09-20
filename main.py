from config import get_config

import numpy as np
import torch
import torch.backends.cudnn
import random
import os

from trainer import Trainer
from attacker import Attacker
from generator import Generator
from converter import Converter
from spliter import Spliter

from datetime import date, datetime

# seed
def set_seed(seed):
    random.seed(seed)  # python random module
    np.random.seed(seed)  # np module
    torch.manual_seed(seed)  # for both CPU & GPU
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    return


def main(args, save_dir):

    set_seed(333)

    # mode: [basetrain / advtrain / attack / generate]
    if args.mode == "basetrain" or args.mode == "advtrain":
        net = Trainer(args, save_dir)
        net.train(args.mode)

    elif args.mode == "attack":
        net = Attacker(args)
        net.run()

    elif args.mode == "generate":
        net = Generator(args)
        net.run()

    elif args.mode == "convert":
        net = Converter(args)
        net.run()

    elif args.mode == "split":
        net = Spliter(args)
        net.run()

    else:
        print('#### Please execute with mode: ex) --mode "basetrain"')
        print(
            "#### MODE: [basetrain / advtrain / attack / generate / converter / split]"
        )


if __name__ == "__main__":

    config, unparsed = get_config()
    run_time = date.today().strftime("%m-%d") + datetime.now().strftime("-%H-%M")
    save_dir = config.save_path + run_time + "/"
    # store configuration
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir +  "config.txt", "w") as f:
        f.write("Parameters for " + config.mode + ":\n\n")
        for arg in vars(config):
            argname = arg
            contents = str(getattr(config, arg))
            # print(argname + ' = ' + contents)
            f.write(argname + " = " + contents + "\n")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

    main(config, save_dir)
