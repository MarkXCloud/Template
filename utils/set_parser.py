import argparse


def parse_args():
    parser = argparse.ArgumentParser("Basic necessary kwargs for training and validation", add_help=False)
    # basic kwargs
    parser.add_argument('--cfg', type=str, default='', help="model.py path")
    parser.add_argument('--save_dir',type=str,default='./runs',help="checkpoint path")
    parser.add_argument('--seed', type=int, default=0, help="random seed")

    opt = parser.parse_args()
    return opt
