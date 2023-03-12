import argparse


def parse_args():
    parser = argparse.ArgumentParser("Basic necessary kwargs for training and validation", add_help=False)
    # basic kwargs
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--model', type=str, default='', help="model.yaml path")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")

    parser.add_argument("--data", type=str, required=True, help="dataset cfg")
    parser.add_argument("--num_workers", type=int, default=8, help='"maximum number of dataloader workers"')
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    parser.add_argument("--opt",type=str,default='utils/hyp/Adam.yaml',help="optimizer.yaml path")
    # distributed training
    parser.add_argument('--dist', type=bool, default=False, help='"use distributed training"')
    parser.add_argument('--gpus', type=str, nargs='+', help='"gpu ids"')
    parser.add_argument("--local_rank", type=int, default=-1, help='"local rank for DistributedDataParallel"')
    opt = parser.parse_args()
    return opt
