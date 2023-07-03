import argparse


def parse_args():
    parser = argparse.ArgumentParser("Basic necessary kwargs for training", add_help=False)
    # basic kwargs
    parser.add_argument('config', type=str, default='', help="model.yaml path")
    parser.add_argument('--seed', type=int, default=3407, help="random seed")

    parser.add_argument('-e','--epoch',type=int,default=24,help="training epoch")

    parser.add_argument('--batch-per-gpu',type=int,default=64,help="batch size per gpu for dataloader")
    parser.add_argument('--num-workers',type=int,default=4,help="num_workers of dataloader")
    parser.add_argument('--grad-step',type=int,default=2,help="gradient accumulation steps")

    parser.add_argument('--compile', action='store_true', help="to compile the model")

    parser.add_argument('--ckpt',type=str,default='',help="checkpoint path")

    parser.add_argument('--wandb',action='store_true',help="use wandb logger")
    parser.add_argument('--tb',action='store_true',help="use tensorboardx logger")

    opt = parser.parse_args()
    return opt

def parse_args_val():
    parser = argparse.ArgumentParser("Validation necessary kwargs for validation", add_help=False)
    # basic kwargs
    parser.add_argument('config', type=str, default='', help="model.yaml path")
    parser.add_argument('--seed', type=int, default=3407, help="random seed")

    parser.add_argument('--batch-size',type=int,default=64,help="batch_per_gpu of dataloader")
    parser.add_argument('--num-workers',type=int,default=4,help="num_workers of dataloader")

    parser.add_argument('-l','--load-from',type=str,required=True,help="path to the model weights")
    parser.add_argument('--compile',action='store_true',help="to compile the model")

    opt = parser.parse_args()
    return opt
