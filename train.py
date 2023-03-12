import torch
from utils import parse_args
from data import load_dataset
from utils import load_optimizer

def main():
    cfg = parse_args()
    # train_loader,test_loader,classes = load_dataset(cfg=cfg)
    #
    # for x,y in train_loader:
    #     print(x.shape,y.shape)
    #     exit(1)
    # a = torch.nn.Linear(in_features=3,out_features=5)
    # opt = load_optimizer(model_params=a.parameters(),cfg=cfg)
    # print(opt)

if __name__ == '__main__':
    main()