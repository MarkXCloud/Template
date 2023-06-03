import torch
from torchinfo import summary
from utils import parse_args,load_module
from ptflops import get_model_complexity_info


if __name__=='__main__':
    config = parse_args()
    module_loader = load_module(config.cfg)
    model = module_loader.model
    input_size: tuple = module_loader.input_size
    img_size: tuple = module_loader.img_size
    del module_loader

    with torch.cuda.device(0):
        summary(model,input_size)
        macs, params = get_model_complexity_info(model, img_size, as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))