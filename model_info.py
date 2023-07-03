import torch
from torchinfo import summary
from template import parse_args
from template.util import load_module
from ptflops import get_model_complexity_info

if __name__ == '__main__':
    args = parse_args()
    module_loader = load_module(args.config)
    model = module_loader.model
    img_size: tuple = module_loader.img_size
    input_size = (args.batch_per_gpu,) + img_size
    del module_loader

    with torch.cuda.device(0):
        summary(model, input_size)
        macs, params = get_model_complexity_info(model, img_size, as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
