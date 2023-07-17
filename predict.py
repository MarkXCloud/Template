import torch
from template.set_parser import val_args
import template.util as util
import albumentations as A
from albumentations.pytorch import ToTensorV2
from template.visualizer import ImageClassificationVisualizer
import cv2


if __name__=='__main__':
    infer_args = val_args()
    infer_args.add_argument('--img',required=True,help="path to the image")
    args = infer_args.parse_args()

    raw_img = cv2.imread(args.img,cv2.IMREAD_COLOR)

    module_loader = util.load_module(args.config)

    model = module_loader.model
    model.load_state_dict(torch.load(args.load_from, map_location="cpu"))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    vis = ImageClassificationVisualizer()

    # preprocess
    image = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    transform = A.Compose([
        A.Resize(module_loader.img_size[-2], module_loader.img_size[-1]),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()])
    image = transform(image=image)['image']
    image = torch.unsqueeze(image,0)
    image = image.to(device)
    with torch.no_grad():
        output = model(image)

    #post process
    out_idx = torch.argmax(output,-1)
    out_idx = torch.squeeze(out_idx,0).cpu().item()
    out_class = module_loader.classes[out_idx]

    vis.draw(image=raw_img,label=out_class)











