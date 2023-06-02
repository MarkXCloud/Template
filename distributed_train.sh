#!/bin/sh
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  accelerate launch --multi_gpu --mixed_precision=fp16 tools/train.py --cfg configs/test_cfg.py