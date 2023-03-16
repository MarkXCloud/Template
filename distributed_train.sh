#!/bin/sh
CUDA_VISIBLE_DEVICES="0,7"  accelerate launch --multi_gpu --mixed_precision=fp16 train.py --cfg model/hub/test_cfg.py