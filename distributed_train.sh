#!/bin/sh
CUDA_VISIBLE_DEVICES="0,1,2,3"  accelerate launch --multi_gpu --mixed_precision=fp16 main.py train configs/test_cfg.py