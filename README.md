# A template codebase for various deep learning paradigms

This repo is a template codebase for various deep learning trainging paradigms, such as image classification, object
detection and so on. 

It is very **tedious** to write a deep learning codebase with Trainer, Logger or other functional modules to determine a training process by ourselves. Therefore,  we sometimes choose to learn some modules written by other people so that we can determine a training process quickly with various functions, like pytorch-lightning, transformers and so on. But, the short-coming of them is that it is difficult to change some details and design some special process of our own because their codes are written too complicatedly to step into to insert any changes. So, I want to write some simple and neat template codes with full functions of deep learning training paradigm, while users can easily change any detail to meet their wants.

**Template** can work with various functions necessary in a training paradigm:

1. Custom model definition, supported by [timm](https://github.com/huggingface/pytorch-image-models).
2. Custom dataset definition, rawly supported by `torch.utils.data.Dataset`, with [albumentations](https://albumentations.ai/docs/) for data augmentation.
3. Config file in `.py`, where you can import everything in raw python code, which is much more readable and rewritable than other types of config. Then you can just assign it in terminal.  This function is performed using `importlib`.
4. Distributed training, supported by [accelerate](https://huggingface.co/docs/accelerate/index). With some useful features of accelerate we can easily launch distributed training with small changes of code, while avoid some tedious changes including `sampler.set_epoch()` (used to set random seed every epoch in ddp), `model_saved=ddp_model.module` (because primitive model is wrapped for ddp).
5. [WandB](https://wandb.ai/site) for logging the info of everything at first, and tracing the loss or other metrics curve.
6. Evaluation supported by [evaluate](https://huggingface.co/docs/evaluate/index). Initiate the evaluator in config, call it in test loop to record predictions, and finally compute all the metrics. If you want to use a custom metric you should rewrite `add_batch()` method and `compute()`method.
7. Saver to save the latest model with custom interval and the best model with specific metric.
8. Model #params and #MACS supported by [torchinfo](https://github.com/TylerYep/torchinfo) and [ptflops](https://github.com/LukasHedegaard/ptflops).
9. All the codes are very simple and neat to make you easy to change everywhere for custom function. If you want to define other modules like loss or scheduler, you can just create a python file, write it and import it in you config.

# Requirements

- torch
- torchvision
- timm
- tqdm
- accelerate
- evaluate
- albumentations
- torchinfo
- ptflops

# Training

To train a model directly:

```sh
accelerate launch train.py configs/res50_cifar10.py
```

Then you can find your run under `runs/res50_cifar10/$local_time$/` with your config file and `.pt` weights.

To use distributed training:

```sh
accelerate launch --multi_gpu train.py configs/res50_cifar10.py
```

Other shell configurations please refer [accelerate](https://huggingface.co/docs/accelerate/index).

# Calculate parameters and MACS

To show the #params and MACS of your model:

```sh
python model_info.py configs/res50_cifar10.py
```



# Custom modules

The repo structure is listed as followed:

```
configs/
	|-config_file.py # your config file
data/
	|-dataset.py # your dataset 
evaluation/
	|-(nothing) # your evaluation metrics
model/
	|-(nothing) # your model
runs/ # restore the weights and configs of one training run
template/
	|-launcher.py # main functions in it
	|-Paradigm.py # different training task paradigm
	|-set_parser.py # parser
	|-visualizer.py # nothing, to be used soon
hyp_search.py # for hyper parameter search, haven't been finishied
model_info.py # show #params and calculate MACS
predict.py # perform prediction, to be used soon
train.py # launch training
validation.py # launch validation, to be used soon
```

No matter what modules you want to customize, you can just write it in the corresponding file and import it in your own config.

If you want to perform a new task or paradigm of some deep learning tasks, you should define a new paradigm in `Paradigm.py` and import it in your config.

# Acknowledgement
Thanks to every wonderful module used in this repo. Your effort helps me to finish all parts of the repo in an easier way.