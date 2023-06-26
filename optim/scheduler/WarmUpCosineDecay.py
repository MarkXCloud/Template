import torch
__all__=['WarpUpCosineDecay']

class WarpUpCosineDecay(torch.optim.lr_scheduler.SequentialLR):
    def __init__(self, optimizer, start_factor: float, warmup_iter: int, T_max: int):
        super().__init__(optimizer=optimizer,
                         schedulers=[
                             torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=start_factor,
                                                               total_iters=warmup_iter),
                             torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max - warmup_iter)
                         ],
                         milestones=[warmup_iter])
