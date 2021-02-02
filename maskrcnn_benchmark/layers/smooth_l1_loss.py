# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
import torch


# TODO maybe push this to nn?
def smooth_l1_loss(input, target, beta=1. / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    if input.numel() == 0:
        return input.sum()*0.0 # make sure the loss is valid when there is no positive samples

    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()
