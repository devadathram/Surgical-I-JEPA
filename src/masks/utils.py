# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Modified by Devadath Ram for Surgical Cholec80 Dataset.

import torch


def apply_masks(x, masks):
    """
    :param x: tensor of shape [B, N, D]
    :param masks: list of tensors containing indices
    """
    all_x = []
    for m in masks:
        # If m is [B, K], make it [B, K, D]
        if m.dim() == 2:
            m = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        
        # Ensure indices are valid (clamp any -1 padding if it exists)
        mask_keep = torch.clamp(m, min=0)
        
        # Gather along the token dimension
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
        
    return torch.cat(all_x, dim=0)
