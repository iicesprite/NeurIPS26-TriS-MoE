import torch
from utils.autosearch import structural_searching_multip_alternating_group_rc

import logging
logger = logging.getLogger()

'''
Used to generate masks for minor structural 2-bit salient data and split major 1-bit normal data according to different metric.
'''
def structural_guassian_distribution_multip_alternating_group_x(tmp, H=None, up_lim=30, num_p=1, inp=None):
    target_weights = tmp ** 2 / (torch.diag(H).reshape((1, -1))) ** 2
    optimal_split_list,mask_list = structural_searching_multip_alternating_group_rc(target_weights, up_lim, num_p, inp)
    # print(mask1.sum() / mask1.numel(), mask2.sum() / mask2.numel(), mask3.sum() / mask3.numel())
    mask_ratio = []
    for i in range(len(mask_list)):
        mask_ratio.append(mask_list[i].sum() / mask_list[i].numel())

    return mask_list
