from numpy import mean
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
index = 0
@torch.no_grad()
def part_mean(tensor, op='-'):
    non_zero = tensor*(tensor!=0)

    mean_val = non_zero.mean(-1).view(-1, 1)

    return mean_val

@torch.no_grad()
def high_order_residual(x, mask, order=2):
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        masked_x_tensor -= mean_tensor_all[:, None]
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(torch.isnan(scale_tensor_all), torch.zeros_like(scale_tensor_all), scale_tensor_all)

        binary= torch.sign(masked_x_tensor)
        binary *= scale_tensor_all[:, None]
        binary += mean_tensor_all[:, None]
        sum_order = sum_order + binary*mask
    
    return sum_order

@torch.no_grad()
def high_order_residual_rc(x, mask, order=2):
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        # mean row
        mean_tensor_all_r = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all_r = torch.where(torch.isnan(mean_tensor_all_r), torch.zeros_like(mean_tensor_all_r), mean_tensor_all_r)
        masked_x_tensor -= mean_tensor_all_r[:, None]
        # mean column
        mean_tensor_all_c = torch.nanmean(masked_x_tensor, dim=0)
        mean_tensor_all_c = torch.where(torch.isnan(mean_tensor_all_c), torch.zeros_like(mean_tensor_all_c), mean_tensor_all_c)
        masked_x_tensor -= mean_tensor_all_c[None, :]

        # alpha row
        scale_tensor_all_r = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all_r = torch.where(torch.isnan(scale_tensor_all_r), torch.zeros_like(scale_tensor_all_r), scale_tensor_all_r)
        # alpha column
        scale_tensor_all_c = torch.nanmean(torch.abs(masked_x_tensor / scale_tensor_all_r[:, None]), dim=0)
        scale_tensor_all_c = torch.where(torch.isnan(scale_tensor_all_c), torch.zeros_like(scale_tensor_all_c), scale_tensor_all_c)

        binary= torch.sign(masked_x_tensor)
        binary *= scale_tensor_all_r[:, None]
        binary *= scale_tensor_all_c[None, :]
        binary += mean_tensor_all_r[:, None] + mean_tensor_all_c[None, :]
        sum_order = sum_order + binary*mask

    return sum_order

@torch.no_grad()
def high_order_residual_alternating_order1(x, mask, order=2, iter=15):
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        masked_x_tensor -= mean_tensor_all[:, None]
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(torch.isnan(scale_tensor_all), torch.zeros_like(scale_tensor_all), scale_tensor_all)

        binary= torch.sign(masked_x_tensor)
        new_binary = binary.clone()
        binary *= scale_tensor_all[:, None]
        binary += mean_tensor_all[:, None]
        sum_order = sum_order + binary*mask

    # Alternating update
    refine_mean = mean_tensor_all.clone()
    sum_order_alternating = sum_order.clone()

    for k in range(iter):
        # 1. Fix alpha and B, update mean
        residual = new_matrix - sum_order_alternating
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        refine_mean += mean_tensor_all.clone()
        
        # 2. Fix mean and B, update alpha
        new_alpha = 1. / (torch.sum(new_binary * mask * new_binary * mask, dim=1) + 1e-8) * torch.sum(new_binary * mask * (new_matrix - refine_mean[:, None] * mask), dim=1)
        
        # 3. Fix mean and alpha, update B
        new_binary = torch.sign(new_matrix - refine_mean[:, None] * mask)

        # Final refine results
        sum_order_alternating = torch.zeros_like(x) + (new_alpha[:, None] * new_binary + refine_mean[:, None]) * mask


    return sum_order_alternating

@torch.no_grad()
def high_order_residual_alternating_order1_x(x, mask, order=2, S=None, iter=15, iter2=15):
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        masked_x_tensor -= mean_tensor_all[:, None]
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(torch.isnan(scale_tensor_all), torch.zeros_like(scale_tensor_all), scale_tensor_all)

        binary= torch.sign(masked_x_tensor)
        new_binary = binary.clone()
        binary *= scale_tensor_all[:, None]
        binary += mean_tensor_all[:, None]
        sum_order = sum_order + binary*mask

    # Alternating update
    refine_mean = mean_tensor_all.clone()
    sum_order_alternating = sum_order.clone()
    new_alpha = scale_tensor_all.clone()

    for k in range(iter):
        # 1. Fix alpha and B, update mean
        residual = new_matrix - sum_order_alternating
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        refine_mean += mean_tensor_all.clone()
        
        # 2. Fix mean and B, update alpha
        new_alpha = 1. / (torch.sum(new_binary * mask * new_binary * mask, dim=1) + 1e-8) * torch.sum(new_binary * mask * (new_matrix - refine_mean[:, None] * mask), dim=1)
        
        # 3. Fix mean and alpha, update B
        new_binary = torch.sign(new_matrix - refine_mean[:, None] * mask)

        # Final refine results
        sum_order_alternating = torch.zeros_like(x) + (new_alpha[:, None] * new_binary + refine_mean[:, None]) * mask

    MM = mask[:, :, None] * mask[:, None, :]
    refine_mean_den = torch.sum(S * MM, dim=(1,2), dtype=torch.float32) + 1e-10
    masked_B = new_binary * mask
    new_alpha_den = torch.sum(S * masked_B[:, :, None] * masked_B[:, None, :], dim=(1,2)) + 1e-10
    # diag_S = torch.diag(S)
    for kk in range(iter2):
        # X error update mean
        refine_mean = torch.sum(S * (new_matrix - new_alpha[:, None] * new_binary * mask)[:, :, None] * MM, dim=(1,2)) / refine_mean_den

        # X error update alpha
        new_alpha = torch.sum(S * masked_B[:, :, None] * (new_matrix - refine_mean[:, None] * mask)[:, None, :], dim=(1,2)) / new_alpha_den

    sum_order_alternating = torch.zeros_like(x) + (new_alpha[:, None] * new_binary + refine_mean[:, None]) * mask

    return sum_order_alternating

@torch.no_grad()
def high_order_residual_alternating_order2_rc_nomean(x, mask, order=2, iter=15):
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    binary_list = []
    alpha_list_r = []
    alpha_list_c = []
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        # alpha row
        scale_tensor_all_r = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all_r = torch.where(torch.isnan(scale_tensor_all_r), torch.zeros_like(scale_tensor_all_r), scale_tensor_all_r)
        alpha_list_r.append(scale_tensor_all_r.clone())
        # alpha column
        scale_tensor_all_c = torch.nanmean(torch.abs(masked_x_tensor / scale_tensor_all_r[:, None]), dim=0)
        scale_tensor_all_c = torch.where(torch.isnan(scale_tensor_all_c), torch.zeros_like(scale_tensor_all_c), scale_tensor_all_c)
        alpha_list_c.append(scale_tensor_all_c.clone())

        binary= torch.sign(masked_x_tensor)
        binary_list.append(binary.clone())
        binary *= scale_tensor_all_r[:, None]
        binary *= scale_tensor_all_c[None, :]
        sum_order = sum_order + binary*mask

    # Alternating update
    sum_order_alternating = sum_order.clone()

    for k in range(iter):        
        # 2-1. Fix mean, alpha column, and B, update alpha row 0
        W_tilde = new_matrix - (alpha_list_c[1][None, :] * alpha_list_r[1][:, None] * binary_list[1]) * mask
        alpha_c_B = alpha_list_c[0][None, :] * binary_list[0] * mask
        alpha_list_r[0] = torch.sum(alpha_c_B * W_tilde, dim=1) / (torch.sum(alpha_c_B * alpha_c_B, dim=1) + 1e-8)
        
        # 2-2. Fix mean, alpha row, and B, update alpha column 0
        alpha_r_B =  alpha_list_r[0][:, None] * binary_list[0] * mask
        alpha_list_c[0] = torch.sum(alpha_r_B * W_tilde, dim=0) / (torch.sum(alpha_r_B * alpha_r_B, dim=0) + 1e-8)

        # 2-3. Fix mean, alpha column, and B, update alpha row 1
        W_tilde = new_matrix - (alpha_list_c[0][None, :] * alpha_list_r[0][:, None] * binary_list[0]) * mask
        alpha_c_B = alpha_list_c[1][None, :] * binary_list[1] * mask
        alpha_list_r[1] = torch.sum(alpha_c_B * W_tilde, dim=1) / (torch.sum(alpha_c_B * alpha_c_B, dim=1) + 1e-8)
        
        # 2-4. Fix mean, alpha row, and B, update alpha column 1
        alpha_r_B =  alpha_list_r[1][:, None] * binary_list[1] * mask
        alpha_list_c[1] = torch.sum(alpha_r_B * W_tilde, dim=0) / (torch.sum(alpha_r_B * alpha_r_B, dim=0) + 1e-8)

        # 3. Fix mean and alpha, update B
        new_matrix_expanded = new_matrix.unsqueeze(-1)
        comb0 = alpha_list_r[0].reshape(-1, 1) @ alpha_list_c[0].reshape(1, -1)
        comb1 = alpha_list_r[1].reshape(-1, 1) @ alpha_list_c[1].reshape(1, -1)
        v = torch.stack([-comb0 - comb1, -comb0 + comb1, 
                    comb0 - comb1, comb0 + comb1], dim=2)

        min_indices = torch.argmin(torch.abs(new_matrix_expanded - v), dim=-1)

        binary_list[0] = torch.ones_like(min_indices)
        binary_list[0][(min_indices == 0) | (min_indices == 1)] = -1
        binary_list[1] = torch.ones_like(min_indices)
        binary_list[1][(min_indices == 0) | (min_indices == 2)] = -1 

        # Final refine results
        sum_order_alternating = torch.zeros_like(x) + (alpha_list_c[0][None, :] * alpha_list_r[0][:, None] * binary_list[0] + alpha_list_c[1][None, :] * alpha_list_r[1][:, None] * binary_list[1]) * mask

    return sum_order_alternating, alpha_list_c, alpha_list_r, binary_list

@torch.no_grad()
def high_order_residual_alternating_order1_rc_nomean(x, mask, order=2, iter=15):
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        # alpha row
        scale_tensor_all_r = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all_r = torch.where(torch.isnan(scale_tensor_all_r), torch.zeros_like(scale_tensor_all_r), scale_tensor_all_r)
        # alpha column
        scale_tensor_all_c = torch.nanmean(torch.abs(masked_x_tensor / scale_tensor_all_r[:, None]), dim=0)
        scale_tensor_all_c = torch.where(torch.isnan(scale_tensor_all_c), torch.zeros_like(scale_tensor_all_c), scale_tensor_all_c)

        binary= torch.sign(masked_x_tensor)
        new_binary = binary.clone()
        binary *= scale_tensor_all_r[:, None]
        binary *= scale_tensor_all_c[None, :]
        sum_order = sum_order + binary*mask

    # Alternating update
    sum_order_alternating = sum_order.clone()
    new_alpha_r = scale_tensor_all_r.clone()
    new_alpha_c = scale_tensor_all_c.clone()
    for k in range(iter):        
        # 1-1. Fix mean, alpha column, and B, update alpha row
        alpha_c_B = new_alpha_c[None, :] * new_binary * mask
        new_alpha_r = torch.sum(alpha_c_B * new_matrix, dim=1) / (torch.sum(alpha_c_B * alpha_c_B, dim=1) + 1e-8)
        
        # 1-2. Fix mean, alpha row, and B, update alpha column
        alpha_r_B = new_alpha_r[:, None] * new_binary * mask
        new_alpha_c = torch.sum(alpha_r_B * new_matrix, dim=0) / (torch.sum(alpha_r_B * alpha_r_B, dim=0) + 1e-8)

        # Final refine results
        sum_order_alternating = torch.zeros_like(x) + new_alpha_c[None, :] * new_alpha_r[:, None] * new_binary * mask
    alpha_list_c=[new_alpha_c]
    alpha_list_r=[new_alpha_r]
    binary_list=[new_binary]
    return sum_order_alternating, alpha_list_c, alpha_list_r, binary_list

@torch.no_grad()
def high_order_residual_alternating_mean(x, mask, order=2, num_iters=15):
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    binary_list = []
    alpha_list = []
    refine_mean = torch.zeros(x.shape[0], device=x.device)
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        refine_mean += mean_tensor_all.clone()
        masked_x_tensor -= mean_tensor_all[:, None]
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(torch.isnan(scale_tensor_all), torch.zeros_like(scale_tensor_all), scale_tensor_all)
        alpha_list.append(scale_tensor_all.clone())

        binary = torch.sign(masked_x_tensor)
        binary_list.append(binary.clone())
        binary *= scale_tensor_all[:, None]
        binary += mean_tensor_all[:, None]
        sum_order = sum_order + binary*mask

    new_matrix = x.clone() * mask
    sum_order_alternating = sum_order.clone()
    
    for k in range(num_iters):
        # 1. Fix alpha1, alpha2, B1, and B2, update mean
        residual = new_matrix - sum_order_alternating
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        refine_mean += mean_tensor_all.clone()

        # 2. Fix mean, B1, and B2, update alpha1 and alpha2
        alpha_list[0] = 1. / (torch.sum(binary_list[0] * mask * binary_list[0] * mask, dim=1) + 1e-8) * torch.sum(binary_list[0] * mask * (new_matrix - refine_mean[:, None] * mask - alpha_list[1][:, None] * binary_list[1] * mask), dim=1)
        alpha_list[1] = 1. / (torch.sum(binary_list[1] * mask * binary_list[1] * mask, dim=1) + 1e-8) * torch.sum(binary_list[1] * mask * (new_matrix - refine_mean[:, None] * mask - alpha_list[0][:, None] * binary_list[0] * mask), dim=1)

        # 3. Fix mean, alpha1, and alpha2, update B1 and B2
        new_matrix_expanded = (new_matrix - refine_mean[:, None] * mask).unsqueeze(-1)
        v = torch.stack([-alpha_list[0] - alpha_list[1], -alpha_list[0] + alpha_list[1], 
                    alpha_list[0] - alpha_list[1], alpha_list[0] + alpha_list[1]], dim=1).unsqueeze(1)

        min_indices = torch.argmin(torch.abs(new_matrix_expanded - v), dim=-1)

        binary_list[0] = torch.ones_like(min_indices)
        binary_list[0][(min_indices == 0) | (min_indices == 1)] = -1
        binary_list[1] = torch.ones_like(min_indices)
        binary_list[1][(min_indices == 0) | (min_indices == 2)] = -1 

        sum_order_alternating = torch.zeros_like(x) + (alpha_list[0][:, None] * binary_list[0] + alpha_list[1][:, None] * binary_list[1] + refine_mean[:, None]) * mask

    return sum_order_alternating

@torch.no_grad()
def high_order_residual_alternating_mean_x(x, mask, order=2, S=None, num_iters=15, iter2=15):
    sum_order = torch.zeros_like(x)
    new_matrix = x.clone()
    new_matrix = new_matrix * mask
    global index
    index += 1
    binary_list = []
    alpha_list = []
    refine_mean = torch.zeros(x.shape[0], device=x.device)
    for od in range(order):
        residual = new_matrix - sum_order
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))

        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        refine_mean += mean_tensor_all.clone()
        masked_x_tensor -= mean_tensor_all[:, None]
        scale_tensor_all = torch.nanmean(torch.abs(masked_x_tensor), dim=1)
        scale_tensor_all = torch.where(torch.isnan(scale_tensor_all), torch.zeros_like(scale_tensor_all), scale_tensor_all)
        alpha_list.append(scale_tensor_all.clone())

        binary = torch.sign(masked_x_tensor)
        binary_list.append(binary.clone())
        binary *= scale_tensor_all[:, None]
        binary += mean_tensor_all[:, None]
        sum_order = sum_order + binary*mask

    new_matrix = x.clone() * mask
    sum_order_alternating = sum_order.clone()
    
    for k in range(num_iters):
        # 1. Fix alpha1, alpha2, B1, and B2, update mean
        residual = new_matrix - sum_order_alternating
        masked_x_tensor = torch.where(mask, residual, torch.tensor(float('nan')))
        mean_tensor_all = torch.nanmean(masked_x_tensor, dim=1)
        mean_tensor_all = torch.where(torch.isnan(mean_tensor_all), torch.zeros_like(mean_tensor_all), mean_tensor_all)
        refine_mean += mean_tensor_all.clone()

        # 2. Fix mean, B1, and B2, update alpha1 and alpha2
        alpha_list[0] = 1. / (torch.sum(binary_list[0] * mask * binary_list[0] * mask, dim=1) + 1e-8) * torch.sum(binary_list[0] * mask * (new_matrix - refine_mean[:, None] * mask - alpha_list[1][:, None] * binary_list[1] * mask), dim=1)
        alpha_list[1] = 1. / (torch.sum(binary_list[1] * mask * binary_list[1] * mask, dim=1) + 1e-8) * torch.sum(binary_list[1] * mask * (new_matrix - refine_mean[:, None] * mask - alpha_list[0][:, None] * binary_list[0] * mask), dim=1)

        # 3. Fix mean, alpha1, and alpha2, update B1 and B2
        new_matrix_expanded = (new_matrix - refine_mean[:, None] * mask).unsqueeze(-1)
        v = torch.stack([-alpha_list[0] - alpha_list[1], -alpha_list[0] + alpha_list[1], 
                    alpha_list[0] - alpha_list[1], alpha_list[0] + alpha_list[1]], dim=1).unsqueeze(1)

        min_indices = torch.argmin(torch.abs(new_matrix_expanded - v), dim=-1)

        binary_list[0] = torch.ones_like(min_indices)
        binary_list[0][(min_indices == 0) | (min_indices == 1)] = -1
        binary_list[1] = torch.ones_like(min_indices)
        binary_list[1][(min_indices == 0) | (min_indices == 2)] = -1 

        sum_order_alternating = torch.zeros_like(x) + (alpha_list[0][:, None] * binary_list[0] + alpha_list[1][:, None] * binary_list[1] + refine_mean[:, None]) * mask

    MM = mask[:, :, None] * mask[:, None, :]
    refine_mean_den = torch.sum(S * MM, dim=(1,2)) + 1e-10
    masked_B0 = binary_list[0] * mask
    new_alpha0_den = torch.sum(S * masked_B0[:, :, None] * masked_B0[:, None, :], dim=(1,2)) + 1e-10
    masked_B1 = binary_list[1] * mask
    new_alpha1_den = torch.sum(S * masked_B1[:, :, None] * masked_B1[:, None, :], dim=(1,2)) + 1e-10
    for kk in range(iter2):
        # X error update mean
        refine_mean = torch.sum(S * (new_matrix - (alpha_list[0][:, None] * binary_list[0] + alpha_list[1][:, None] * binary_list[1]) * mask)[:, :, None] * MM, dim=(1,2)) / refine_mean_den

        # X error update alpha
        masked_W_mu = new_matrix - refine_mean[:, None] * mask
        alpha_list[0] = torch.sum(S * masked_B0[:, :, None] * (masked_W_mu[:, None, :] - (alpha_list[1][:, None] * masked_B1)[:, None, :]), dim=(1,2)) / new_alpha0_den
        alpha_list[1] = torch.sum(S * masked_B1[:, :, None] * (masked_W_mu[:, None, :] - (alpha_list[0][:, None] * masked_B0)[:, None, :]), dim=(1,2)) / new_alpha1_den

    sum_order_alternating = torch.zeros_like(x) + (alpha_list[0][:, None] * binary_list[0] + alpha_list[1][:, None] * binary_list[1] + refine_mean[:, None]) * mask

    return sum_order_alternating

@torch.no_grad()
def normal_quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


class Binarization(nn.Module):
    def __init__(self, weight, groupsize=-1):
        super().__init__()
        oc,ic=weight.shape
        if groupsize==-1:
            groupsize=ic
        self.groupsize=groupsize
        self.n_groups=math.ceil(ic/groupsize)
        self.mean = 0

    def quantize(self, w, mask, order=2, groupi=0, S=None):
        if order == 2:
            w, alpha_list_c, alpha_list_r, binary_list = high_order_residual_alternating_order2_rc_nomean(w, mask, order=order)
        else:
            w, alpha_list_c, alpha_list_r, binary_list = high_order_residual_alternating_order1_rc_nomean(w, mask, order=order)  
        return w, alpha_list_c, alpha_list_r, binary_list
