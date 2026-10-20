"""Four-bit GPTQ for shared right factors and low-rank correction matrices."""

import logging

import torch
import torch.nn as nn
import transformers

from gptq.quant import Quantizer, quantize

logger = logging.getLogger()


class AuxiliaryMatrixQuantizationMixin:
    def _quantize_auxiliary_matrix(self, ns_matrix, H_matrix, blocksize=128, percdamp=0.01, groupsize=-1, actorder=False, static_groups=False):
        actorder = False

        W = ns_matrix.clone().float().to(self.dev)
        rows, columns = W.shape
        dev = W.device

        quantizer = Quantizer()
        quantizer.configure(bits=4, perchannel=True, sym=True, mse=False)

        if not quantizer.ready():
            quantizer.find_params(W, weight=True)

        g_idx = []
        scale = []
        zero = []
        now_idx = 1

        H = H_matrix.clone().float()

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(columns, device=dev)
        H[diag, diag] += damp

        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        if static_groups:
            import copy
            groups = []
            for i in range(0, columns, groupsize):
                #quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                scale.append(quantizer.scale)
                zero.append(quantizer.zero)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        for i1 in range(0, columns, blocksize):
            i2 = min(i1 + blocksize, columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)

                        if ((i1 + i) // groupsize) - now_idx == -1:
                            scale.append(quantizer.scale)
                            zero.append(quantizer.zero)
                            now_idx += 1
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        quantizer = groups[idx // groupsize]

                q = quantize(
                    w.unsqueeze(1), quantizer.scale, quantizer.zero, quantizer.maxq
                ).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
        self._synchronize_device(Q)

        groupsize = groupsize if groupsize != -1 else columns
        if static_groups and actorder:
            g_idx = [perm[i] // groupsize for i in range(columns)]
        else:
            g_idx = [i // groupsize for i in range(columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
        if actorder:
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()

        if scale == []:
            scale.append(quantizer.scale)
            zero.append(quantizer.zero)
        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)


        del W, H, Hinv, Losses, Err1, W1, Q1, Losses1, Hinv1
        if actorder:
            del perm, invperm

        import gc
        gc.collect()
        return Q, scale, zero, g_idx
