"""Running Hessian and input second-moment statistics used during calibration."""

import math

import torch
import torch.nn as nn
import transformers


class CalibrationStatisticsMixin:
    def _accumulate_input_statistics(self, inp):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)

        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D) or "MoEGate" in type(self.layer).__name__:
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        prev_nsamples = self.nsamples
        self.H *= prev_nsamples / (prev_nsamples + tmp)
        self.nsamples += tmp

        inp_fp = inp.float()
        h_inp = math.sqrt(2 / self.nsamples) * inp_fp
        self.H += h_inp.matmul(h_inp.t())

        current_second_moment = inp_fp.matmul(inp_fp.t())
        if self.input_second_moment is None:
            self.input_second_moment = torch.zeros_like(current_second_moment, device=self.dev)
        else:
            if self.input_second_moment.shape != current_second_moment.shape:
                self.input_second_moment = torch.zeros_like(current_second_moment, device=self.dev)
            self.input_second_moment *= prev_nsamples / self.nsamples

        self.input_second_moment += current_second_moment / self.nsamples
