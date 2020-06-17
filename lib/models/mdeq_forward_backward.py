# Modified based on the DEQ repo.

import torch
from torch import nn
import torch.nn.functional as functional
from torch.autograd import Function
import numpy as np

import sys
sys.path.append("../../")
from modules.deq2d import *

__author__ = "shaojieb"


class MDEQWrapper(DEQModule2d):
    def __init__(self, func, func_copy):
        super(MDEQWrapper, self).__init__(func, func_copy)
    
    def forward(self, z1, u, **kwargs):
        train_step = kwargs.get('train_step', -1)
        threshold = kwargs.get('threshold', 30)
        writer = kwargs.get('writer', None)

        if u is None:
            raise ValueError("Input injection is required.")

        new_z1 = list(DEQFunc2d.apply(self.func, z1, u, threshold, train_step, writer))
        cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in new_z1]
        if self.training:
            new_z1 = DEQFunc2d.list2vec(DEQFunc2d.f(self.func, new_z1, u, threshold, train_step))
            new_z1 = self.Backward.apply(self.func_copy, new_z1, u, threshold, train_step, writer)
            new_z1 = DEQFunc2d.vec2list(new_z1, cutoffs)
        return new_z1

