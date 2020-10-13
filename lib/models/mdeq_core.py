from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import functools

from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
sys.path.append("lib/models")
sys.path.append("lib/modules")
sys.path.append("../modules")
from optimizations import *
from deq2d import *
from mdeq_forward_backward import MDEQWrapper

BN_MOMENTUM = 0.1
DEQ_EXPAND = 5        # Don't change the value here. The value is controlled by the yaml files.
NUM_GROUPS = 4        # Don't change the value here. The value is controlled by the yaml files.
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)


def norm_diff(new, old, show_list=False):
    if show_list:
        return [(new[i] - old[i]).norm().item() for i in range(len(new))]
    return np.sqrt(sum((new[i] - old[i]).norm().item()**2 for i in range(len(new))))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout=0.0, wnorm=False):
        """
        A canonical residual block with two 3x3 convolutions and an intermediate ReLU. Corresponds to Figure 2
        in the paper.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, DEQ_EXPAND*planes, stride)
        self.gn1 = nn.GroupNorm(NUM_GROUPS, DEQ_EXPAND*planes, affine=True)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(DEQ_EXPAND*planes, planes)
        self.gn2 = nn.GroupNorm(NUM_GROUPS, planes, affine=True)
        
        self.downsample = downsample
        self.stride = stride
        
        self.gn3 = nn.GroupNorm(NUM_GROUPS, planes, affine=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.drop = VariationalHidDropout2d(dropout)
        if wnorm: self._wnorm()
    
    def _wnorm(self):
        self.conv1, self.conv1_fn = weight_norm(self.conv1, names=['weight'], dim=0)
        self.conv2, self.conv2_fn = weight_norm(self.conv2, names=['weight'], dim=0)
    
    def _reset(self, x):
        if 'conv1_fn' in self.__dict__:
            self.conv1_fn.reset(self.conv1)
        if 'conv2_fn' in self.__dict__:
            self.conv2_fn.reset(self.conv2)
        self.drop.reset_mask(x)
    
    def _copy(self, other):
        self.conv1.weight.data = other.conv1.weight.data.clone()
        self.conv2.weight.data = other.conv2.weight.data.clone()
        self.drop.mask = other.drop.mask.clone()
        if self.downsample:
            assert False, "Shouldn't be here. Check again"
            self.downsample.weight.data = other.downsample.weight.data
        for i in range(1,4):
            try:
                eval(f'self.gn{i}').weight.data = eval(f'other.gn{i}').weight.data.clone()
                eval(f'self.gn{i}').bias.data = eval(f'other.gn{i}').bias.data.clone()
            except:
                print(f"Did not set affine=True for gnorm(s) in gn{i}?")
            
    def forward(self, x, injection=None):
        if injection is None:
            injection = 0
        residual = x

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu(out)
        
        out = self.drop(self.conv2(out)) + injection
        out = self.gn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.gn3(self.relu3(out))

        return out
    
       
blocks_dict = {
    'BASIC': BasicBlock
}


class BranchNet(nn.Module):
    def __init__(self, blocks):
        """
        The residual block part of each resolution stream
        """
        super().__init__()
        self.blocks = blocks
    
    def forward(self, x, injection=None):
        blocks = self.blocks
        y = blocks[0](x, injection)
        for i in range(1, len(blocks)):
            y = blocks[i](y)
        return y
    
    
class DownsampleModule(nn.Module):
    def __init__(self, num_channels, in_res, out_res):
        """
        A downsample step from resolution j (with in_res) to resolution i (with out_res). A series of 2-strided convolutions.
        """
        super(DownsampleModule, self).__init__()
        # downsample (in_res=j, out_res=i)
        conv3x3s = []
        inp_chan = num_channels[in_res]
        out_chan = num_channels[out_res]
        self.level_diff = level_diff = out_res - in_res
        
        kwargs = {"kernel_size": 3, "stride": 2, "padding": 1, "bias": False}
        for k in range(level_diff):
            intermediate_out = out_chan if k == (level_diff-1) else inp_chan
            components = [('conv', nn.Conv2d(inp_chan, intermediate_out, **kwargs)), 
                          ('gnorm', nn.GroupNorm(NUM_GROUPS, intermediate_out, affine=True))]
            if k != (level_diff-1):
                components.append(('relu', nn.ReLU(inplace=True)))
            conv3x3s.append(nn.Sequential(OrderedDict(components)))
        self.net = nn.Sequential(*conv3x3s)  
        
    def _copy(self, other):
        for k in range(self.level_diff):
            self.net[k].conv.weight.data = other.net[k].conv.weight.data.clone()
            try:
                self.net[k].gnorm.weight.data = other.net[k].gnorm.weight.data.clone()
                self.net[k].gnorm.bias.data = other.net[k].gnorm.bias.data.clone()
            except:
                print("Did not set affine=True for gnorm(s)?")
            
    def forward(self, x):
        return self.net(x)


class UpsampleModule(nn.Module):
    def __init__(self, num_channels, in_res, out_res):
        """
        An upsample step from resolution j (with in_res) to resolution i (with out_res). 
        Simply a 1x1 convolution followed by an interpolation.
        """
        super(UpsampleModule, self).__init__()
        # upsample (in_res=j, out_res=i)
        inp_chan = num_channels[in_res]
        out_chan = num_channels[out_res]
        self.level_diff = level_diff = in_res - out_res
        
        self.net = nn.Sequential(OrderedDict([
                        ('conv', nn.Conv2d(inp_chan, out_chan, kernel_size=1, bias=False)),
                        ('gnorm', nn.GroupNorm(NUM_GROUPS, out_chan, affine=True)),
                        ('upsample', nn.Upsample(scale_factor=2**level_diff, mode='nearest'))
                   ]))
    
    def _copy(self, other):
        self.net.conv.weight.data = other.net.conv.weight.data.clone()
        try:
            self.net.gnorm.weight.data = other.net.gnorm.weight.data.clone()
            self.net.gnorm.bias.data = other.net.gnorm.bias.data.clone()
        except:
            print("Did not set affine=True for gnorm(s)?")
        
    def forward(self, x):
        return self.net(x)

    
class MDEQModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_channels, fuse_method, dropout=0.0):
        """
        An MDEQ layer (note that MDEQ only has one layer). 
        """
        super(MDEQModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_channels)

        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.num_channels = num_channels

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels, dropout=dropout)
        self.fuse_layers = self._make_fuse_layers()
        self.post_fuse_layers = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('relu', nn.ReLU(False)),
                ('conv', nn.Conv2d(num_channels[i], num_channels[i], kernel_size=1, bias=False)),
                ('gnorm', nn.GroupNorm(NUM_GROUPS // 2, num_channels[i], affine=True))
            ])) for i in range(num_branches)])   # shaojie
        self.relu = nn.ReLU(False)

    def _check_branches(self, num_branches, blocks, num_blocks, num_channels):
        """
        To check if the config file is consistent
        """
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _wnorm(self):
        """
        Apply weight normalization to the learnable parameters of MDEQ
        """
        self.post_fuse_fns = []
        for i, branch in enumerate(self.branches):
            for block in branch.blocks:
                block._wnorm()
            conv, fn = weight_norm(self.post_fuse_layers[i].conv, names=['weight'], dim=0)
            self.post_fuse_fns.append(fn)
            self.post_fuse_layers[i].conv = conv
        
        # Throw away garbage
        torch.cuda.empty_cache()
    
    def _copy(self, other):
        """
        Copy the parameter of an MDEQ layer. First copy the residual block, then the multiscale fusion part.
        """
        num_branches = self.num_branches
        for i, branch in enumerate(self.branches):
            for j, block in enumerate(branch.blocks):
                # Step 1: Basic block copying
                block._copy(other.branches[i].blocks[j])    
        
        for i in range(num_branches):
            for j in range(num_branches):
                # Step 2: Fuse layer copying
                if i != j:
                    self.fuse_layers[i][j]._copy(other.fuse_layers[i][j])     
            self.post_fuse_layers[i].conv.weight.data = other.post_fuse_layers[i].conv.weight.data.clone()
            try:
                self.post_fuse_layers[i].gnorm.weight.data = other.post_fuse_layers[i].gnorm.weight.data.clone()
                self.post_fuse_layers[i].gnorm.bias.data = other.post_fuse_layers[i].gnorm.bias.data.clone()
            except:
                print("Did not set affine=True for gnorm(s)?")
        
    def _reset(self, xs):
        """
        Reset the dropout mask and the learnable parameters (if weight normalization is applied)
        """
        for i, branch in enumerate(self.branches):
            for block in branch.blocks:
                block._reset(xs[i])
            if 'post_fuse_fns' in self.__dict__:
                self.post_fuse_fns[i].reset(self.post_fuse_layers[i].conv)    # Re-compute (...).conv.weight using _g and _v

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1, dropout=0.0):
        layers = nn.ModuleList()
        n_channel = num_channels[branch_index]
        for i in range(num_blocks[branch_index]):
            layers.append(block(n_channel, n_channel, dropout=dropout))
        return BranchNet(layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels, dropout=0.0):
        """
        Make the residual block (s; default=1 block) of MDEQ's f_\theta layer
        """
        branch_layers = [self._make_one_branch(i, block, num_blocks, 
                                               num_channels, dropout=dropout) for i in range(num_branches)]
        
        # branch_layers[i] gives the module that operates on input from resolution i
        return nn.ModuleList(branch_layers)

    def _make_fuse_layers(self):
        """
        Create the multiscale fusion layer (which does simultaneous up- and downsamplings).
        """
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_channels = self.num_channels
        fuse_layers = []
        for i in range(num_branches):
            fuse_layer = []                    # The fuse modules into branch #i
            for j in range(num_branches):
                if i == j:
                    fuse_layer.append(None)    # Identity if the same branch
                else:
                    module = UpsampleModule if j > i else DownsampleModule
                    fuse_layer.append(module(num_channels, in_res=j, out_res=i))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        # fuse_layers[i][j] gives the (series of) conv3x3s that convert input from branch j to branch i
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_channels

    def forward(self, x, injection, *args):
        """
        The two steps of a multiscale DEQ module (see paper): a per-resolution residual block and 
        a parallel multiscale fusion step.
        """
        if injection is None:
            injection = [0] * len(x)
        if self.num_branches == 1:
            return [self.branches[0](x[0], injection[0])]

        # Step 1: Per-resolution residual block
        x_block = []
        for i in range(self.num_branches):
            x_block.append(self.branches[i](x[i], injection[i]))
        
        # Step 2: Multiscale fusion
        x_fuse = []
        for i in range(self.num_branches):
            y = 0
            
            # Start fusing all #j -> #i up/down-samplings
            for j in range(self.num_branches):
                y += x_block[j] if i == j else self.fuse_layers[i][j](x_block[j])
            x_fuse.append(self.post_fuse_layers[i](y))
            
        return x_fuse


class MDEQNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        """
        Build an MDEQ model with the given hyperparameters
        """
        super(MDEQNet, self).__init__()
        global BN_MOMENTUM
        BN_MOMENTUM = kwargs.get('BN_MOMENTUM', 0.1)
        self.parse_cfg(cfg)
        init_chansize = self.init_chansize

        self.downsample = nn.Sequential(
            conv3x3(3, init_chansize, stride=(2 if self.downsample_times >= 1 else 1)),
            nn.BatchNorm2d(init_chansize, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            conv3x3(init_chansize, init_chansize, stride=(2 if self.downsample_times >= 2 else 1)),
            nn.BatchNorm2d(init_chansize, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        
        # PART I: Input injection module
        if self.downsample_times == 0 and self.num_branches <= 2:
            # We use the downsample module above as the injection transformation
            self.stage0 = None
        else:
            self.stage0 = nn.Sequential(nn.Conv2d(self.init_chansize, self.init_chansize, kernel_size=1, bias=False),
                                        nn.BatchNorm2d(self.init_chansize, momentum=BN_MOMENTUM),
                                        nn.ReLU(False))
        
        # PART II: MDEQ's f_\theta layer
        self.fullstage_cfg = cfg['MODEL']['EXTRA']['FULL_STAGE']      
        num_channels = self.num_channels
        block = blocks_dict[self.fullstage_cfg['BLOCK']]
        self.fullstage = self._make_stage(self.fullstage_cfg, num_channels, dropout=self.dropout)
        self.fullstage_copy = copy.deepcopy(self.fullstage)
        
        if self.wnorm:
            self.fullstage._wnorm()
            
        for param in self.fullstage_copy.parameters():
            param.requires_grad_(False)
        self.deq = MDEQWrapper(self.fullstage, self.fullstage_copy)
        self.iodrop = VariationalHidDropout2d(0.0)
        
    def parse_cfg(self, cfg):
        global DEQ_EXPAND, NUM_GROUPS
        self.num_branches = cfg['MODEL']['EXTRA']['FULL_STAGE']['NUM_BRANCHES']
        self.num_channels = cfg['MODEL']['EXTRA']['FULL_STAGE']['NUM_CHANNELS']
        self.init_chansize = self.num_channels[0]
        self.num_layers = cfg['MODEL']['NUM_LAYERS']
        self.dropout = cfg['MODEL']['DROPOUT']
        self.wnorm = cfg['MODEL']['WNORM']
        self.f_thres = cfg['MODEL']['F_THRES']
        self.b_thres = cfg['MODEL']['B_THRES']
        self.num_classes = cfg['MODEL']['NUM_CLASSES']
        self.downsample_times = cfg['MODEL']['DOWNSAMPLE_TIMES']
        self.pretrain_steps = cfg['TRAIN']['PRETRAIN_STEPS']
        DEQ_EXPAND = cfg['MODEL']['EXPANSION_FACTOR']
        NUM_GROUPS = cfg['MODEL']['NUM_GROUPS']
            
    def _make_stage(self, layer_config, num_channels, dropout=0.0):
        """
        Build an MDEQ block with the given hyperparameters
        """
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        block_type = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']
        
        return MDEQModule(num_branches, block_type, num_blocks, num_channels, fuse_method, dropout=dropout)
    
    def _forward(self, x, train_step=-1, **kwargs):
        """
        The core MDEQ module. In the starting phase, we can (optionally) enter a shallow stacked f_\theta training mode
        to warm up the weights (specified by the self.pretrain_steps; see below)
        """
        num_branches = self.num_branches
        f_thres = kwargs.get('f_thres', self.f_thres)
        b_thres = kwargs.get('b_thres', self.b_thres)
        writer = kwargs.get('writer', None)     # For tensorboard
        x = self.downsample(x)
        dev = x.device
        
        # Inject only to the highest resolution...
        x_list = [self.stage0(x) if self.stage0 else x]
        for i in range(1, num_branches):
            bsz, _, H, W = x_list[-1].shape
            x_list.append(torch.zeros(bsz, self.num_channels[i], H//2, W//2).to(dev))   # ... and the rest are all zeros
            
        z_list = [torch.zeros_like(elem) for elem in x_list]
        
        # For variational dropout mask resetting and weight normalization re-computations
        self.fullstage._reset(z_list)
        self.fullstage_copy._copy(self.fullstage)
        
        # Multiscale Deep Equilibrium!
        if 0 <= train_step < self.pretrain_steps:
            for layer_ind in range(self.num_layers):
                z_list = self.fullstage(z_list, x_list)
        else:
            if train_step == self.pretrain_steps:
                torch.cuda.empty_cache()
            z_list = self.deq(z_list, x_list, threshold=f_thres, train_step=train_step, writer=writer)
        
        y_list = self.iodrop(z_list)
        return y_list
    
    def forward(self, x, train_step=-1, **kwargs):
        raise NotImplemented    # To be inherited & implemented by MDEQClsNet and MDEQSegNet (see mdeq.py)