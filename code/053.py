import os
import gc
import copy
import time
import random
import string
import datetime

# For data manipulation
import numpy as np
import pandas as pd
from scipy.special import softmax, expit, logit

from tqdm import tqdm

# Pytorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import register_parametrization

import torch.optim as optim
from torch.optim import lr_scheduler, Adam, AdamW
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import torch.utils.checkpoint
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import json
import pickle as pkl

from einops import rearrange, einsum, repeat
from einops.layers.torch import Rearrange

from rotary_embedding_torch import RotaryEmbedding

from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
from pprint import pprint, pformat


def get_logger(cfg):
    logger = getLogger(cfg.fname)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    if cfg.logger_file and cfg.train_model:
        filename = cfg.checkpoint_path / 'run.log'
        handler2 = FileHandler(filename=filename)
        handler2.setFormatter(Formatter("%(message)s"))
        logger.addHandler(handler2)
    return logger

def seed_torch(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
    if torch.backends.cudnn.is_available:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_corpus(file_path, corpus):
    data = load_json(file_path)
    for k,v in data.items():
        for sample in v['train']:
            sample['input'] = np.asarray(sample['input'], dtype='int8')
            sample['output'] = np.asarray(sample['output'], dtype='int8')
        for sample in v['test']:
            sample['input'] = np.asarray(sample['input'], dtype='int8')
            if sample.get('output', None):
                sample['output'] = np.asarray(sample['output'], dtype='int8')
        v['key'] = k
        v['corpus'] = corpus
    return data

def load_solutions(file_path):
    data = load_json(file_path)
    for k,v in data.items():
        data[k] = [np.asarray(a, dtype='int8') for a in v]
    return data

def load_generate_tasks(key, cfg):
    data = load_json(cfg.data_path / f'{key}.json')
    new_data = []
    for idx, sample in enumerate(data):
        new_sample = {'input' : np.asarray(sample['input'], dtype='int8'),
                      'output' : np.asarray(sample['output'], dtype='int8'),
                     }
        new_data.append(new_sample)
        if new_sample['input'].shape != new_sample['output'].shape:
            print('error size', key, idx)
            return None
            
    new_data = np.array(new_data)
    return new_data

def get_original_tasks(corpus_samples, corpus_solutions):
    new_data = {}
    for ((key, sample)) in corpus_samples.items():
        samples = sample['train'].copy()
        solution = corpus_solutions[key]
        for test_sample, test_solution in zip(sample['test'], solution):
            test_sample['output'] = test_solution
            samples.append(test_sample)
        new_data[key] = np.array(samples)
    return new_data

def get_color_perm(key, samples, num_perm, cfg):
    x = np.arange(10)
    perms = [x]
    images = [sample['input'] for sample in samples] + [sample['output'] for sample in samples] 
    color_sets = [set(np.unique(image)) for image in images]
    i = 0
    while len(perms) < num_perm:
        i += 1
        if i >= 100:
            cfg.logger.info('fail')
            break
        perm = np.random.permutation(x)
        diff = set(x[x != perm])
        for color_set in color_sets:
            if len(diff & color_set) == 0:
                break
        else:
            perms.append(perm)
    return perms

def load_data(cfg):
    dataset = {}
    
    training_challenges =  load_corpus(cfg.input_path  / 'arc-agi_training_challenges.json', 'train')
    training_solutions =   load_solutions(cfg.input_path / 'arc-agi_training_solutions.json')
    evaluation_challenges = load_corpus(cfg.input_path / 'arc-agi_evaluation_challenges.json', 'evaluation')
    evaluation_solutions = load_solutions(cfg.input_path / 'arc-agi_evaluation_solutions.json')
    
    with open(cfg.input_path / 'fixed_size.pkl', 'rb') as file:
        fixed_size_keys = pkl.load(file)

    fixed_size_train_keys = set(fixed_size_keys) & set(training_challenges.keys())
    keys = []   
    for key in tqdm(fixed_size_train_keys):
        data_key = load_generate_tasks(key, cfg)
        if data_key is not None:
            dataset[key] = data_key
            keys.append(key)
        elif cfg.local_rank == 0:
            try:
                cfg.logger.info(('error fixed size: ' + key))
            except:
                print('error fixed size: ' + key)
    dataset['keys'] = keys      

    dataset['train'] = get_original_tasks(training_challenges, training_solutions)

    if cfg.aug_color:
        perms = {}
        for key, samples in tqdm(dataset['train'].items()):
            #print(key)
            perms[key] = get_color_perm(key, samples, cfg.aug_color, cfg)
        dataset['aug_color'] = perms
    
    return dataset

def load_train(cfg):
    if cfg.aug_color:
        dataset = '_dataset_051.pkl'
    else:
        dataset = '_dataset_024.pkl'
    try:
        with open(cfg.save_path / (cfg.fname + dataset), 'rb') as file:
            data = pkl.load(file)    
    except:
        print('creating dataset', cfg.fname + dataset)
        data = load_data(cfg)
    
        with open(cfg.save_path / (cfg.fname + dataset), 'wb') as file:
            pkl.dump(data, file )
        
    return data 

def apply_symmetry(image, t, x, y):
    if t:
        image = image.T
    if x:
        image = image[::-1, :]
    if y:
        image = image[:, ::-1]
    return image.copy()

def apply_perm(image, perm):
    return perm[image]

def get_inv_perm(perm):
    x = np.zeros(len(perm), dtype='int')
    for i, j in enumerate(perm):
        x[j] = i
    return x

class ARCDataset(Dataset):
    
    def __init__(self, dataset, cfg):
        
        self.dataset = dataset
        self.cfg = cfg
        self.keys = dataset['keys']
        self.task_size = len(dataset[self.keys[0]]) // cfg.sample_batch_size
        length = len(self.keys) * self.task_size 
        
        if cfg.aug_sym:
            length = length * 8
            sym = [(t, x, y) for t in [0, 1] for x in [0, 1] for y in [0, 1]]
            self.sym = {i : s for i, s in enumerate(sym)}
            
        if cfg.aug_transpose:
            length = length * 2
            sym = [(t, x, y) for t in [0, 1] for x in [0] for y in [0]]
            self.sym = {i : s for i, s in enumerate(sym)}
            
        if cfg.aug_color:
            length = length * cfg.aug_color
            self.aug_color = dataset['aug_color']

        self.length = length 

        print('dataset size', self.length, (len(self.keys), self.task_size, cfg.sample_batch_size))
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        
        cfg = self.cfg
        
        if cfg.aug_sym:
            sym_idx = idx % 8
            idx = idx // 8
        else:
            sym_idx = 0
       
            
        if cfg.aug_transpose:
            sym_idx = idx % 2
            idx = idx // 2
        else:
            sym_idx = 0

        if cfg.aug_color:
            perm_idx = idx % cfg.aug_color
            idx = idx // cfg.aug_color    
        else:
            perm_idx = 0

        key_idx = idx // self.task_size
        sample_idx = idx % self.task_size
        
        key = self.keys[key_idx]
        key_data = self.dataset[key]
        
        sample_data = key_data[sample_idx]
        sample_input = sample_data['input']
        sample_output = sample_data['output']
        
        if cfg.aug_sym or cfg.aug_transpose:
            t, x, y = self.sym[sym_idx]
            sample_input = apply_symmetry(sample_input, t, x, y)
            sample_output = apply_symmetry(sample_output, t, x, y)

        if cfg.aug_color:     
            perms = self.aug_color[key]
            perm = perms[perm_idx]
            sample_input = apply_perm(sample_input, perm)
            sample_output = apply_perm(sample_output, perm)
            inv_perm = get_inv_perm(perm)
        else:
            perm = np.arange(10)
            inv_perm = perm
            
        x, y = sample_input.shape
            
        out = {
            'input' : torch.tensor(sample_input, dtype=torch.long).unsqueeze(0),
            'output' : torch.tensor(sample_output, dtype=torch.long).unsqueeze(0),
            'task' : torch.tensor([[key_idx]], dtype=torch.long).unsqueeze(0),
            'sym' : torch.tensor([[sym_idx]], dtype=torch.long).unsqueeze(0),
            'x' : torch.tensor([[x]], dtype=torch.long).unsqueeze(0),
            'y' : torch.tensor([[y]], dtype=torch.long).unsqueeze(0),
            'perm_idx' : torch.tensor([[perm_idx]], dtype=torch.long).unsqueeze(0),
            'perm' : torch.tensor(perm, dtype=torch.long).unsqueeze(0),
            'inv_perm' : torch.tensor(inv_perm, dtype=torch.long).unsqueeze(0),
        }
            
        return out
        

class ARCValidDataset(Dataset):
    def __init__(self, dataset, cfg):
        
        self.cfg = cfg
        keys = dataset['keys']
        samples , key_idx = [], []
        data = dataset['train']
        
        for idx, key in enumerate(keys):
            for sample in data[key]:
                samples.append(sample)
                key_idx.append(idx)
        self.samples = np.array(samples)
        self.key_idx = key_idx
        self.keys = keys
        
        
        if cfg.aug_sym:
            sym = [(t, x, y) for t in [0, 1] for x in [0, 1] for y in [0, 1]]
            self.sym = {i : s for i, s in enumerate(sym)}

        if cfg.aug_transpose:
            sym = [(t, x, y) for t in [0, 1] for x in [0] for y in [0]]
            self.sym = {i : s for i, s in enumerate(sym)}

        if cfg.aug_color:
            self.aug_color = dataset['aug_color']

        self.length = len(self.samples)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        cfg = self.cfg
            
        key_idx = self.key_idx[idx]
        
        sample = self.samples[idx]
        sample_input = sample['input']
        sample_output = sample['output']
        
        perm = np.arange(10)
        inv_perm = perm
        perm_idx = 0
        
        sym_idx = 0

        if cfg.aug_transpose and cfg.aug_color:
            
            out = []
            
            key = self.keys[key_idx]
            perms = self.aug_color[key]
            out = []

            for sym_idx, sym in self.sym.items():
                t, x, y = sym
                sample_input_sym = apply_symmetry(sample_input, t, x, y)
                sample_output_sym = apply_symmetry(sample_output, t, x, y)
                x, y = sample_input_sym.shape
                for perm_idx, perm in enumerate(perms):
                    sample_input_perm = apply_perm(sample_input_sym, perm)
                    sample_output_perm = apply_perm(sample_output_sym, perm)
                    inv_perm = get_inv_perm(perm)
                    
                    out.append({
                        'input' : torch.tensor(sample_input_perm, dtype=torch.long),
                        'output' : torch.tensor(sample_output_perm, dtype=torch.long),
                        'task' : torch.tensor([[key_idx]], dtype=torch.long),
                        'sym' : torch.tensor([[sym_idx]], dtype=torch.long),
                        'x' : torch.tensor([[x]], dtype=torch.long),
                        'y' : torch.tensor([[y]], dtype=torch.long),
                        'perm' : torch.tensor(perm, dtype=torch.long),
                        'perm_idx' : torch.tensor([[perm_idx]], dtype=torch.long),
                        'inv_perm' : torch.tensor(inv_perm, dtype=torch.long),
                    })
              
            out = collate_pad(out, stack=True)
            
        elif cfg.aug_sym or cfg.aug_transpose:
            
            out = []
            
            for sym_idx, sym in self.sym.items():
                t, x, y = sym
                sample_input_sym = apply_symmetry(sample_input, t, x, y)
                sample_output_sym = apply_symmetry(sample_output, t, x, y)
                x, y = sample_input_sym.shape
                out.append({
                    'input' : torch.tensor(sample_input_sym, dtype=torch.long),
                    'output' : torch.tensor(sample_output_sym, dtype=torch.long),
                    'task' : torch.tensor([[key_idx]], dtype=torch.long),
                    'sym' : torch.tensor([[sym_idx]], dtype=torch.long),
                    'x' : torch.tensor([[x]], dtype=torch.long),
                    'y' : torch.tensor([[y]], dtype=torch.long),
                    'perm' : torch.tensor(perm, dtype=torch.long),
                    'perm_idx' : torch.tensor([[perm_idx]], dtype=torch.long),
                    'inv_perm' : torch.tensor(inv_perm, dtype=torch.long),
                })
             
            out = collate_pad(out, stack=True)

        elif cfg.aug_color:
            
            key = self.keys[key_idx]
            perms = self.aug_color[key]
            
            out = []
            
            for perm_idx, perm in enumerate(perms):
                sample_input_perm = apply_perm(sample_input, perm)
                sample_output_perm = apply_perm(sample_output, perm)
                x, y = sample_input_perm.shape
                inv_perm = get_inv_perm(perm)
                
                out.append({
                    'input' : torch.tensor(sample_input_perm, dtype=torch.long),
                    'output' : torch.tensor(sample_output_perm, dtype=torch.long),
                    'task' : torch.tensor([[key_idx]], dtype=torch.long),
                    'sym' : torch.tensor([[sym_idx]], dtype=torch.long),
                    'x' : torch.tensor([[x]], dtype=torch.long),
                    'y' : torch.tensor([[y]], dtype=torch.long),
                    'perm' : torch.tensor(perm, dtype=torch.long),
                    'perm_idx' : torch.tensor([[perm_idx]], dtype=torch.long),
                    'inv_perm' : torch.tensor(inv_perm, dtype=torch.long),
                })
             
            out = collate_pad(out, stack=True)
            
           
        else:
            
            x, y = sample_input.shape
            
            out = {
                'input' : torch.tensor(sample_input, dtype=torch.long).unsqueeze(0),
                'output' : torch.tensor(sample_output, dtype=torch.long).unsqueeze(0),
                'task' : torch.tensor([[key_idx]], dtype=torch.long).unsqueeze(0),
                'sym' : torch.tensor([[sym_idx]], dtype=torch.long).unsqueeze(0),
                'x' : torch.tensor([[x]], dtype=torch.long).unsqueeze(0),
                'y' : torch.tensor([[y]], dtype=torch.long).unsqueeze(0),
                'perm' : torch.tensor(perm, dtype=torch.long).unsqueeze(0),
                'perm_idx' : torch.tensor([[perm_idx]], dtype=torch.long).unsqueeze(0),
                'inv_perm' : torch.tensor(inv_perm, dtype=torch.long).unsqueeze(0),
            }
            
        return out
        
        
def collate_pad(batch, stack=False):
    
    batch_dict = {}
    
    for key in ['input', 'output']:
        h_max = np.max([sample[key].shape[-2] for sample in batch])
        w_max = np.max([sample[key].shape[-1] for sample in batch])
        #print(h_max, w_max, flush=True)
        tensors = [F.pad(b[key], 
                         (0, w_max - b[key].shape[-1], 0, h_max - b[key].shape[-2], ), 
                         value=10, # cfg.num_color
                        ) 
                   for b in batch]
        if stack:
            batch_dict[key] = torch.stack(tensors)
        else:
            batch_dict[key] = torch.cat(tensors, 0)
            
    for key in ['task', 'sym', 'x', 'y', 'perm', 'perm_idx', 'inv_perm']:
        tensors = [b[key] for b in batch]
        if stack:
            batch_dict[key] = torch.stack(tensors)
        else:
            batch_dict[key] = torch.cat(tensors, 0)
            
    return batch_dict
    

def batch_to_device(batch, device):
    return {k:batch[k].to(device, non_blocking=True) for k in batch.keys() if k not in []}

def get_data_loader(dataset, istrain, cfg):
    if istrain:
        batch_size = cfg.train_batch_size
    else:
        batch_size = cfg.valid_batch_size
    if cfg.ddp and (istrain or cfg.ddp_valid):
        sampler = DistributedSampler(dataset,
                                     num_replicas=cfg.world_size,
                                     rank=cfg.local_rank,
                                     shuffle=istrain,
                                     seed=cfg.seed,
                                     drop_last=istrain,
                                    )
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=cfg.workers,
            shuffle=False,
            pin_memory=False,
            sampler=sampler,
            collate_fn=collate_pad,
        )
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=cfg.workers,
            shuffle=istrain,
            pin_memory=False,
            drop_last = istrain,
            collate_fn=collate_pad,
        )
    return data_loader 



def default(v, d):
    return v if v is not None else d

def l2norm(
    t,
    dim = -1,
    norm_eps = 0.,
    eps = None,
):

    if norm_eps == 0.:
        out = F.normalize(t, dim = dim, p = 2)
    else:
        eps = default(eps, 1e-5 if t.dtype == torch.float16 else 1e-10)
        norm = t.norm(dim = dim, keepdim = True)
        target_norm = norm.detach().clamp(min = 1. - norm_eps, max = 1. + norm_eps)
        divisor = norm / target_norm
        out = t / divisor.clamp(min = eps)
        
    return out

class Scale(nn.Module):
    """
    latter part of section 2.5 in the paper
    """
    def __init__(
        self,
        dim,
        init = 1.,
        scale = 1.
    ):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim) * scale)
        self.forward_scale = init / scale

    def forward(self):
        return self.scale * self.forward_scale
        
class Residual(nn.Module):
    def __init__(
        self,
        fn: nn.Module,
        dim: int,
        init: float,
        scale: float
    ):
        super().__init__()
        self.fn = fn
        self.branch_scale = Scale(dim, init, default(scale, dim ** -0.5))

    def forward(self, x, **kwargs):
        residual = x

        branch_out = l2norm(self.fn(x, **kwargs))
        out = l2norm(residual.lerp(branch_out, self.branch_scale()))

        return out

class EmbedAddNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        init: float,
        scale: float
    ):
        super().__init__()
        self.branch_scale = Scale(dim, init, default(scale, dim ** -0.5))
        self.l2norm = L2Norm(dim=-1)

    def forward(self, x, y):
        out = self.l2norm(x.lerp(y, self.branch_scale()))
        return out

class L2Norm(nn.Module):
    def __init__(self, dim = -1, norm_eps = 0.):
        super().__init__()
        self.dim = dim
        self.norm_eps = norm_eps

    def forward(self, t):
        return l2norm(t, dim = self.dim, norm_eps = self.norm_eps)

class NormLinear(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        norm_dim_in=True,
        parametrize=None,
        norm_eps=0.,
        cfg=None,
    ):
        super().__init__()
        parametrize = not cfg.manual_norm_weights
        self.linear = nn.Linear(dim, dim_out, bias = False)

        self.scale = 1
        self.parametrize = parametrize
        self.l2norm = L2Norm(dim=-1 if norm_dim_in else 0, norm_eps=norm_eps)

        if parametrize:
            register_parametrization(
                self.linear,
                'weight',
                self.l2norm
            )

        self.norm_weights_()

    @torch.no_grad()
    def norm_weights_(self):
        if self.parametrize:
            normed = self.weight
            original = self.linear.parametrizations.weight.original
            original.copy_(normed)
        else:
            self.weight.copy_(self.l2norm(self.weight))

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, x):
        return self.linear(x) * self.scale

class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_head,
        heads,
        dropout,
        cfg,
    ):
        super().__init__()
        self.cfg = cfg

        self.heads = heads
        
        dim_inner = dim_head * heads
        self.to_q = NormLinear(dim, dim_inner, cfg=cfg)
        self.to_k = NormLinear(dim, dim_inner, cfg=cfg)
        self.to_v = NormLinear(dim, dim_inner, cfg=cfg)

        self.dropout = dropout
        
        self.rotary_emb = RotaryEmbedding(dim_head)
        
        if cfg.qk_scale:
            self.qk_scale = Scale(dim, 1, dim ** -0.5)
        else:
            self.q_scale = Scale(dim, 1, dim ** -0.5)
            self.k_scale = Scale(dim, 1, dim ** -0.5)
        
        self.attn_scale = dim_head ** 0.5
        
        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_out = NormLinear(dim_inner, dim, norm_dim_in=cfg.norm_input, cfg=cfg)

    def forward(
        self,
        x,
        mask, 
    ):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        q, k, v = map(self.split_heads, (q, k, v))

        # query key rmsnorm

        q, k = map(l2norm, (q, k))
        if self.cfg.qk_scale:
            qk_scale = rearrange(self.qk_scale(), '(h d) -> h 1 d', h = self.heads)
            q = q * qk_scale
            k = k * qk_scale
        else:
            q = q * rearrange(self.q_scale(), '(h d) -> h 1 d', h = self.heads)
            k = k * rearrange(self.k_scale(), '(h d) -> h 1 d', h = self.heads)

        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        mask = rearrange(mask, 'b n -> b 1 n 1') == rearrange(mask, 'b n -> b 1 1 n')

        # scale is 1., as scaling factor is moved to s_qk (dk ^ 0.25) - eq. 16

        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.,
            scale=self.attn_scale,
        )

        out = self.merge_heads(out)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(
        self, 
        dim,
        cfg,
    ):
        super().__init__()
        self.attention = AttentionBlock(dim, cfg.dim_heads, cfg.heads, cfg.dropout, cfg)

    def forward(self, x, mask):
        b,i,j,d = x.shape
        xj = rearrange(x, 'b i j d -> (b i) j d')
        maskj = rearrange(mask, 'b i j -> (b i) j')
        xj = self.attention(xj, maskj)
        xj = rearrange(xj, '(b i) j d -> b i j d', b=b)
        xi = rearrange(x, 'b i j d -> (b j) i d')
        maski = rearrange(mask, 'b i j -> (b j) i')
        xi = self.attention(xi, maski)
        xi = rearrange(xi, '(b j) i d -> b i j d', b=b)
        return (xi + xj) / 2

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        cfg,
        *,
        expand_factor = 4,
        manual_norm_weights = False,
        s_hidden_init = 1.,
        s_hidden_scale = 1.,
        s_gate_init = 1.,
        s_gate_scale = 1.,
        norm_eps = 0.,
    ):
        super().__init__()
        
        self.dim = dim
        dim_inner = int(dim * expand_factor * 2 / 3)

        self.to_hidden = NormLinear(dim, dim_inner, cfg=cfg)
        self.to_gate = NormLinear(dim, dim_inner, cfg=cfg)

        self.hidden_scale = Scale(dim_inner, s_hidden_init, s_hidden_scale)
        self.gate_scale = Scale(dim_inner, s_gate_init, s_gate_scale)

        self.to_out = NormLinear(dim_inner, dim, norm_dim_in=cfg.norm_input, cfg=cfg)

    def forward(self, x):
        hidden, gate = self.to_hidden(x), self.to_gate(x)

        hidden = hidden * self.hidden_scale()
        gate = gate * self.gate_scale() * (self.dim ** 0.5)

        hidden = F.silu(gate) * hidden
        return self.to_out(hidden)
        
class ARCModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dim = cfg.hidden_dim
        alpha_init = 1 / cfg.depth
        alpha_scale = dim ** (-0.5)
        s_logit_init = 1.
        s_logit_scale = dim ** -0.5
        t_init = 0.5
        t_scale = 0.5

        self.color_embed = NormLinear(dim, cfg.num_color + 1, cfg=cfg)
        
        if self.cfg.aug_sym:
            num_task = cfg.num_task * 8
        else:
            num_task = cfg.num_task
            
        if self.cfg.aug_transpose:
            num_task = cfg.num_task * 2
        else:
            num_task = cfg.num_task
            
        if self.cfg.aug_color:
            num_task = cfg.num_task * cfg.aug_color
        else:
            num_task = cfg.num_task
        
        if cfg.task_embed_size:
            self.task_embed = NormLinear(cfg.task_embed_size, num_task, cfg=cfg)
            self.task_lin = NormLinear(cfg.task_embed_size, dim, cfg=cfg)
            self.task_norm = L2Norm(dim=-1)
        else:
            self.task_embed = NormLinear(dim, num_task, cfg=cfg)
            
        self.task_scale = Scale(dim, t_init, t_scale)
        self.embed_norm= L2Norm(dim=-1)

        self.layers =  nn.ModuleList([])
        for _ in range(cfg.depth):
            attention = Attention(dim, cfg)
            attn_with_residual = Residual(attention, dim, alpha_init, alpha_scale)
            
            ff = FeedForward(dim, cfg)
            ff_with_residual = Residual(ff, dim, alpha_init, alpha_scale)
            
            self.layers.append(nn.ModuleList([attn_with_residual, ff_with_residual]))
        
        self.to_logits = NormLinear(dim, cfg.num_color) if not cfg.tied_embedding else None
        self.logit_scale = Scale(cfg.num_color, s_logit_init, s_logit_scale)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=10, reduction='none')

    @torch.no_grad()
    def norm_weights_(self):
        for module in self.modules():
            if not isinstance(module, NormLinear):
                continue

            module.norm_weights_()

    def prepare_data(self, batch):
        
        input_ = batch['input']
        output = batch['output']
        mask = batch['input'] < 10
        task = batch['task']
        
        if self.cfg.aug_sym:
            sym = batch['sym']
            task = 8 * task + sym
            
        if self.cfg.aug_transpose:
            sym = batch['sym']
            task = 2 * task + sym
            
        if self.cfg.aug_color:
            perm_idx = batch['perm_idx']
            task = self.cfg.aug_color * task + perm_idx
            
        return input_, output, mask, task

    def forward(self, batch, return_loss=False):
        input_, output, mask, task = self.prepare_data(batch)
        #print('f', input_.shape, output.shape, mask.shape)
        
        color_embed = self.color_embed.weight[input_]
        
        task_embed = self.task_embed.weight[task]
        if self.cfg.task_embed_size:
            task_embed = self.task_lin(task_embed)
            task_embed = self.task_norm(task_embed)

        x =  self.embed_norm(color_embed.lerp(task_embed, self.task_scale()))
            
        #print('a', x.shape)
        for att, ff in self.layers:
            x = att(x, mask=mask)
            x = ff(x)
        
        #print('c', x.shape)
        if self.to_logits is not None:
            logits = self.to_logits(x)
        else:
            # tied embeddings
            logits = einsum(x, self.color_embed.weight, 'b i j d, c d -> b i j c')    
            if cfg.tie_task_embed:
                task_logits = einsum(x, task_embed, 'b i j d, b 1 1 d -> b i j 1')
                logits = logits.lerp(task_logits, self.task_scale())
            logits = logits[:, :, :, :self.cfg.num_color]
            logits = self.embed_norm(logits)
            
        logits = logits * self.logit_scale()
        
        #print('d', logits.shape)
        if return_loss:
            x = rearrange(logits, 'b i j d -> b d i j')
            loss = self.loss_fn(x, output)
            #print('g', loss.shape, mask.shape)
            loss = loss * mask
            loss = loss.sum((-1, -2)) / (mask.sum((-1, -2)) + 1e-2)
            loss = loss.mean()
            return logits, loss
        else:
            return logits


def train_epoch(loader, valid_loader, model, optimizer, scheduler, scaler, device, cfg):
    model.train()
    model.zero_grad()
    if cfg.verbose == 2:
        bar = tqdm(range(len(loader)))
    else:
        bar = range(len(loader))
    load_iter = iter(loader)
    loss_l = []
    grad_norm_l = []
    
    accumulate = cfg.accumulate
    start_time = datetime.datetime.now()
    
    for i, batch in zip(bar, load_iter):
        model.train()
        input_dict = batch_to_device(batch, device)
        with autocast('cuda', dtype=torch.float16, enabled=cfg.fp16):
            _, loss, = model(input_dict, return_loss=True)
        if cfg.fp16:
            scaler.scale(loss / cfg.accumulate).backward() 
        else:
            loss = loss / cfg.accumulate
            loss.backward()
        accumulate -= 1
        if accumulate == 0:
            if cfg.grad_norm:
                if cfg.fp16:
                    scaler.unscale_(optimizer)
                total_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm).item()
                if np.isnan(total_norm):
                    total_norm = cfg.grad_norm
                else:
                    total_norm = np.clip(total_norm, 0, cfg.grad_norm)
                grad_norm_l.append(total_norm)
            if cfg.fp16:
                scaler.step(optimizer)     
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

            if cfg.manual_norm_weights:
                model.norm_weights_()
            
            accumulate = cfg.accumulate
            scheduler.step()  
        loss_l.append(loss.detach().cpu().item())
        del loss, input_dict
        if cfg.verbose == 2:
            if cfg.grad_norm:
                bar.set_description('loss: %.4f grad norm %.1f' % (
                    np.mean(loss_l[-cfg.loss_length:]), np.mean(grad_norm_l[-cfg.loss_length:]),))
            else:
                bar.set_description('loss: %.4f ' % np.mean(loss_l[-cfg.loss_length:]))
            #if i > 0 and i % cfg.loss_length == 0:
            #    cfg.logger.info(('iter %6d loss: %.4f ' % (i, np.mean(loss_l[-cfg.loss_length:]))))
        if valid_loader is not None and i > 0 and (i % cfg.valid == 0 or i == len(loader) - 1):
            if cfg.ddp:
                torch.distributed.barrier()
            if cfg.local_rank == 0:
                valid_loss, avg_valid_loss = valid_epoch(valid_loader, model, device, cfg)
                curr_time = datetime.datetime.now()
                elapsed_time = str(curr_time - start_time)
                elapsed_time = elapsed_time.split('.')[0]
                if cfg.verbose:
                    cfg.logger.info(('iter %6d / %6d loss: %.4f valid loss : %.4f avg loss : %.4f elapsed %s' % (
                        i, len(loader), np.mean(loss_l[-cfg.loss_length:]), valid_loss, avg_valid_loss, elapsed_time)))
            if cfg.ddp:
                torch.distributed.barrier()
    optimizer.zero_grad()
    del loss_l, bar
    gc.collect()

def valid_epoch(loader, model, device, cfg):
    if cfg.ddp and not cfg.ddp_valid and cfg.train_model:
        model = model.module
    model.eval()
    model.zero_grad()
    if 0 and cfg.verbose:
        bar = tqdm(range(len(loader)))
    else:
        bar = range(len(loader))
    load_iter = iter(loader)
    loss_l = []
    #targets_l = []
    #logits_l = []
    avg_loss_l = []
    num_samples = 0
    with torch.no_grad():
        for i, batch in zip(bar, load_iter):                      
            input_dict = batch_to_device(batch, device)
            with autocast('cuda', enabled=cfg.fp16):
                logits, loss,  = model(input_dict, return_loss=True)  
            num_sample = logits.shape[0]
            num_samples += num_sample
            loss = loss.detach().cpu().item() 
            if cfg.aug_transpose and cfg.aug_color:
                with torch.no_grad():
                    avg_loss = get_avg_loss_color_transpose(logits, input_dict, cfg)
            elif cfg.aug_sym:
                with torch.no_grad():
                    avg_loss = get_avg_loss(logits, input_dict, 8)
            elif cfg.aug_transpose:
                with torch.no_grad():
                    avg_loss = get_avg_loss(logits, input_dict, 2)
            elif cfg.aug_color:
                with torch.no_grad():
                    avg_loss = get_avg_color_loss(logits, input_dict, cfg.aug_color, cfg)
            else:
                avg_loss = loss
            loss_l.append(loss * num_sample)
            avg_loss_l.append(avg_loss * num_sample)
            del loss, avg_loss, input_dict, batch
            #targets = batch['output']
            #targets_l.append(targets)
    return np.sum(loss_l) / num_samples, np.sum(avg_loss_l) / num_samples

def apply_inverse_symmetry(image, sym):
    t, x, y = sym
    if y:
        image = image.flip(1)
    if x:
        image = image.flip(0)
    if t:
        image = image.transpose(0, 1)
    return image

def apply_inverse_perm(image, perm):
    return image[:, :, perm]

def get_avg_loss_color_transpose(logits, batch, cfg):
    num_sym = 2
    num_perm = cfg.aug_color
    
    xs = batch['x']
    ys = batch['y']
    outputs = batch['output']
    logits = logits #.detach().cpu()
    
    syms = [(t, x, y) for t in [0, 1] for x in [0] for y in [0]]

    sym_idxs = batch['sym']
    perms = batch['perm']
    inv_perms = batch['inv_perm']
    
    loss_fn = nn.CrossEntropyLoss()
    
    logits = torch.split(logits, num_sym * num_perm)
    xs = torch.split(xs, num_sym * num_perm)
    ys = torch.split(ys, num_sym * num_perm)
    outputs = torch.split(outputs, num_sym * num_perm)
    sym_idxs  = torch.split(sym_idxs, num_sym * num_perm)
    perms  = torch.split(perms, num_sym * num_perm)
    inv_perms  = torch.split(inv_perms, num_sym * num_perm)
    
    avg_loss_l = []
    
    for logit_batch, output_batch, x_batch, y_batch, sym_batch, perm_batch, inv_perm_batch in zip(
        logits, outputs, xs, ys, sym_idxs, perms, inv_perms):
        
        logit_avg = 0.
        
        output = output_batch[0][:x_batch[0], :y_batch[0]]
        
        for i  in range(len(syms)):
            
            for j in range(num_perm):
                
                ij = i * num_perm + j
                
                logit = logit_batch[ij, :x_batch[ij], :y_batch[ij], :]
                
                perm_ij = perm_batch[ij]
                logit0 = apply_inverse_perm(logit, perm_ij)
                
                sym_ij = syms[sym_batch[ij]]
                logit = apply_inverse_symmetry(logit0, sym_ij)
                #print(x_batch[i], y_batch[i], logit0.shape, logit.shape, sym_ij, flush=True)
                
                output_ij = output_batch[ij, :x_batch[ij], :y_batch[ij]]
                
                inv_perm_ij = inv_perm_batch[ij]
                output_ij = inv_perm_ij[output_ij]
                
                output_ij = apply_inverse_symmetry(output_ij, sym_ij)
                
                if not torch.all(output == output_ij):
                    print('error', flush)
                    
                logit_avg = logit_avg + logit / (num_sym * num_perm)
           
        
        logit_avg = rearrange(logit_avg, 'i j d -> 1 d i j')
        output = rearrange(output, 'i j -> 1 i j')
        avg_loss = loss_fn(logit_avg, output).detach().item()
        avg_loss_l.append(avg_loss)
        
    return np.mean(avg_loss_l)

def get_avg_loss(logits, batch, num_sym):
    
    xs = batch['x']
    ys = batch['y']
    outputs = batch['output']
    logits = logits #.detach().cpu()
    
    if num_sym == 8:
        syms = [(t, x, y) for t in [0, 1] for x in [0, 1] for y in [0, 1]]
    elif num_sym == 2:
        syms = [(t, x, y) for t in [0, 1] for x in [0] for y in [0]]

    loss_fn = nn.CrossEntropyLoss()
    
    logits = torch.split(logits, num_sym)
    xs = torch.split(xs, num_sym)
    ys = torch.split(ys, num_sym)
    outputs = torch.split(outputs, num_sym)
    avg_loss_l = []
    
    for logit_batch, output_batch, x_batch, y_batch in zip(logits, outputs, xs, ys):
        logit_avg = 0.
        for i, sym in enumerate(syms):
            logit0 = logit_batch[i, :x_batch[i], :y_batch[i], :]
            logit = apply_inverse_symmetry(logit0, sym)
            #print(x_batch[i], y_batch[i], logit0.shape, logit.shape, sym, flush=True)
            logit_avg = logit_avg + logit / num_sym
            
        output = output_batch[0][:x_batch[0], :y_batch[0]]
        
        logit_avg = rearrange(logit_avg, 'i j d -> 1 d i j')
        output = rearrange(output, 'i j -> 1 i j')
        avg_loss = loss_fn(logit_avg, output).detach().item()
        avg_loss_l.append(avg_loss)
        
    return np.mean(avg_loss_l)

def get_avg_color_loss(logits, batch, num_perm, cfg):
    
    xs = batch['x']
    ys = batch['y']
    outputs = batch['output']
    logits = logits #.detach().cpu()
    
    perms = batch['perm']
    inv_perms = batch['inv_perm']
    #if cfg.local_rank == 0:
    #    print(xs.shape, outputs.shape, logits.shape, perms.shape)

    loss_fn = nn.CrossEntropyLoss()
    
    logits = torch.split(logits, num_perm)
    xs = torch.split(xs, num_perm)
    ys = torch.split(ys, num_perm)
    outputs = torch.split(outputs, num_perm)
    perms  = torch.split(perms, num_perm)
    inv_perms  = torch.split(inv_perms, num_perm)

    avg_loss_l = []
    
    for logit_batch, output_batch, x_batch, y_batch, perm_batch, inv_perm_batch in zip(logits, outputs, xs, ys, perms, inv_perms):
        logit_avg = 0.
        output = output_batch[0][:x_batch[0], :y_batch[0]]
        for i in range(num_perm):
            logit0 = logit_batch[i, :x_batch[i], :y_batch[i], :]
            perm_i = perm_batch[i]
            #print(logit0.shape, perm_batch.shape, num_perm, perm_batch[i], flush=True)
            logit = apply_inverse_perm(logit0, perm_i)
            
            logit_avg = logit_avg + logit / num_perm

            output_i = output_batch[i, :x_batch[i], :y_batch[i]]
            inv_perm_i = inv_perm_batch[i]
            output_i = inv_perm_i[output_i]
            if not torch.all(output == output_i):
                print('error', flush)
        
        logit_avg = rearrange(logit_avg, 'i j d -> 1 d i j')
        output = rearrange(output, 'i j -> 1 i j')
        avg_loss = loss_fn(logit_avg, output).detach().item()
        avg_loss_l.append(avg_loss)
        
    return np.mean(avg_loss_l)

def get_optimizer(model, cfg):
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters()],
         'lr': cfg.lr, 'weight_decay': 0},
    ]
    if cfg.optimizer == 'AdamW':
        optimizer = AdamW(optimizer_parameters, 
                                      betas=(cfg.opt_beta1, cfg.opt_beta2),
                                      eps=cfg.opt_eps,
                         )
    else:
        optimizer = Adam(optimizer_parameters, 
                                      betas=(cfg.opt_beta1, cfg.opt_beta2),
                                      eps=cfg.opt_eps,
                         )  
    return optimizer

def get_scheduler(optimizer, train_data_loader, cfg):
    if optimizer is None:
        return None
    if cfg.scheduler == 'cosine':
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.lr,
            epochs=cfg.num_epochs,
            steps_per_epoch=len(train_data_loader),
            pct_start=cfg.pct_start,
            anneal_strategy="cos",
            final_div_factor=cfg.final_div_factor,
        )
    return scheduler

def save_checkpoint(model, cfg):
    if cfg.ddp:
        model = model.module
    checkpoint_path = cfg.checkpoint_path
    save_path = checkpoint_path / ('%s_%d.pt' % (cfg.fname, cfg.seed))
    if cfg.local_rank == 0:
        cfg.logger.info('saving %s ...' % save_path)
        checkpoint = {
            'model': model.state_dict(),
            'seed': cfg.seed,
            }
        checkpoint_path = cfg.checkpoint_path
        torch.save(checkpoint, save_path)
        cfg.logger.info('done')

def load_checkpoint(cfg):
    checkpoint_path = cfg.checkpoint_path
    save_path = checkpoint_path / ('%s_%d.pt' % (cfg.fname, cfg.seed))
    if cfg.local_rank == 0:
        cfg.logger.info('loading %s ...' % save_path)
    checkpoint = torch.load(save_path, map_location='cpu')
    model = get_model(cfg)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    if cfg.local_rank == 0:
        cfg.logger.info('done')
    return model

def get_model(cfg):
    model = ARCModel(cfg).to(cfg.device)
    return model

def train_model(cfg, train_dataset, valid_dataset):    
    train_dataloader = get_data_loader(train_dataset, istrain=True, cfg=cfg)
    if valid_dataset is not None:
        valid_dataloader = get_data_loader(valid_dataset, istrain=False, cfg=cfg)
    else:
        valid_dataloader = None
    device = cfg.device
    
    model = get_model(cfg)
            
    if cfg.ddp:
        model = DistributedDataParallel(model, 
                                        device_ids=[cfg.local_rank],
                                        #find_unused_parameters=True,
                                       )
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, train_dataloader, cfg)
    if cfg.fp16:
        scaler = GradScaler('cuda')
    else:
        scaler = None
    score = None
    preds = None
    targets = None
    logits = None
    for epoch in range(cfg.num_epochs):
        if cfg.ddp:
            torch.distributed.barrier()

        train_epoch(train_dataloader, valid_dataloader, model, optimizer, scheduler, scaler, device, cfg)
        if cfg.ddp:
            torch.distributed.barrier()
        if cfg.local_rank == 0:
            save_checkpoint(model, cfg)
    if cfg.local_rank == 0:
        save_checkpoint(model, cfg)
    del model, optimizer, scheduler, scaler, train_dataloader, 
    if valid_dataset is not None:
        del valid_dataloader
    gc.collect()
    torch.cuda.empty_cache()
    if cfg.ddp:
        torch.distributed.barrier()

def set_save_path(cfg):
    
    fname_path = cfg.save_path / cfg.fname
    if not fname_path.exists():
        fname_path.mkdir()
        
    exp = 0
    while((fname_path / ('exp_%d' % exp)).exists()):
        exp += 1
    cfg.exp = exp
    cfg.checkpoint_path = fname_path / ('exp_%d' % exp)
    cfg.checkpoint_path.mkdir()

def get_exp(cfg):
    device = cfg.device
    local_rank = cfg.local_rank
    world_size = cfg.world_size
    if local_rank == 0:
        exp = torch.tensor([cfg.exp], dtype=torch.int64, device=cfg.device)
    else:
        exp = torch.tensor([0], dtype=torch.int64, device=cfg.device)
    torch.distributed.broadcast(exp, 0)
    exp = exp.item()
    return exp


def run(cfg):
    if cfg.ddp:
        # os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        device = rank % torch.cuda.device_count()
        print(local_rank, rank, device)
        cfg.device = device
        cfg.local_rank = local_rank
        cfg.world_size = world_size
        if local_rank != 0:
            cfg.verbose = False
        cfg.print = (local_rank == 0)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu
        device = torch.device('cuda')
        cfg.device = device
        cfg.print = True
        cfg.local_rank = 0
        
    if cfg.train_model and cfg.local_rank == 0:
        set_save_path(cfg)
        
    if cfg.train_model and cfg.ddp:
        torch.distributed.barrier(device_ids=[cfg.device])
        exp = get_exp(cfg)
        if cfg.local_rank != 0:
            cfg.checkpoint_path = cfg.save_path / cfg.fname / ('exp_%d' % exp)
            print(cfg.device, cfg.checkpoint_path)

    if cfg.local_rank == 0:
        cfg.logger = get_logger(cfg)
        cfg.logger.info(pformat(cfg))
        cfg.logger.info('using ' + cfg.fname)
        cfg.logger.info('saving checkpoints to %s' % cfg.checkpoint_path)
        
    if cfg.train_model :

        train = load_train(cfg)
        
        len_train = len(train)
        scores = []
        all_targets = []
        all_preds = []

        train_dataset = ARCDataset(train, cfg)

        if cfg.valid:
            valid_dataset = ARCValidDataset(train, cfg)
        else:
            valid_dataset = None
        seed_torch(cfg.seed)
        
        train_model(cfg, train_dataset, valid_dataset)
        
        if cfg.local_rank == 0:
            msg = ('seed %d ' % cfg.seed) + ('done')
            cfg.logger.info(msg)
            
    if cfg.ddp:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == '__main__': 
    import os
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    import sys
    import argparse
    import importlib
    from copy import copy
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--cfg", help='config file to be used')

    parser_args, other_args = parser.parse_known_args(sys.argv)
    cfg_name = parser_args.cfg
    print('Using config:', cfg_name, flush=True)
    cfg_path = cfg_name.split('/')[:-1]
    cfg_name = cfg_name.split('/')[-1]
    if cfg_path != []:
        cfg_path = '/'.join(cfg_path)
        sys.path.append(cfg_path)
    cfg_module = importlib.import_module(cfg_name)
    cfg = copy(cfg_module.cfg)

    unused_args = []
    override_args = []
    if len(other_args) > 1:
        other_args = {k.replace('-',''):v for k, v in zip(other_args[1::2], other_args[2::2])}
        for key in other_args:
            if key in cfg.__dict__:
                override_args.append((key, cfg.__dict__[key], other_args[key]))
                #print(f'overwriting cfg.{key}: {cfg.__dict__[key]} -> {other_args[key]}')
                cfg_type = type(cfg.__dict__[key])
                if cfg_type == bool:
                    cfg.__dict__[key] = other_args[key] == 'True'
                elif cfg_type == type(None):
                    cfg.__dict__[key] = other_args[key]
                else:
                    cfg.__dict__[key] = cfg_type(other_args[key])
            else:
                unused_args.append(key)
    if cfg.local_rank == 0 and len(unused_args) > 0:
        print('WARNING: unused args:', unused_args)
    try:
        cfg_module.finalize_cfg(cfg)
    except:
        pass
    run(cfg)

