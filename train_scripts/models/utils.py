import numpy as np
import math

from collections import Counter

import torch
import torch.nn.functional as F
from torch import nn, Tensor

class Scale(nn.Module):
    """
    Multiply the output regression range by a learnable constant value
    """
    def __init__(self, init_value=1.0):
        """
        init_value : initial value for the scalar
        """
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, x):
        """
        input -> scale * input
        """
        return x * self.scale

class LayerNorm(nn.Module):
    """
    LayerNorm that supports inputs of size B, C, T
    """
    def __init__(
        self,
        num_channels,
        eps = 1e-5,
        affine = True,
        device = None,
        dtype = None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x**2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # apply weight and bias
        if self.affine:
            out *= self.weight
            out += self.bias

        return out

class Segment():
    def __init__(self, action, start, end):
        assert start >= 0
        self.action = action
        self.start = start
        self.end = end
        self.len = end - start + 1
    
    def __repr__(self):
        return "<%r %d-%d>" % (self.action, self.start, self.end)
    
    def intersect(self, s2):
        s = max([self.start, s2.start])
        e = min([self.end, s2.end])
        return max(0, e-s+1)

    def union(self, s2):
        s = min([self.start, s2.start])
        e = max([self.end, s2.end])
        return e-s+1

def parse_label(label: np.array):
    if not isinstance(label, np.ndarray):
        label = np.array(label)

    loc = label[:-1] != label[1:]
    loc = np.where(loc)[0]
    segs = []
    
    if len(loc) == 0:
        return [ Segment(label[0], 0, len(label)-1) ]
        
    for i, l in enumerate(loc):
        if i == 0:
            start = 0
            end = l
        else:
            start = loc[i-1]+1
            end = l
        
        seg = Segment(label[start], start, end)
        segs.append(seg)
        
    segs.append(Segment(label[loc[-1]+1], loc[-1]+1, len(label)-1))
    return segs


#############################################
def expand_frame_label(label, target_len: int):
    if len(label) == target_len:
        return label

    import torch
    is_numpy = isinstance(label, np.ndarray)
    if is_numpy:
        label = torch.from_numpy(label).float()
    if isinstance(label, list):
        label = torch.FloatTensor(label)

    label = label.view([1, 1, -1])
    resized = torch.nn.functional.interpolate(
        label, size=target_len, mode="nearest"
    ).view(-1)
    resized = resized.long()
    
    if is_numpy:
        resized = resized.detach().numpy()

    return resized

def shrink_frame_label(label: list, clip_len: int) -> list:
    num_clip = ((len(label) - 1) // clip_len) + 1
    new_label = []
    for i in range(num_clip):
        s = i * clip_len
        e = s + clip_len
        l = label[s:e]
        ct = Counter(l)
        l = ct.most_common()[0][0]
        new_label.append(l)

    return new_label

def easy_reduce(scores, mode="mean", skip_nan=False):
    assert isinstance(scores, list), type(scores)

    if len(scores) == 0:
        return np.nan

    elif isinstance(scores[0], list):
        average = []
        L = len(scores[0])
        for i in range(L):
            average.append( easy_reduce([s[i] for s in scores ], mode=mode, skip_nan=skip_nan) )

    elif isinstance(scores[0], np.ndarray):
        assert len(scores[0].shape) == 1
        stack = np.stack(scores, axis=0)
        average = stack.mean(0)

    elif isinstance(scores[0], tuple):
        average = []
        L = len(scores[0])
        for i in range(L):
            average.append( easy_reduce([s[i] for s in scores ], mode=mode, skip_nan=skip_nan) )
        average = tuple(average)

    elif isinstance(scores[0], dict):
        average = {}
        for k in scores[0]:
            average[k] = easy_reduce([s[k] for s in scores], mode=mode, skip_nan=skip_nan)

    elif isinstance(scores[0], float) or isinstance(scores[0], int) or isinstance(scores[0], np.float32): # TODO - improve
        if skip_nan:
            scores = [ x for x in scores if not np.isnan(x) ]

        if mode == "mean":
            average = np.mean(scores)
        elif mode == "max":
            average = np.max(scores)
        elif mode == "median":
            average = np.median(scores)
    else:
        raise TypeError("Unsupport Data Type %s" % type(scores[0]) )

    return average

###################################

def to_numpy(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, list):
        return np.array(x)
    else:
        return x

def egoprocel_vname2dataset(vname):
    if 'tent' in vname: #EPIC
        return 'EPIC'
    elif vname.startswith('S'): # CMU
        return 'CMU'
    elif 'Head' in vname: # PC
        return 'PC'
    elif vname.startswith('OP') or vname.startswith('P'): # egtea
        return 'EGTEA'
    elif vname.startswith('00'): # MECCANO
        return 'MECCANO'
    else:
        raise ValueError(vname)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Conv1d(n, k, kernel_size, 1, (kernel_size - 1) // 2) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        x = x.permute([1, 2, 0]) # 1, H, T
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        x = x.permute([2, 0, 1]) # T, 1, H 
        return x

class RandomBoxPerturber():
    def __init__(self, x_noise_scale=0.2, y_noise_scale=0.2, w_noise_scale=0.2, h_noise_scale=0.2) -> None:
        self.noise_scale = torch.Tensor([x_noise_scale, y_noise_scale])

    def __call__(self, refanchors: Tensor) -> Tensor:
        nq, bs, query_dim = refanchors.shape
        device = refanchors.device

        noise_raw = torch.rand_like(refanchors)
        noise_scale = self.noise_scale.to(device)[:query_dim]

        new_refanchors = refanchors * (1 + (noise_raw - 0.5) * noise_scale)
        return new_refanchors.clamp_(0, 1)

def gen_sineembed_for_position(pos_tensor, dim_t):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(dim_t, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos

def probabilities_to_one_hot(probabilities: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor of probabilities to one-hot encoding.

    Parameters:
    probabilities (torch.Tensor): A tensor of shape [T, bs, N] representing the probabilities
                                   for each second over N categories.

    Returns:
    torch.Tensor: A tensor of shape [T, bs, N] representing the one-hot encoded categories.
    """
    T, bs, N = probabilities.shape

    # Get the index of the maximum probability for each second
    _, max_indices = torch.max(probabilities, dim=-1)  # Shape [T, bs]

    # Create a tensor for one-hot encoding
    one_hot = torch.zeros(T, bs, N, device=probabilities.device)
    one_hot.scatter_(2, max_indices.unsqueeze(-1), 1)

    return one_hot

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

class RandomSegPerturber():
    def __init__(self, x_noise_scale=0.2, y_noise_scale=0.2) -> None:
        self.noise_scale = torch.Tensor([x_noise_scale, y_noise_scale])

    def __call__(self, refanchors: Tensor) -> Tensor:
        
        nq, query_dim = refanchors.shape
        device = refanchors.device

        noise_raw = torch.rand_like(refanchors.float())
        noise_scale = self.noise_scale.to(device)[:query_dim]

        new_refanchors = refanchors * (1 + (noise_raw - 0.5) * noise_scale)
        return new_refanchors.long()