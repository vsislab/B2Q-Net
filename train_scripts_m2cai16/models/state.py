import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
from timm.models.layers import DropPath

def add_positional_encoding(tensor, pos):
    if pos is None:
        return tensor
    else:
        d = pos.size(-1)
        tensor = tensor.clone()
        tensor[:, :, :d] = tensor[:, :, :d] + pos
        return tensor

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class SALayer(nn.Module):
    """
    self attention
    """

    def __init__(self, q_dim, nhead, dim_feedforward=2048, kv_dim=None,
                 dropout=0.1, attn_dropout=0.1,
                 activation="relu", vpos=False):
        super().__init__()

        kv_dim = q_dim if kv_dim is None else kv_dim
        self.multihead_attn = nn.MultiheadAttention(q_dim, nhead, kdim=kv_dim, vdim=kv_dim, dropout=attn_dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(q_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, q_dim)

        self.norm1 = nn.LayerNorm(q_dim)
        self.norm2 = nn.LayerNorm(q_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.q_dim = q_dim
        self.kv_dim=kv_dim
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward

        self.use_vpos = vpos
        self.dropout_rate = (dropout, attn_dropout)

    def __str__(self) -> str:
        return f"SALayer( q({self.q_dim})xkv({self.kv_dim})->{self.q_dim}, head:{self.nhead}, ffdim:{self.dim_feedforward}, dropout:{self.dropout_rate}, vpos:{self.use_vpos} )"
    
    def __repr__(self):
        return str(self)

    def forward(self, tgt, key, value, 
            query_pos: Optional[Tensor] = None,
            key_pos: Optional[Tensor] = None,
            value_pos: Optional[Tensor] = None):
        """
        tgt : query
        memory: key and value
        """
        query=add_positional_encoding(tgt, query_pos)
        key=add_positional_encoding(key, key_pos)
        if self.use_vpos:
            value=add_positional_encoding(value, value_pos)

        tgt2, self.attn = self.multihead_attn(query, key, value, average_attn_weights=False) # attn: nhead, batch, q, k

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        return tgt 

class state_MultiHeadAttention(nn.Module):
    def __init__(self, s_dim, i_dim, embed_dim, num_heads, dropout=0.1, cfg=None):
        super(state_MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.s_dim = s_dim
        self.i_dim = i_dim

        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.s_proj = nn.Linear(self.s_dim, self.embed_dim)
        self.i_proj = nn.Linear(self.i_dim, self.embed_dim)
        self.values_s_proj = nn.Linear(self.s_dim, self.embed_dim)
        self.values_i_proj = nn.Linear(self.i_dim, self.embed_dim)

        self.out_s_proj = nn.Linear(self.embed_dim, self.s_dim)
        self.out_i_proj = nn.Linear(self.embed_dim, self.i_dim)

        self.stable_softmax_2d = True
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.s_proj.weight)
        self.s_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.i_proj.weight)
        self.i_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_s_proj.weight)
        self.values_s_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.values_i_proj.weight)
        self.values_i_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_s_proj.weight)
        self.out_s_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_i_proj.weight)
        self.out_i_proj.bias.data.fill_(0)

    def forward(self, s, i, attention_mask_s=None, attention_mask_i=None):
        """
        Forward pass for the state_MultiHeadAttention.

        Args:
            s (torch.Tensor): Input source features of shape (bsz, n_img, dim).
            i (torch.Tensor): Input input features of shape (bsz, n_text, dim).
            attention_mask_s (torch.Tensor, optional): Attention mask for source features.
            attention_mask_i (torch.Tensor, optional): Attention mask for input features.

        Returns:
            tuple: Updated source and input features.
        """
        s = s.permute([1, 0, 2])
        i = i.permute([1, 0, 2])

        bsz, tgt_len, _ = s.size()

        query_states = self.s_proj(s) * self.scale
        key_states = self._shape(self.i_proj(i), -1, bsz)
        value_s_states = self._shape(self.values_s_proj(s), -1, bsz)
        value_i_states = self._shape(self.values_i_proj(i), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_s_states = value_s_states.view(*proj_shape)
        value_i_states = value_i_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))  # bs*nhead, n_img, n_text

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(
                attn_weights, min=-50000
            )
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(
                attn_weights, max=50000
            )
        
        attn_weights_T = attn_weights.transpose(1, 2)
        attn_weights_i = attn_weights_T - torch.max(attn_weights_T, dim=-1, keepdim=True)[0]
        if self.clamp_min_for_underflow:
            attn_weights_i = torch.clamp(
                attn_weights_i, min=-50000
            )
        if self.clamp_max_for_overflow:
            attn_weights_i = torch.clamp(
                attn_weights_i, max=50000
            )

        # Mask for input
        if attention_mask_s is not None:
            attention_mask_s = (
                attention_mask_s[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            attn_weights_i.masked_fill_(attention_mask_s, float("-inf"))

        attn_weights_i = attn_weights_i.softmax(dim=-1)

        # Mask for source
        if attention_mask_i is not None:
            attention_mask_i = (
                attention_mask_i[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            attn_weights.masked_fill_(attention_mask_i, float("-inf"))
        attn_weights_s = attn_weights.softmax(dim=-1)

        attn_probs_s = F.dropout(attn_weights_s, p=self.dropout, training=self.training)
        attn_probs_i = F.dropout(attn_weights_i, p=self.dropout, training=self.training)

        attn_output_s = torch.bmm(attn_probs_s, value_i_states)
        attn_output_i = torch.bmm(attn_probs_i, value_s_states)

        if attn_output_s.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_s` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_s.size()}"
            )

        if attn_output_i.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output_i` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_i.size()}"
            )

        attn_output_s = attn_output_s.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_s = attn_output_s.transpose(1, 2)
        attn_output_s = attn_output_s.reshape(bsz, tgt_len, self.embed_dim)

        attn_output_i = attn_output_i.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_i = attn_output_i.transpose(1, 2)
        attn_output_i = attn_output_i.reshape(bsz, src_len, self.embed_dim)

        attn_output_s = self.out_s_proj(attn_output_s)
        attn_output_i = self.out_i_proj(attn_output_i)

        return attn_output_s.permute([1, 0, 2]), attn_output_i.permute([1, 0, 2])
        # return attn_output_s, attn_output_i

class state_AttentionBlock(nn.Module):
    def __init__(
        self,
        s_dim: int,
        i_dim: int,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        init_values: float = 1,
    ):
        """
        Initializes the state_AttentionBlock.

        Args:
            s_dim (int): Dimensionality of the state feature vectors.
            i_dim (int): Dimensionality of the input feature vectors.
            embed_dim (int): Dimensionality of input and attention feature vectors.
            num_heads (int): Number of heads to use in the Multi-Head Attention block.
            dropout (float): Amount of dropout to apply in the feed-forward network.
            drop_path (float): Probability of dropping paths in the network.
            init_values (float): Initial scaling values for layer normalization.
        """
        super(state_AttentionBlock, self).__init__()

        # Pre-layer normalization
        self.layer_norm_s = nn.LayerNorm(s_dim)
        self.layer_norm_i = nn.LayerNorm(i_dim)

        self.s_self_attn = SALayer(q_dim=s_dim, nhead=num_heads)
        self.i_self_attn = SALayer(q_dim=i_dim, nhead=num_heads)

        self.attn = state_MultiHeadAttention(
            s_dim=s_dim, i_dim=i_dim, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )

        # Layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_s = nn.Parameter(init_values * torch.ones(s_dim), requires_grad=True)
        self.gamma_i = nn.Parameter(init_values * torch.ones(i_dim), requires_grad=True)

        # State FFN
        self.activation_s = _get_activation_fn('relu')
        self.linear1_s = nn.Linear((s_dim), embed_dim)
        self.linear2_s = nn.Linear(embed_dim, s_dim)
        self.dropout_s = nn.Dropout(dropout)
        self.dropout2_s = nn.Dropout(dropout)

        # Input FFN
        self.activation_i = _get_activation_fn('relu')
        self.linear1_i = nn.Linear((i_dim), embed_dim)
        self.linear2_i = nn.Linear(embed_dim, i_dim)
        self.dropout_i = nn.Dropout(dropout)
        self.dropout2_i = nn.Dropout(dropout)

    def forward(self, s: torch.Tensor, i: torch.Tensor, s_pos: torch.Tensor, i_pos: torch.Tensor,
                attention_mask_s: torch.Tensor = None, 
                attention_mask_i: torch.Tensor = None) -> tuple:
        """
        Forward pass for the state_AttentionBlock.

        Args:
            s (torch.Tensor): Input state features.
            i (torch.Tensor): Input input features.
            attention_mask_s (torch.Tensor, optional): Attention mask for source features.
            attention_mask_i (torch.Tensor, optional): Attention mask for input features.

        Returns:
            tuple: Updated source and input features.
        """
        s = self.layer_norm_s(s)
        i = self.layer_norm_i(i)

        s_1 = self.s_self_attn(s, s, s, s_pos, s_pos, s_pos)
        i_1 = self.i_self_attn(i, i, i, i_pos, i_pos, i_pos)

        delta_s, delta_i = self.attn(
            s_1, i_1, attention_mask_s=attention_mask_s, attention_mask_i=attention_mask_i
        )

        s_output = s_1 + self.drop_path(self.gamma_s * delta_s)
        i_output = i_1 + self.drop_path(self.gamma_i * delta_i)
        
        s_output2 = self.linear2_s(self.dropout_s(self.activation_s(self.linear1_s(s_output))))
        s_output = s_output + self.dropout2_s(s_output2)

        i_output2 = self.linear2_i(self.dropout_i(self.activation_i(self.linear1_i(i_output))))
        i_output = i_output + self.dropout2_i(i_output2)

        return s_output, i_output