# %%
import math
import typing as ty
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor

import lib


# %%
class Tokenizer(nn.Module):
    category_offsets: ty.Optional[Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_token: int,
        bias: bool,
    ) -> None:
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)

        # take [CLS] token into account
        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        # The initialization is inspired by nn.Linear
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.weight.shape[0])
            nn_init.uniform_(self.bias, -bound, bound)

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None
        x_num = torch.cat(
            [torch.ones(len(x_some), 1, device=x_some.device)]  # [CLS]
            + ([] if x_num is None else [x_num]),
            dim=1,
        )
        x = self.weight[None] * x_num[:, :, None]
        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=1,
            )
        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )
            x = x + bias[None]
        return x


class MultiheadGEAttention(nn.Module):
    """
    Learn relations among features and feature selection strategy in data-driven manner.
    """
    def __init__(
        # Normal Attention Args
        self, d: int, n_heads: int, dropout: float, initialization: str,
        # FR-Graph Args
        n: int, sym_weight: bool = True, sym_topology: bool = False, nsi: bool = True,
    ) -> None:
        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None
        # head and tail transformation
        self.W_head = nn.Linear(d, d)
        if sym_weight:
            self.W_tail = self.W_head # symmetric weights
        else:
            self.W_tail = nn.Linear(d, d) # ASYM
        # relation embedding: learnable diagonal matrix
        self.rel_emb = nn.Parameter(torch.ones(n_heads, d // self.n_heads))

        for m in [self.W_head, self.W_tail, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

        self.n_cols = n + 1 # Num of Nodes: input feature nodes + [Cross-level Readout]
        self.nsi = nsi # no self-interaction

        # column embeddings: semantics for each column
        d_col = math.ceil(2 * math.log2(self.n_cols)) # dim for column header embedding -> d_header += d
        self.col_head = nn.Parameter(Tensor(self.n_heads, self.n_cols, d_col))
        if not sym_topology:
            self.col_tail = nn.Parameter(Tensor(self.n_heads, self.n_cols, d_col))
        else:
            self.col_tail = self.col_head # share the parameter
        for W in [self.col_head, self.col_tail]:
            if W is not None:
                # correspond to Tokenizer initialization
                nn_init.kaiming_uniform_(W, a=math.sqrt(5))

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        key_compression: ty.Optional[nn.Linear],
        value_compression: ty.Optional[nn.Linear],
    ) -> ty.Tuple[Tensor, Tensor]:
        batch_size, n_q_tokens, d = x_q.shape
        _, n_kv_tokens, _ = x_kv.shape
        n_head_nodes = n_q_tokens
        d_value = d // self.n_heads

        # FR-Graph: Edge weights
        f_head = self.W_head(x_q)  # [batch_size, n_q_tokens, d]
        f_tail = self.W_tail(x_kv) # [batch_size, n_kv_tokens, d]
        f_v = self.W_v(x_kv)       # [batch_size, n_kv_tokens, d]

        # Reshape for multi-head attention
        f_head = self._reshape(f_head)  # [batch_size * n_heads, n_q_tokens, d_head]
        f_tail = self._reshape(f_tail)  # [batch_size * n_heads, n_kv_tokens, d_head]

        # Apply relation embedding (element-wise multiplication)
        f_head = f_head * self.rel_emb.view(self.n_heads, 1, -1).repeat(batch_size, 1, 1).view(-1, 1, d_value)
        f_tail = f_tail * self.rel_emb.view(self.n_heads, 1, -1).repeat(batch_size, 1, 1).view(-1, 1, d_value)

        # Compute edge weights: f_head @ f_tail^T
        weight_score = torch.bmm(f_head, f_tail.transpose(-2, -1))  # [batch_size * n_heads, n_q_tokens, n_kv_tokens]
        weight_score = weight_score / math.sqrt(d_value)

        # FR-Graph: Graph topology
        # Column embeddings for graph structure
        col_head = self.col_head  # [n_heads, n_cols, d_col]
        col_tail = self.col_tail  # [n_heads, n_cols, d_col]

        # Compute adjacency matrix based on column embeddings
        adj = torch.bmm(col_head, col_tail.transpose(-2, -1))  # [n_heads, n_cols, n_cols]
        
        # Apply no self-interaction constraint
        if self.nsi:
            mask = torch.eye(self.n_cols, device=adj.device, dtype=torch.bool)
            adj = adj.masked_fill(mask, 0)

        # Apply softmax to get normalized adjacency
        comp_func = F.softmax
        adj = comp_func(adj, dim=-1)  # [n_heads, n_cols, n_cols]

        # Expand adjacency matrix to match batch size and slice to match actual token dimensions
        adj = adj.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [batch_size, n_heads, n_cols, n_cols]
        # Slice adjacency matrix to match query and key token dimensions
        adj = adj[:, :, :n_q_tokens, :n_kv_tokens]  # [batch_size, n_heads, n_q_tokens, n_kv_tokens]
        adj = adj.view(batch_size * self.n_heads, n_q_tokens, n_kv_tokens)

        # graph assembling: apply FR-Graph on interaction like attention mask
        adj_mask = (1.0 - adj) * -10000 # analogous to attention mask

        # FR-Graph of this layer
        # Can be used for visualization on Feature Relation and Readout Collection
        fr_graph = comp_func(weight_score + adj_mask, dim=-1) # choose `softmax` as competitive function

        if self.dropout is not None:
            fr_graph = self.dropout(fr_graph)
        x = fr_graph @ self._reshape(f_v)
        x = (
            x.transpose(1, 2)
            .reshape(batch_size, n_head_nodes, self.n_heads * d_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x, fr_graph.detach()


class FRMETA(nn.Module):
    """FRM-ETA
    """
    def __init__(
        self,
        *,
        # tokenizer
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        token_bias: bool,
        # transformer
        n_layers: int,
        d_token: int,
        n_heads: int,
        d_ffn_factor: float,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        activation: str,
        prenormalization: bool,
        initialization: str,
        # linformer
        kv_compression: ty.Optional[float],
        kv_compression_sharing: ty.Optional[str],
        # graph estimator
        sym_weight: bool = True,
        sym_topology: bool = False,
        nsi: bool = True,
        #
        d_out: int,
    ) -> None:
        assert (kv_compression is None) ^ (kv_compression_sharing is not None)

        super().__init__()
        self.tokenizer = Tokenizer(d_numerical, categories, d_token, token_bias)
        n_tokens = self.tokenizer.n_tokens

        def make_kv_compression():
            assert kv_compression
            compression = nn.Linear(
                n_tokens, int(n_tokens * kv_compression), bias=False
            )
            if initialization == 'xavier':
                nn_init.xavier_uniform_(compression.weight)
            return compression

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression and kv_compression_sharing == 'layerwise'
            else None
        )

        def make_normalization():
            return nn.LayerNorm(d_token)

        n_tokens = d_numerical if categories is None else d_numerical + len(categories)
        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadGEAttention(
                        d_token, n_heads, attention_dropout, initialization,
                        n_tokens, sym_weight=sym_weight, sym_topology=sym_topology, nsi=nsi,
                    ),
                    'linear0': nn.Linear(
                        d_token, d_hidden * (2 if activation.endswith('glu') else 1)
                    ),
                    'linear1': nn.Linear(d_hidden, d_token),
                    'norm1': make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
            if kv_compression and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert kv_compression_sharing == 'key-value'
            self.layers.append(layer)

        self.activation = lib.get_activation_fn(activation)
        self.last_activation = lib.get_nonglu_activation_fn(activation)
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)

    def _get_kv_compressions(self, layer):
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer['key_compression'], layer['value_compression'])
            if 'key_compression' in layer and 'value_compression' in layer
            else (layer['key_compression'], layer['key_compression'])
            if 'key_compression' in layer
            else (None, None)
        )

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor], return_fr: bool = False) -> Tensor:
        fr_graphs = [] 
        x = self.tokenizer(x_num, x_cat)

        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)
            layer = ty.cast(ty.Dict[str, nn.Module], layer)

            x_residual = self._start_residual(x, layer, 0)
            x_residual, fr_graph = layer['attention'](
                # for the last attention, it is enough to process only [CLS]
                (x_residual[:, :1] if is_last_layer else x_residual),
                x_residual,
                *self._get_kv_compressions(layer),
            )
            fr_graphs.append(fr_graph)
            if is_last_layer:
                x = x[:, : x_residual.shape[1]]
            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)

        assert x.shape[1] == 1
        x = x[:, 0]
        if self.last_normalization is not None:
            x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        x = x.squeeze(-1)
        return x if not return_fr else (x, fr_graphs)

    def froze_topology(self):
        """API to froze FR-Graph topology in training"""
        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            layer['attention'].frozen = True
