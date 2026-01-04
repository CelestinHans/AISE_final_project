"""
This file contains the implementation of the attention module.

Reference: https://github.com/meta-llama/llama3/blob/main/llama/model.py
"""
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass, asdict, field
from omegaconf import OmegaConf
from rotary_embedding_torch import RotaryEmbedding
from .mlp import ConditionedNorm
from .utils.dataclass import shallow_asdict
import math
import os

def _set_save_attn(model, enabled: bool):
    """
    Active/désactive la capture des matrices d'attention pour toutes les couches du Transformer.
    Adapte si ton chemin n'est pas exactement model.processor.*
    """
    proc = model.processor  # dans GAOT c'est self.processor

    for blk in proc.encoder_layers:
        blk.attn.save_attn = enabled
        if enabled:
            blk.attn.last_attn = None

    if proc.middle_layer is not None:
        proc.middle_layer.attn.save_attn = enabled
        if enabled:
            proc.middle_layer.attn.last_attn = None

    for blk in proc.decoder_layers:
        blk.attn.save_attn = enabled
        if enabled:
            blk.attn.last_attn = None


def _dump_attn_pngs(model, out_dir: str, prefix: str, save_attn_heatmap):
    os.makedirs(out_dir, exist_ok=True)
    proc = model.processor

    # encoder
    for li, blk in enumerate(proc.encoder_layers):
        A = blk.attn.last_attn
        if A is not None:
            save_attn_heatmap(A, os.path.join(out_dir, f"{prefix}_enc{li}.png"),
                              title=f"{prefix} | Encoder layer {li}")

    # middle
    if proc.middle_layer is not None and proc.middle_layer.attn.last_attn is not None:
        save_attn_heatmap(proc.middle_layer.attn.last_attn,
                          os.path.join(out_dir, f"{prefix}_mid.png"),
                          title=f"{prefix} | Middle layer")

    # decoder
    for li, blk in enumerate(proc.decoder_layers):
        A = blk.attn.last_attn
        if A is not None:
            save_attn_heatmap(A, os.path.join(out_dir, f"{prefix}_dec{li}.png"),
                              title=f"{prefix} | Decoder layer {li}")


############
# Config
############
@dataclass
class AttentionConfig:
    num_heads: int = 8                      # Number of attention heads (for multi-head attention)
    num_kv_heads: int = 8                   # Number of attention heads for Key and Value (Grouped Query Attention)
    use_conditional_norm: bool = False      # Whether to use time conditional normalization
    cond_norm_hidden_size: int = 4          # Hidden size for the time conditional normalization
    atten_dropout: float = 0.0              # Dropout probability in the attention module

@dataclass
class TransformerConfig:
    patch_size: int = 8                              # Size of the patches for the structured latent tokens
    hidden_size: int = 256                           # Hidden size of the transformer
    use_attn_norm: bool = True                       # Whether to use normalization in the attention module
    use_ffn_norm: bool = True                        # Whether to use normalization in the feedforward network
    norm_eps: float = 1e-6                           # Epsilon value for layer normalization
    num_layers: int = 3                              # Number of transformer blocks
    positional_embedding: str = 'absolute'           # Positional embedding type, supports ['absolute', 'rope']
    use_long_range_skip: bool = True                 # Set it to True for UViT processor
    ffn_multiplier: int = 4                          # FFN hidden size multiplier (ffn_hidden = hidden_size * ffn_multiplier)
    attn_config: AttentionConfig = field(default_factory=AttentionConfig)   # Configuration for the attention sub-module


    use_rel_dist_bias: bool = False
    rel_bias_hidden_size: int = 64


    use_cross_attention: bool = False
    num_seed_tokens: int = 64

############
# Attention
############
class DistanceRelativeBias(nn.Module):
    """
    Continuous relative bias based on Euclidean distance ||xi-xj||.
    Returns bias shaped [B, H, N, N] to be added to attention logits.
    """
    def __init__(self, num_heads: int, hidden_size: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, num_heads),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [B, N, D]
        d = torch.cdist(coords, coords)                 # [B, N, N]
        b = self.mlp(d.unsqueeze(-1))                   # [B, N, N, H]
        return b.permute(0, 3, 1, 2).contiguous()       # [B, H, N, N]




class GroupQueryFlashAttention(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int = 8,
        num_kv_heads: int = 8,
        use_conditional_norm: bool = False,
        cond_norm_hidden_size: int = 4,
        atten_dropout: float = 0.0,
        positional_embedding: str = "absolute",
        use_rel_dist_bias: bool = False,
        rel_bias_hidden_size: int = 64,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_repeat = num_heads // num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.atten_dropout = atten_dropout

        kv_hidden_size = self.head_dim * self.num_kv_heads

        self.q_proj = nn.Linear(input_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(input_size, kv_hidden_size, bias=False)
        self.v_proj = nn.Linear(input_size, kv_hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, input_size, bias=False)

        self.correction = ConditionedNorm(1, input_size, cond_norm_hidden_size) if use_conditional_norm else None

        self.positional_embedding = positional_embedding
        if positional_embedding == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.head_dim)

        self.use_rel_dist_bias = use_rel_dist_bias
        self.rel_bias = DistanceRelativeBias(num_heads, rel_bias_hidden_size) if use_rel_dist_bias else None

        self.save_attn = False
        self.last_attn = None



    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[float] = None,
        relative_positions: Optional[torch.Tensor] = None,
        token_coords: Optional[torch.Tensor] = None,   # NEW
    ) -> torch.Tensor:

        if self.correction is not None:
            x = self.correction(c=condition, x=x)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        B, N, _ = q.size()

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,N,hd]
        k = k.view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B,Hkv,N,hd]
        v = v.view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_repeat, dim=1)  # [B,H,N,hd]
            v = v.repeat_interleave(self.num_repeat, dim=1)

###
        if relative_positions is not None:
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        if self.save_attn:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,H,L,L]
            attn = torch.softmax(scores, dim=-1)

            # self.last_attn = attn.detach().mean(dim=1).squeeze(0).float().cpu()

            attn_mean = attn.detach().mean(dim=1)      # [B, N, N]  (mean over heads)
            attn_mean = attn_mean.mean(dim=0)          # [N, N]     (mean over batch)
            self.last_attn = attn_mean.float().cpu()

            x = torch.matmul(attn, v)  # [B,H,L,Dh]
        else:
            dp = self.atten_dropout if self.training else 0.0
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dp)  # [B,H,L,Dh]


        x = x.transpose(1, 2).contiguous().view(B, N, -1)  # FIX HERE
        x = self.o_proj(x)
        return x

###








        # if relative_positions is not None and self.positional_embedding == "rope":
        #     q = self.rotary_emb.rotate_queries_or_keys(q)
        #     k = self.rotary_emb.rotate_queries_or_keys(k)

        # dp = self.atten_dropout if self.training else 0.0


        # use_bias = (self.use_rel_dist_bias and token_coords is not None)
        # if use_bias:
        #     if token_coords.dim() == 2:
        #         token_coords = token_coords.unsqueeze(0).expand(B, -1, -1)  # [B,N,D]
        #     bias = self.rel_bias(token_coords)  # [B,H,N,N]

        #     # Manual attention (logits + bias)
        #     scale = (self.head_dim ** -0.5)
        #     attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B,H,N,N]
        #     attn_logits = attn_logits + bias

        #     attn = torch.softmax(attn_logits, dim=-1)
        #     if dp > 0:
        #         attn = torch.dropout(attn, p=dp, train=True)

        #     out = torch.matmul(attn, v)  # [B,H,N,hd]
        # else:
        #     out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dp)

        # out = out.transpose(1, 2).contiguous().view(B, N, -1)  # [B,N,H*hd]
        # out = self.o_proj(out)
        # return out

    @classmethod
    def from_config(
        cls,
        input_size: int,
        hidden_size: int,
        config: AttentionConfig,
        positional_embedding: str = "absolute",
        use_rel_dist_bias: bool = False,
        rel_bias_hidden_size: int = 64,
    ):
        return cls(
            input_size=input_size,
            hidden_size=hidden_size,
            positional_embedding=positional_embedding,
            use_rel_dist_bias=use_rel_dist_bias,
            rel_bias_hidden_size=rel_bias_hidden_size,
            **shallow_asdict(config),
        )


class CrossAttention(nn.Module):
    """
    Cross-attention: queries attend to key/value.
    Shapes:
      q_in: [B, Nq, D], kv_in: [B, Nk, D]
    """
    def __init__(self, dim: int, hidden_size: int, num_heads: int, dropout: float = 0.0,
                 use_rel_dist_bias: bool = False, rel_bias_hidden_size: int = 64):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(dim, hidden_size, bias=False)
        self.k_proj = nn.Linear(dim, hidden_size, bias=False)
        self.v_proj = nn.Linear(dim, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, dim, bias=False)

        self.use_rel_dist_bias = use_rel_dist_bias
        self.rel_bias = DistanceRelativeBias(num_heads, rel_bias_hidden_size) if use_rel_dist_bias else None

    def forward(self, q_in, kv_in, q_coords=None, kv_coords=None):
        B, Nq, D = q_in.shape
        Nk = kv_in.shape[1]

        q = self.q_proj(q_in).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)   # [B,H,Nq,hd]
        k = self.k_proj(kv_in).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,Nk,hd]
        v = self.v_proj(kv_in).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        dp = self.dropout if self.training else 0.0

        use_bias = self.use_rel_dist_bias and (q_coords is not None) and (kv_coords is not None)
        if use_bias:
            if q_coords.dim() == 2:
                q_coords = q_coords.unsqueeze(0).expand(B, -1, -1)
            if kv_coords.dim() == 2:
                kv_coords = kv_coords.unsqueeze(0).expand(B, -1, -1)

            # bias: [B,H,Nq,Nk] from distances between q and kv
            d = torch.cdist(q_coords, kv_coords)  # [B,Nq,Nk]
            b = self.rel_bias.mlp(d.unsqueeze(-1)).permute(0, 3, 1, 2).contiguous()  # [B,H,Nq,Nk]

            scale = (self.head_dim ** -0.5)
            logits = torch.matmul(q, k.transpose(-2, -1)) * scale + b
            attn = torch.softmax(logits, dim=-1)
            if dp > 0:
                attn = torch.dropout(attn, p=dp, train=True)
            out = torch.matmul(attn, v)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dp)

        out = out.transpose(1, 2).contiguous().view(B, Nq, -1)
        return self.o_proj(out)

############
# Feedforward Network
############
class FFN(nn.Module):
    def __init__(self,
                input_size: int, 
                ffn_hidden_size: int,  # Directly specify FFN hidden size
                use_conditional_norm: bool = False, 
                cond_norm_hidden_size: int = 4
                ):
        super().__init__()
        self.w1 = nn.Linear(input_size, ffn_hidden_size, bias=False)
        self.w2 = nn.Linear(ffn_hidden_size, input_size, bias=False)
        self.w3 = nn.Linear(input_size, ffn_hidden_size, bias=False)

        if use_conditional_norm:
            self.correction = ConditionedNorm(1, input_size, cond_norm_hidden_size)
        else:
            self.correction = None

    def forward(self, x, condition: Optional[float] = None):
        x = self.w2(F.silu(self.w1(x)) * self.w3(x))

        if self.correction is not None:
            x = self.correction(c=condition, x=x)

        return x

############
# Normalization
############
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

############
# Transformer Block
############
class TransformerBlock(nn.Module):
    def __init__(self, 
                input_size: int, 
                config: TransformerConfig,
                skip_connection: bool = False
                ):
        super().__init__()
        hidden_size = config.hidden_size
        ffn_hidden_size = hidden_size * config.ffn_multiplier
        
        self.attn = GroupQueryFlashAttention.from_config(
            input_size=input_size, 
            hidden_size=hidden_size,
            config=config.attn_config,
            positional_embedding=config.positional_embedding
        )
        
        self.ffn = FFN(
            input_size=input_size, 
            ffn_hidden_size=ffn_hidden_size,
            use_conditional_norm=config.attn_config.use_conditional_norm,
            cond_norm_hidden_size=config.attn_config.cond_norm_hidden_size
        )

        self.attn_norm = RMSNorm(input_size, eps=config.norm_eps) if config.use_attn_norm else None 
        self.ffn_norm = RMSNorm(input_size, eps=config.norm_eps) if config.use_ffn_norm else None 

        self.skip_connection = skip_connection
        if self.skip_connection:
            self.skip_proj = nn.Linear(input_size * 2, input_size)
            
    def forward(self, x, condition=None, relative_positions=None, token_coords=None, skip=None):
        if self.skip_connection and skip is not None:
            x = torch.cat([x, skip], dim=-1)
            x = self.skip_proj(x)

        h = x if self.attn_norm is None else self.attn_norm(x)
        h = x + self.attn(h, condition=condition, relative_positions=relative_positions, token_coords=token_coords)
        h = h if self.ffn_norm is None else self.ffn_norm(h)
        out = h + self.ffn(h, condition=condition)
        return out






############
# Transformer
############
class Transformer(nn.Module):
    def __init__(self, 
                input_size: int, 
                output_size: int, 
                config: TransformerConfig = TransformerConfig()
                ):
        super().__init__()
        hidden_size = config.hidden_size
        num_layers = config.num_layers
        self.use_long_range_skip = config.use_long_range_skip


        if input_size != hidden_size:
            self.input_proj = nn.Linear(input_size, hidden_size)
            working_size = hidden_size
        else:
            self.input_proj = nn.Identity()
            working_size = input_size

        if working_size != output_size:
            self.output_proj = nn.Linear(working_size, output_size)
        else:
            self.output_proj = nn.Identity()

        num_encoder_layers = num_layers // 2
        num_decoder_layers = num_layers // 2
        middle_layer_exists = (num_layers % 2 == 1)

        self.encoder_layers = nn.ModuleList([
            TransformerBlock(
                input_size=working_size,
                config=config,
                skip_connection=False
            ) for _ in range(num_encoder_layers)
        ])

        self.middle_layer = None
        if middle_layer_exists:
            self.middle_layer = TransformerBlock(
                input_size=working_size,
                config=config,
                skip_connection=False
            )

        self.decoder_layers = nn.ModuleList([
            TransformerBlock(
                input_size=working_size,
                config=config,
                skip_connection=True
            ) for _ in range(num_decoder_layers)
        ])


        self.use_cross_attention = config.use_cross_attention
        self.num_seed_tokens = config.num_seed_tokens

        if self.use_cross_attention:
            self.seed_tokens = nn.Parameter(torch.randn(self.num_seed_tokens, working_size) * 0.02)

            # cross in/out
            self.cross_in = CrossAttention(
                dim=working_size,
                hidden_size=config.hidden_size,
                num_heads=config.attn_config.num_heads,
                dropout=config.attn_config.atten_dropout,
                use_rel_dist_bias=config.use_rel_dist_bias,
                rel_bias_hidden_size=config.rel_bias_hidden_size,
            )
            self.cross_out = CrossAttention(
                dim=working_size,
                hidden_size=config.hidden_size,
                num_heads=config.attn_config.num_heads,
                dropout=config.attn_config.atten_dropout,
                use_rel_dist_bias=False,  # souvent inutile au retour, tu peux activer si tu veux
                rel_bias_hidden_size=config.rel_bias_hidden_size,
            )

    def forward(self, x, condition=None, relative_positions=None, token_coords=None):
        x = self.input_proj(x)

        B, N, D = x.shape

        if self.use_cross_attention:
            seeds = self.seed_tokens.unsqueeze(0).expand(B, -1, -1)  # [B,M,D]

            # Cross-attn: seeds attend to tokens
            seeds = self.cross_in(
                q_in=seeds,
                kv_in=x,
                q_coords=None,            # option: coords apprises pour seeds si tu veux
                kv_coords=token_coords,   # coords des tokens irréguliers
            )

            # Run U-shaped transformer blocks on the SMALL sequence (seeds)
            skips = []
            for layer in self.encoder_layers:
                seeds = layer(seeds, condition=condition, relative_positions=None, token_coords=None)
                skips.append(seeds)

            if self.middle_layer is not None:
                seeds = self.middle_layer(seeds, condition=condition, relative_positions=None, token_coords=None)

            for layer in self.decoder_layers:
                skip = skips.pop() if self.use_long_range_skip else None
                seeds = layer(seeds, condition=condition, relative_positions=None, token_coords=None, skip=skip)

            # Cross-attn: tokens attend to seeds (project back to N)
            x = self.cross_out(q_in=x, kv_in=seeds)

            x = self.output_proj(x)
            return x






        skips = []

        for layer in self.encoder_layers:
            x = layer(x, condition=condition, relative_positions=relative_positions, token_coords=token_coords)
            skips.append(x)

        if self.middle_layer is not None:
            x = self.middle_layer(x, condition=condition, relative_positions=relative_positions, token_coords=token_coords)

        for layer in self.decoder_layers:
            skip = skips.pop() if self.use_long_range_skip else None
            x = layer(x, condition=condition, relative_positions=relative_positions, token_coords=token_coords, skip=skip)

        x = self.output_proj(x)
        return x
