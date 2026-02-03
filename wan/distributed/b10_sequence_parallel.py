# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import torch

from .sequence_parallel import rope_apply
from ..modules.model import sinusoidal_embedding_1d
from .util import gather_forward, get_rank, get_world_size
from ..modules.attention import flash_attention
from .util import b10_collect_tokens, b10_collect_heads, b10_collect_tokens_wait, b10_collect_heads_wait
import logging
from ..kernels import B10WanRope, b10_mult_and_add
from .b10_attn_cache import B10UNCONDCACHE, B10CONDNCACHE
from ..b10_config import enable_b10_kernel

def b10_sp_dit_forward_2_batch(self,
                               x,
                               t,
                               context,
                               context_null,
                               seq_len,
                               y=None,
                               clip_fea=None):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:   A list of text embeddings each with shape [L, C].
    context_null: A list of text embeddings each with shape [L, C].
    """
    assert len(x) == 1, "x should be a list of length 1"
    IF_USE_CFG_CACHE_THIS_STEP = B10UNCONDCACHE.if_use_cache_this_step()
    IF_USE_CFG_CACHE_NEXT_STEP = B10UNCONDCACHE.if_use_cache_next_step()
    sp_size, sp_rank = get_world_size(), get_rank()
    if self.model_type == 'i2v' or self.model_type == 'flf2v':
        assert y is not None
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) if u.size(1) < seq_len else u
        for u in x
    ])

    # time embeddings
    EXPAND_TIME_EMBEDDING = False
    if t.dim() == 1:
        EXPAND_TIME_EMBEDDING = True
    with torch.amp.autocast('cuda', dtype=torch.float32):
        bt = t.size(0)
        t = t.flatten()
        sinusoidal_embedding = sinusoidal_embedding_1d(self.freq_dim, t)
        e = self.time_embedding(sinusoidal_embedding.unflatten(0, (bt, 1 if EXPAND_TIME_EMBEDDING else seq_len)).float())
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32
    
    if EXPAND_TIME_EMBEDDING:
        e = e.expand(bt, seq_len // sp_size, self.dim)
        e0 = e0.expand(bt, seq_len // sp_size, 6, self.dim)
    else:
        e = torch.chunk(e, sp_size, dim=1)[sp_rank]
        e0 = torch.chunk(e0, sp_size, dim=1)[sp_rank]
        logging.warning(f"got {t.shape=}, this branch has low efficiency, please find why it is called")

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))
    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)
    if IF_USE_CFG_CACHE_THIS_STEP:
        context_null = None
    else:
        context_null = self.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context_null
            ]))
        if clip_fea is not None:
            context_null = torch.concat([context_clip, context_null], dim=1)

    # Context Parallel
    x = torch.chunk(x, get_world_size(), dim=1)[get_rank()]

    # arguments
    if enable_b10_kernel() and grid_sizes.shape[0] == 1:
        assert seq_lens.max() % sp_size == 0, f"seq_lens.max():{seq_lens.max()} calculated using ((F-1)//4 + 1) * (oh*ow) / 1024 must be divisible by sp_size:{sp_size}"
        seq_len_per_rank = seq_lens.max() // sp_size
        freqs = B10WanRope.preprocess_freqs(self.freqs, grid_sizes, sp_size, sp_rank, seq_len_per_rank)
    else:
        freqs = self.freqs
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=freqs,
        context=context,
        context_null=context_null,
        context_lens=context_lens)

    if IF_USE_CFG_CACHE_THIS_STEP:
        x_null = None
        for block_id, block in enumerate(self.blocks):
            B10CONDNCACHE.current_layer_id = block_id
            x, _ = block(x=x, x_null=x_null, **kwargs)
        x = self.head(x, e)
        x = gather_forward(x, dim=1)
        x = self.unpatchify(x, grid_sizes)
        x_null = [B10UNCONDCACHE.get_cached_output(x[0])]
        return [u.float() for u in x], [u.float() for u in x_null]
    else:
        x_null = x
        for block_id, block in enumerate(self.blocks):
            B10CONDNCACHE.current_layer_id = block_id
            x, x_null = block(x=x, x_null=x_null, **kwargs)
        # head
        x = self.head(x, e)
        x_null = self.head(x_null, e)

        # Context Parallel
        x = gather_forward(x, dim=1)
        x_null = gather_forward(x_null, dim=1)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        x_null = self.unpatchify(x_null, grid_sizes)
        if IF_USE_CFG_CACHE_NEXT_STEP:
            B10UNCONDCACHE.set_cached_output(x[0], x_null[0])
        return [u.float() for u in x], [u.float() for u in x_null]

def b10_sp_block_forward_2_batch(self, x, x_null, e, seq_lens, grid_sizes, freqs, context, context_null, context_lens):
    r"""
    Args:
        x(Tensor): Shape [B, L, C]
        x_null(Tensor): Shape [B, L, C]
        e(Tensor): Shape [B, L1, 6, C]

        seq_lens(Tensor): Shape [B], length of each sequence in batch
    """
    assert e.dtype == torch.float32
    e = e.chunk(6, dim=2) # [B, L1, 1, C]
    modulation = self.modulation.chunk(6, dim=1)
    assert e[0].dtype == torch.float32

    IF_USE_ATTN_CACHE_THIS_STEP = B10CONDNCACHE.if_use_cache_this_step()

    # self-attention
    self.norm1.time_weight = e[1].squeeze(2)
    self.norm1.time_bias = e[0].squeeze(2)
    self.norm1.weight_modulation = modulation[1]
    self.norm1.bias_modulation = modulation[0]
    self.norm1.elementwise_affine = True
    self.norm1.use_time_embed_as_affine = True
    self.norm1.save_fp32 = False
    if not IF_USE_ATTN_CACHE_THIS_STEP:
        x_norm = self.norm1(x)
        q, k, v, q_handle, k_handle, v_handle = b10_sp_attn_pre_forward(self.self_attn, x_norm, grid_sizes, freqs, prefix="cond")
        
    if x_null is not None:
        x_null_norm = self.norm1(x_null)
        q_null, k_null, v_null, q_handle_null, k_handle_null, v_handle_null = b10_sp_attn_pre_forward(self.self_attn, x_null_norm, grid_sizes, freqs, prefix="uncond")
    if not IF_USE_ATTN_CACHE_THIS_STEP:
        y, y_handle = b10_sp_attn_forward_core(self.self_attn, q, k, v, q_handle, k_handle, v_handle, seq_lens, prefix="cond")
    if x_null is not None:
        y_null, y_handle_null = b10_sp_attn_forward_core(self.self_attn, q_null, k_null, v_null, q_handle_null, k_handle_null, v_handle_null, seq_lens, prefix="uncond")
    del self.norm1.time_weight, self.norm1.time_bias
    if not IF_USE_ATTN_CACHE_THIS_STEP:
        y = b10_sp_attn_post_forward(self.self_attn, y, y_handle, prefix="cond")
        x = b10_mult_and_add(
            x=y, 
            w1d=modulation[2],
            w2d=e[2],
            b1d=None,
            b2d=x,
            SAVE_FP32=False
        )
        B10CONDNCACHE.set_cached_output(x)
    else:
        x = B10CONDNCACHE.get_cached_output()


    # cross-attention & ffn function
    @torch.profiler.record_function("cross_attn_ffn")
    def cross_attn_ffn(x, context, context_lens, e):
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        self.norm2.time_weight = e[4].squeeze(2)
        self.norm2.time_bias = e[3].squeeze(2)
        self.norm2.weight_modulation = modulation[4]
        self.norm2.bias_modulation = modulation[3]
        self.norm2.elementwise_affine = True
        self.norm2.use_time_embed_as_affine = True
        self.norm2.save_fp32 = False
        x_norm = self.norm2(x)
        y = self.ffn(x_norm)
        del self.norm2.time_weight, self.norm2.time_bias
        return b10_mult_and_add(
            x=y, 
            w1d=modulation[5],
            w2d=e[5],
            b1d=None,
            b2d=x,
            SAVE_FP32=False
        )

    x = cross_attn_ffn(x, context, context_lens, e)
    if x_null is not None:
        y_null = b10_sp_attn_post_forward(self.self_attn, y_null, y_handle_null, prefix="uncond")
        x_null = b10_mult_and_add(
            x=y_null, 
            w1d=modulation[2],
            w2d=e[2],
            b1d=None,
            b2d=x_null,
            SAVE_FP32=False
        )
        x_null = cross_attn_ffn(x_null, context_null, context_lens, e)
    return x, x_null

def b10_sp_attn_forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16, prefix="cond"):
    q, k, v, q_handle, k_handle, v_handle = b10_sp_attn_pre_forward(self, x, grid_sizes, freqs, dtype, prefix)
    x, x_handle = b10_sp_attn_forward_core(self, q, k, v, q_handle, k_handle, v_handle, seq_lens, prefix)
    x = b10_sp_attn_post_forward(self, x, x_handle, prefix)
    return x

def b10_sp_attn_pre_forward(self, x, grid_sizes, freqs, dtype=torch.bfloat16, prefix=""):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)
    
    v = self.v(x).view(b, s, n, d)
    v = half(v)
    v, v_handle = b10_collect_tokens(v, group_name=prefix+"v", launch_barrier=False, reuse_streams_group=None)

    q = self.norm_q(self.q(x)).view(b, s, n, d)
    if len(grid_sizes) == 1:
        # logging.info(f"q.shape: {q.shape}, freqs.shape: {freqs.shape}")
        q = B10WanRope.b10_rope_apply(q, freqs, save_fp16=True)
    else:
        q = rope_apply(q, grid_sizes, freqs)
        q = half(q)
    q, q_handle = b10_collect_tokens(q, group_name=prefix+"q", launch_barrier=False, reuse_streams_group=prefix+"v")

    k = self.norm_k(self.k(x)).view(b, s, n, d)
    if grid_sizes.shape[0] == 1:
        # logging.info(f"k.shape: {k.shape}, freqs.shape: {freqs.shape}")
        k = B10WanRope.b10_rope_apply(k, freqs, save_fp16=True)
    else:
        k = rope_apply(k, grid_sizes, freqs)
        k = half(k)
    k, k_handle = b10_collect_tokens(k, group_name=prefix+"k", launch_barrier=True, reuse_streams_group=prefix+"v")
    return q, k, v, q_handle, k_handle, v_handle

def b10_sp_attn_forward_core(self, q, k, v, q_handle, k_handle, v_handle, seq_lens, prefix=""):
    # apply attention
    q = b10_collect_tokens_wait(q, q_handle)
    k = b10_collect_tokens_wait(k, k_handle)
    v = b10_collect_tokens_wait(v, v_handle)
    x = flash_attention(
        q,
        k,
        v,
        k_lens=seq_lens,
        window_size=self.window_size,
    )
    x, x_handle = b10_collect_heads(x, group_name=prefix+"k", launch_barrier=True, reuse_streams_group=prefix+"v")
    return x, x_handle

def b10_sp_attn_post_forward(self, x, x_handle, prefix="cond"):
    x = b10_collect_heads_wait(x, x_handle)
    x = x.flatten(2)
    x = self.o(x)
    return x
