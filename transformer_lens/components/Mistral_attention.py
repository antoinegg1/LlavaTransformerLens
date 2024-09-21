import math
from typing import Dict, Optional, Tuple, Union

import torch
import einops
import torch.nn as nn
from jaxtyping import Float, Int
import torch.nn.functional as F
from transformer_lens.utilities.attention import simple_attn_linear, complex_attn_linear
from transformer_lens.components.abstract_attention import AbstractAttention
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
import logger
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCacheEntry
# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class MistralRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    # copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.forward
    # TODO(joao): add me back asap :)
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    

class MistralAttention(AbstractAttention):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, cfg: Union[Dict, HookedTransformerConfig],attn_type: str = "global", layer_id: Optional[int] = None):
        super().__init__(cfg, attn_type, layer_id)
        
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        if layer_id is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.attn_type = attn_type

        # self.num_key_value_heads = config.num_key_value_heads
        # self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        # self.max_position_embeddings = config.max_position_embeddings
        # self.rope_theta = config.rope_theta
        # self.is_causal = True
        self.W_K = nn.Parameter(
                torch.empty(
                    self.cfg.n_key_value_heads,
                    self.cfg.d_model,
                    self.cfg.d_head,
                    dtype=self.cfg.dtype,
                )
            )
        self.W_V = nn.Parameter(
                torch.empty(
                    self.cfg.n_key_value_heads,
                    self.cfg.d_model,
                    self.cfg.d_head,
                    dtype=self.cfg.dtype,
                )
            )
        self.b_K = nn.Parameter(torch.zeros(self.cfg.n_key_value_heads, self.cfg.d_head, dtype=self.cfg.dtype))
        self.b_V = nn.Parameter(torch.zeros(self.cfg.n_key_value_heads, self.cfg.d_head, dtype=self.cfg.dtype))
        self.o_proj = nn.Linear(self.cfg.n_heads * self.cfg.d_head, self.cfg.d_model, bias=False)
        # self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        # self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.layer_id=layer_id
        self.rotary_emb = MistralRotaryEmbedding(
            self.cfg.d_head, max_position_embeddings=self.cfg.max_position_embeddings, base=self.cfg.rotary_base
        )
        self.hook_k = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_q = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_v = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_z = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_attn_scores = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_pattern = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_result = HookPoint()  # [batch, pos, head_index, d_model]
    # def apply_rotary(
    #     self,
    #     x: Float[torch.Tensor, "batch pos head_index d_head"],
    #     past_kv_pos_offset=0,
    #     attention_mask: Optional[Int[torch.Tensor, "batch offset_pos"]] = None,
    # ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    #     # Only apply rotary to first rotary_dim dimensions (eg, if rotary_dim=64 and d_head=256, only apply to first 1/4 of dimensions)
    #     x_pos = x.size(1)
    #     x_rot = x[..., : self.cfg.rotary_dim]
    #     x_pass = x[..., self.cfg.rotary_dim :]
    #     x_flip = self.rotate_every_two(x_rot)

    #     if attention_mask is None:
    #         rotary_cos = self.rotary_cos[
    #             None, past_kv_pos_offset : past_kv_pos_offset + x_pos, None, :
    #         ]
    #         rotary_sin = self.rotary_sin[
    #             None, past_kv_pos_offset : past_kv_pos_offset + x_pos, None, :
    #         ]
    #         x_rotated = x_rot * rotary_cos + x_flip * rotary_sin
    #     else:
    #         offset_position_ids = get_offset_position_ids(past_kv_pos_offset, attention_mask)
    #         offset_position_ids = offset_position_ids.to(self.rotary_cos.device)
    #         mask_rotary_cos = self.rotary_cos[offset_position_ids, None, :]
    #         mask_rotary_sin = self.rotary_sin[offset_position_ids, None, :]
    #         x_rotated = x_rot * mask_rotary_cos + x_flip * mask_rotary_sin

    #     return torch.cat([x_rotated, x_pass], dim=-1)

    def calculate_attention_scores(
        self,
        q: Float[torch.Tensor, "batch query_pos head_index d_head"],
        k: Float[torch.Tensor, "batch key_pos head_index d_head"],
    ) -> Float[torch.Tensor, "batch head_index query_pos key_pos"]:
        q_ = einops.rearrange(
            q, "batch query_pos head_index d_head -> batch head_index query_pos d_head"
        )
        k_ = einops.rearrange(
            k, "batch key_pos head_index d_head -> batch head_index d_head key_pos"
        )
        attn_scores = torch.matmul(q_,k_) /math.sqrt(self.cfg.d_head) 
        # if self.cfg.attn_scores_soft_cap > 0:
        #     attn_scores = self.cfg.attn_scores_soft_cap * F.tanh(
        #         attn_scores / self.cfg.attn_scores_soft_cap
        #     )
        return attn_scores
    def apply_causal_mask(
        self,
        attn_scores: Float[torch.Tensor, "batch head_index pos pos_plus_past_kv_pos_offset"],
        past_kv_pos_offset: int = 0,
        attention_mask: Optional[Int[torch.Tensor, "batch offset_pos"]] = None,
    ):
        # The query context length is the number of positions we take queries from - if not using a past_kv_cache this is just the context length (for the current prompt), but if we're caching it can be different.
        query_ctx_length = attn_scores.size(-2)
        # The key context length is the number of positions in the past - this includes all positions in the cache
        # If not caching, query_ctx_length == key_ctx_length
        key_ctx_length = attn_scores.size(-1)

        if query_ctx_length + past_kv_pos_offset != key_ctx_length:
            raise ValueError(
                f"query_ctx_length {query_ctx_length} + past_kv_pos_offset {past_kv_pos_offset} != key_ctx_length {key_ctx_length} - you likely have a bug."
            )
        self.mask = torch.triu(
            torch.ones(
                query_ctx_length,
                key_ctx_length,
                device=attn_scores.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        # Index back to front to ensure local attention works
        final_mask = self.mask[None, None, -query_ctx_length:, -key_ctx_length:]  # [1, 1, pos, pos]
        if attention_mask is not None:
            # Apply a causal mask to the attention scores considering the padding
            einsum_str = "batch head pos offset_pos, batch offset_pos -> batch head pos offset_pos"
            final_mask = final_mask.to(attention_mask.device)
            final_mask = einops.einsum(final_mask, attention_mask, einsum_str).bool()
        extreme_negative_value = torch.tensor(-3.4028e+38)
        attn_scores = attn_scores.to(final_mask.device)
        return torch.where(final_mask, attn_scores, extreme_negative_value)
    
    # def calculate_qkv_matrices(
    #     self,
    #     hidden_state,
    # ) -> Tuple[
    #     Float[torch.Tensor, "batch pos head_index d_head"],
    #     Float[torch.Tensor, "batch kv_pos head_index d_head"],
    #     Float[torch.Tensor, "batch kv_pos head_index d_head"],
    # ]:  
    #     # import pdb
    #     # print(value_input.shape, self.W_V.shape, self.b_V.shape)
    #     # pdb.set_trace()
    #     attn_fn = (
    #         complex_attn_linear
    #         if self.cfg.use_split_qkv_input or self.cfg.use_attn_in
    #         else simple_attn_linear
    #     )
        
    #     q = self.hook_q(attn_fn(hidden_state, self.W_Q, self.b_Q))
    #     k = self.hook_k(attn_fn(hidden_state, self.W_K, self.b_K))    
    #     v = self.hook_v(attn_fn(hidden_state, self.W_V, self.b_V))
        
    #     return q, k, v
    
    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, seq_len, d_model]
        position_ids: Optional[torch.Tensor] = None,      # [batch, seq_len]
        attention_mask: Optional[torch.Tensor] = None,    # [batch, seq_len]
        past_key_value: Optional[HookedTransformerKeyValueCacheEntry] = None,  # Cache entry
    ) -> torch.Tensor:


        attn_fn = (
            complex_attn_linear
            if self.cfg.use_split_qkv_input or self.cfg.use_attn_in
            else simple_attn_linear
        )
        
        q = self.hook_q(attn_fn(hidden_states, self.W_Q, self.b_Q))
        k = self.hook_k(attn_fn(hidden_states, self.W_K, self.b_K))
        v = self.hook_v(attn_fn(hidden_states, self.W_V, self.b_V))
        # import pdb
        # pdb.set_trace()
        q=q.transpose(1, 2).contiguous()
        k=k.transpose(1, 2).contiguous()
        v=v.transpose(1, 2).contiguous()
        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        if past_key_value is not None:
        # Append new keys and values to the cache and update the cache
            kv_cache_pos_offset = past_key_value.past_keys.size(1)
            updated_keys, updated_values = past_key_value.append(k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous())
            updated_keys = updated_keys.transpose(1, 2).contiguous()
            updated_values = updated_values.transpose(1, 2).contiguous()
        else:
            # Initialize past_key_value
            past_key_value = HookedTransformerKeyValueCacheEntry(
                past_keys=k.transpose(1, 2).contiguous(),
                past_values=v.transpose(1, 2).contiguous(),
                frozen=False
            )
            updated_keys, updated_values = k, v
            kv_cache_pos_offset = 0
            
        k = updated_keys
        v = updated_values
        # if self.cfg.positional_embedding_type == "rotary":
        #     q = self.hook_rot_q(self.apply_rotary(q, kv_cache_pos_offset, attention_mask))
        #     k = self.hook_rot_k(
        #         self.apply_rotary(k, 0, attention_mask)
        #     )  # keys are cached so no offset
        # q = einops.rearrange(
        #     q, "batch query_pos head_index d_head -> batch head_index query_pos d_head"
        # )
        

        if self.cfg.dtype not in [torch.float32, torch.float64]:
            # If using 16 bits, increase the precision to avoid numerical instabilities
            q = q.to(torch.float32)
            k = k.to(torch.float32)

        k=repeat_kv(k, self.cfg.n_heads//self.cfg.n_key_value_heads)
        v=repeat_kv(v, self.cfg.n_heads//self.cfg.n_key_value_heads)
        # import pdb
        # pdb.set_trace()

        # k_ = einops.rearrange(
        #     k, "batch key_pos head_index d_head -> batch head_index d_head key_pos"
        # )
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.cfg.d_head)  # [batch, if attention_mask is not None:
            # Expand attention_mask to match attn_scores dimensions
        # if attention_mask is not None:
        #     attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len_total]
        #     attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        attn_scores = self.apply_causal_mask(
                attn_scores, kv_cache_pos_offset, attention_mask
            )

        # Apply hooks to attention scores
        attn_scores = self.hook_attn_scores(attn_scores)

        # Compute attention probabilities (patterns)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = torch.where(torch.isnan(attn_probs), torch.zeros_like(attn_probs), attn_probs)
        attn_probs = self.hook_pattern(attn_probs)
        attn_probs = attn_probs.to(self.cfg.dtype)
        attn_probs = attn_probs.to(v.device)

        # Compute attention output
        attn_output = torch.matmul(attn_probs, v)  # [batch, num_heads, seq_len, d_head]
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch, seq_len, num_heads, d_head]
        attn_output = attn_output.reshape(
            attn_output.size(0), attn_output.size(1), self.cfg.d_model
        ).contiguous()  # [batch, seq_len, d_model]
        attn_output = self.hook_z(attn_output)

        # Apply the output projection
        out = self.o_proj(attn_output)  # [batch, seq_len, d_model]

        return out,past_key_value