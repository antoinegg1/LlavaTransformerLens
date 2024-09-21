"""Hooked Transformer Transformer Block Component.

This module contains all the component :class:`TransformerBlock`.
"""

from typing import Callable, Dict, Optional, Union

import torch
import torch.nn as nn
from jaxtyping import Float, Int

from transformer_lens.components import (
    Attention,
    MistralAttention,
    GroupedQueryAttention,
    LayerNorm,
    LayerNormPre,
    RMSNorm,
    RMSNormPre,
    TransformerBlock,
    MistralRMSNorm,
)
from transformer_lens.components.mlps.can_be_used_as_mlp import CanBeUsedAsMLP
from transformer_lens.factories.mlp_factory import MLPFactory
from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCacheEntry
from transformer_lens.utils import repeat_along_head_dimension


# Transformer Block
class MistralBlock(TransformerBlock):

    def __init__(self, cfg: Union[Dict, HookedTransformerConfig], block_index):
        super().__init__(cfg, block_index)
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        normalization_layer: Callable  # type: ignore
        normalization_layer_after: Callable  # type: ignore

        self.normalization_type = self.cfg.normalization_type

        if self.normalization_type == "LN":
            normalization_layer = LayerNorm
        elif self.normalization_type == "LNPre":
            # We've folded in LayerNorm weights, so just need the center + scale parts
            normalization_layer = LayerNormPre
        elif self.normalization_type == "RMS":
            normalization_layer = RMSNorm
        elif self.normalization_type == "RMSPre":
            normalization_layer = RMSNormPre
        elif self.normalization_type == "MistralRMSNorm":
            normalization_layer = MistralRMSNorm
        elif self.normalization_type is None:
            # This should just be the identity.
            # We need to make this a lambda so we can call it on the config, just like the others
            normalization_layer = lambda cfg: nn.Identity()
        else:
            raise ValueError(f"Invalid normalization_type passed in: {self.normalization_type}")

        if self.cfg.use_normalization_before_and_after:
            # If we use LN before and after, we do *not* fold in the weights to the LN
            # after, though we can fold for the one before.
            if self.normalization_type is None:
                normalization_layer_after = lambda cfg: nn.Identity()
            elif self.normalization_type.startswith("RMS"):
                normalization_layer_after = RMSNorm
            elif self.normalization_type.startswith("LayerNorm"):
                normalization_layer_after = LayerNorm
            elif self.normalization_type.startswith("MistralRMSNorm"):
                normalization_layer_after = MistralRMSNorm
        self.ln1 = normalization_layer(cfg)
        if self.cfg.use_normalization_before_and_after:
            self.ln1_post = normalization_layer_after(cfg)
        if not self.cfg.attn_only:
            self.ln2 = normalization_layer(cfg)
            if self.cfg.use_normalization_before_and_after:
                self.ln2_post = normalization_layer_after(cfg)

        attention = MistralAttention
        if not self.cfg.use_local_attn:
            self.attn = attention(self.cfg, "global", block_index)
        else:
            if self.cfg.attn_types is None:
                raise ValueError("attn_types must be set when using local attention")
            attn_type = self.cfg.attn_types[block_index]
            self.attn = attention(self.cfg, attn_type, block_index)
        if not self.cfg.attn_only:
            self.mlp = MLPFactory.create_mlp(self.cfg)
        self.updated_past_key_value=None
        self.hook_attn_in = HookPoint()  # [batch, pos, n_heads, d_model]
        self.hook_q_input = HookPoint()  # [batch, pos, n_heads, d_model]
        self.hook_k_input = HookPoint()  # [batch, pos, n_heads, d_model]
        self.hook_v_input = HookPoint()  # [batch, pos, n_heads, d_model]
        self.hook_mlp_in = HookPoint()  # [batch, pos, d_model]

        self.hook_attn_out = HookPoint()  # [batch, pos, d_model]
        self.hook_mlp_out = HookPoint()  # [batch, pos, d_model]

        self.hook_resid_pre = HookPoint()  # [batch, pos, d_model]
        if not self.cfg.attn_only and not self.cfg.parallel_attn_mlp:
            self.hook_resid_mid = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_post = HookPoint()  # [batch, pos, d_model]
        
    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, seq_len, d_model]
        position_ids: Optional[torch.Tensor] = None,  # [batch, seq_len]
        attention_mask: Optional[torch.Tensor] = None,  # [batch, seq_len]
        past_key_value: Optional[HookedTransformerKeyValueCacheEntry] = None,  # Cache entry for past key/value states
    ) -> torch.Tensor:  # [batch, seq_len, d_model]
        # hidden_states = block(
        #         hidden_states,
        #         attention_mask=attention_mask,
        #         past_key_value=past_key_value,
        #     )
        hidden_states = self.hook_resid_pre(hidden_states)  # [batch, pos, d_model]

        # if self.cfg.use_attn_in or self.cfg.use_split_qkv_input:
        #     # We're adding a head dimension
        #     if shortformer_pos_embed is not None:
        #         shortformer_pos_embed = repeat_along_head_dimension(
        #             shortformer_pos_embed, n_heads=self.cfg.n_heads
        #         )
        # else:
        #     attn_in = hidden_states

        # if self.cfg.use_attn_in:
        #     attn_in = self.hook_attn_in(
        #         repeat_along_head_dimension(resid_pre, n_heads=self.cfg.n_heads)
        #     )

        # if self.cfg.use_split_qkv_input:
        #     n_kv_heads = (
        #         self.cfg.n_key_value_heads
        #         if self.cfg.n_key_value_heads is not None
        #         else self.cfg.n_heads
        #     )
        #     query_input = self.hook_q_input(
        #         repeat_along_head_dimension(resid_pre, n_heads=self.cfg.n_heads)
        #     )
        #     key_input = self.hook_k_input(
        #         repeat_along_head_dimension(resid_pre, n_heads=n_kv_heads)
        #     )
        #     value_input = self.hook_v_input(
        #         repeat_along_head_dimension(resid_pre, n_heads=n_kv_heads)
        #     )
        # else:
        #     query_input = attn_in
        #     key_input = attn_in
        #     value_input = attn_in
        # import pdb; pdb.set_trace()
        normed_hidden_states = self.ln1(hidden_states)
        attn_out, updated_past_key_value = self.attn(
            hidden_states=normed_hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
        )  # [batch, seq_len, d_model]
        self.updated_past_key_value = updated_past_key_value
        # attn_out = (
        #     # hook the residual stream states that are used to calculate the
        #     # queries, keys and values, independently.
        #     # Then take the layer norm of these inputs, and pass these to the attention module.
        #     self.attn(
        #         hidden_state,
        #         past_kv_cache_entry=past_kv_cache_entry,
        #         attention_mask=attention_mask,
        #     )
        # )  # [batch, pos, d_model]
        if self.cfg.use_normalization_before_and_after:
            # If we use LayerNorm both before and after, then apply the second LN after the layer
            # and before the hook. We do it before the hook so hook_attn_out captures "that which
            # is added to the residual stream"
            attn_out = self.ln1_post(attn_out)
        attn_out = self.hook_attn_out(attn_out)
        resid_mid = self.hook_resid_mid(hidden_states + attn_out)
        if not self.cfg.attn_only:
            normed_resid_mid = self.ln2(resid_mid)
            mlp_out = self.apply_mlp(normed_resid_mid)  # [batch, seq_len, d_model]
            resid_post = self.hook_resid_post(resid_mid + mlp_out)  # [batch, seq_len, d_model]
        else:
            resid_post = self.hook_resid_post(resid_mid)  # [batch, seq_len, d_model]

        return resid_post  # [batch, pos, d_model]
