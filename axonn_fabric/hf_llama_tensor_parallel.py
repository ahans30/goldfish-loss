from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, ACT2FN
from axonn.intra_layer import Linear
from typing import Optional, Tuple, Union, List
from transformers.models.llama.modeling_llama import Cache, DynamicCache, StaticCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from axonn import axonn as ax
import torch
from torch import nn
import math

def modified_attention_init(self, config, layer_idx: Optional[int] = None):
    super(LlamaAttention, self).__init__()
    self.config = config
    self.layer_idx = layer_idx
    if layer_idx is None:
        logger.warning_once(  # noqa: F821
            f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "  # noqa: E501
            "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "  # noqa: E501
            "when creating this class."
        )

    self.attention_dropout = config.attention_dropout
    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.head_dim = self.hidden_size // self.num_heads
    self.num_key_value_heads = config.num_key_value_heads
    self.num_key_value_groups = self.num_heads // self.num_key_value_heads
    self.max_position_embeddings = config.max_position_embeddings
    self.rope_theta = config.rope_theta
    self.is_causal = True

    if (self.head_dim * self.num_heads) != self.hidden_size:
        raise ValueError(
            f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"  # noqa: E501
            f" and `num_heads`: {self.num_heads})."
        )

    self.q_proj = Linear(
        self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
    )
    self.k_proj = Linear(
        self.hidden_size,
        self.num_key_value_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.v_proj = Linear(
        self.hidden_size,
        self.num_key_value_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.o_proj = Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias, transpose=True)
    
    self.attention_parallel_world_size = ax.config.G_intra_r
    self._init_rope()

def modified_attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            raise NotImplementedError
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states, gather_output=False)
            key_states = self.k_proj(hidden_states, gather_output=False)
            value_states = self.v_proj(hidden_states, gather_output=False)
            ## kv states divided over row tensot parallel ranks

        query_states = query_states.view(bsz, q_len, self.num_heads // self.attention_parallel_world_size, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads // self.attention_parallel_world_size, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads // self.attention_parallel_world_size, self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads // self.attention_parallel_world_size, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads // self.attention_parallel_world_size, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size // self.attention_parallel_world_size)

        if self.config.pretraining_tp > 1:
            raise NotImplementedError
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output, scatter_input=False)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value



def modified_mlp_init(self, config):
    super(LlamaMLP, self).__init__()
    self.config = config
    self.hidden_size = config.hidden_size
    self.intermediate_size = config.intermediate_size
    self.gate_proj = Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.up_proj = Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.down_proj = Linear(self.intermediate_size, self.hidden_size, bias=False, transpose=True)
    self.act_fn = ACT2FN[config.hidden_act]

def modified_mlp_forward(self, x):
    if self.config.pretraining_tp > 1:
        raise NotImplementedError
        slice = self.intermediate_size // self.config.pretraining_tp
        gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
        up_proj_slices = self.up_proj.weight.split(slice, dim=0)
        down_proj_slices = self.down_proj.weight.split(slice, dim=1)

        gate_proj = torch.cat(
            [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
        )
        up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

        intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
        down_proj = [
            F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
        ]
        down_proj = sum(down_proj)
    else:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x, gather_output=False)) * self.up_proj(x, gather_output=False), scatter_input=False)

    return down_proj



def monkey_patch_llama_with_axonn():
    original_inits = LlamaAttention.__init__, LlamaMLP.__init__
    LlamaAttention.__init__ = modified_attention_init
    LlamaAttention.forward = modified_attention_forward
    LlamaMLP.__init__ = modified_mlp_init
    LlamaMLP.forward = modified_mlp_forward
    return original_inits


def reverse_monkey_patch_llama_with_axonn(original_attention_init, original_mlp_init):
    LlamaAttention.__init__ = original_attention_init
    LlamaMLP.__init__ = original_mlp_init
