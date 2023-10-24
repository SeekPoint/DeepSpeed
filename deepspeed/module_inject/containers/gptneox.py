# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .base import *
from .features.meta_tensor import MetaTensorContainer
from .features.hybrid_megatron import HybridMegatronContainer
from deepspeed.model_implementations.transformers.ds_gpt import DeepSpeedGPTInference
import torch
from ..policy import TransformerPolicy
from ..policy import transformer_param_names
from ..policy import maybe_copy
from packaging import version as pkg_version

from ..policy import maybe_get_lora
from pydebug import gd, infoTensor

class DS_GPTNEOXContainer(MetaTensorContainer, HybridMegatronContainer, BaseTransformerContainer):

    def __init__(self, **kwargs):
        gd.debuginfo(prj="ds")
        super().__init__(**kwargs)

        # All model specific things should be defined here instead of the base class.

    def create_module(self, config=None):
        gd.debuginfo(prj="ds")
        _config = config if config is not None else self.ds_model_config
        self.module = DeepSpeedGPTInference(_config, mp_group=self.mp_group)
        self.module.config.scale_attention = self.scale_attention

        if self.megatron_v2:
            gd.debuginfo(prj="ds")
            self.module.config.rotate_half = True
            self.module.config.rotate_every_two = False

        return self.module

    def get_lora_matched_pair(self):
        """
        Necessary to implement for `HybridEngineContainer`
        """
        gd.debuginfo(prj="ds")
        fc1_lora, fc2_lora, qkv_lora, out_lora = self.get_lora_params()
        ret = [(fc1_lora, self._h4h_w), (fc2_lora, self._4hh_w), (qkv_lora, self.qkvw), (out_lora, self.dense_w)]
        return ret

    def set_lora_params(self):
        """
        Necessary to implement for `HybridEngineContainer`
        """
        if GPTNEOXLayerPolicy.version == 0:
            gd.debuginfo(prj="ds")
            attention = self.policy.client_module.attention
        else:
            gd.debuginfo(prj="ds")
            attention = self.policy.client_module.self_attention

        self.lora_params = [
            maybe_get_lora(p) for p in [
                self.policy.client_module.mlp.dense_h_to_4h, self.policy.client_module.mlp.dense_4h_to_h,
                attention.query_key_value, attention.dense
            ]
        ]

    def load_params(self, module, sd, weight_quantizer, mp_replace, prefix):
        gd.debuginfo(prj="ds")
        param_names = (
            'attention.query_key_value.weight', \
            'attention.query_key_value.bias', \
            'attention.dense.weight', \
            'attention.dense.bias', \
            'mlp.dense_h_to_4h.weight', \
            'mlp.dense_h_to_4h.bias', \
            'mlp.dense_4h_to_h.weight', \
            'mlp.dense_4h_to_h.bias', \
            'post_attention_layernorm.weight', \
            'post_attention_layernorm.bias', \
            'input_layernorm.weight', \
            'input_layernorm.bias'
        )
        for i in range(0, 2):
            maybe_copy(module.attention,
                       sd,
                       weight_quantizer,
                       mp_replace,
                       transformer_param_names[i],
                       prefix + param_names[i],
                       qkv=True,
                       megatron_v2=self.policy.is_megatron_v2,
                       split_qkv=self.policy.split_qkv,
                       heads=self.policy.client_module.attention.num_attention_heads)
        for i in range(2, 4):
            maybe_copy(module.attention, sd, weight_quantizer, mp_replace, transformer_param_names[i],
                       prefix + param_names[i])
        for i in range(4, 10):
            maybe_copy(module.mlp, sd, weight_quantizer, mp_replace, transformer_param_names[i],
                       prefix + param_names[i])
        for i in range(10, 12):
            maybe_copy(module, sd, weight_quantizer, mp_replace, transformer_param_names[i], prefix + param_names[i])


class GPTNEOXLayerPolicy(TransformerPolicy):
    _orig_layer_class = None
    version = 0
    gd.debuginfo(prj="ds")

    def __init__(self, client_module, inference=True, megatron_v2=True, split_qkv=False):
        gd.debuginfo(prj="ds")
        super().__init__(inference, megatron_v2=megatron_v2, split_qkv=split_qkv)
        self.client_module = client_module
        if GPTNEOXLayerPolicy._orig_layer_class is None:
            if pkg_version.parse(torch.__version__) <= pkg_version.parse("1.2"):
                gd.debuginfo(prj="ds")
                GPTNEOXLayerPolicy._orig_layer_class = None
            else:
                try:
                    from transformers import GPTNeoXLayer
                    GPTNEOXLayerPolicy._orig_layer_class = GPTNeoXLayer
                    gd.debuginfo(prj="ds")
                except ImportError:
                    gd.debuginfo(prj="ds")
                    GPTNEOXLayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        if GPTNEOXLayerPolicy.version == 0:
            gd.debuginfo(prj="ds")
            attention = self.client_module.attention
        else:
            gd.debuginfo(prj="ds")
            attention = self.client_module.self_attention

        return self.client_module.attention.hidden_size, \
                self.client_module.attention.num_attention_heads, \
                self.client_module.input_layernorm.eps, \
                DEFAULT_INTERMEDIATE_SIZE

    def attention(self, enable_training=False):
        if GPTNEOXLayerPolicy.version == 0:
            gd.debuginfo(prj="ds")
            attention = self.client_module.attention
        else:
            gd.debuginfo(prj="ds")
            attention = self.client_module.self_attention

        return attention.query_key_value.weight, \
               attention.query_key_value.bias, \
               attention.dense.weight, \
               attention.dense.bias

    def mlp(self, enable_training=False):
        gd.debuginfo(prj="ds")
        return self.client_module.mlp.dense_h_to_4h.weight, \
               self.client_module.mlp.dense_h_to_4h.bias, \
               self.client_module.mlp.dense_4h_to_h.weight, \
               self.client_module.mlp.dense_4h_to_h.bias

    def layernorm(self):
        gd.debuginfo(prj="ds")
        return self.client_module.post_attention_layernorm.weight, \
               self.client_module.post_attention_layernorm.bias, \
               self.client_module.input_layernorm.weight, \
               self.client_module.input_layernorm.bias
