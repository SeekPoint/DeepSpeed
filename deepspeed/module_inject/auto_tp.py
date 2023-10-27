# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Automatic Tensor Parallelism
import re

from torch import nn
from .replace_policy import replace_policies
from pydebug import gd, infoTensor

class AutoTP():

    def in_module_list(module, module_list):
        gd.debuginfo(prj="ds", info=f"module={module} +++ module_list={module_list}")
        for item in module_list:
            if type(item).__name__ == type(module).__name__:
                return True
        return False

    def get_module_list(model):
        mlist = []
        for child in model.children():
            gd.debuginfo(prj="ds", info=f"child={child}")
            if isinstance(child, nn.ModuleList):
                for module in child.children():
                    if not mlist:
                        mlist = [module]
                    elif not AutoTP.in_module_list(module, mlist):
                        mlist = mlist + [module]
            else:
                mlist = mlist + AutoTP.get_module_list(child)

        gd.debuginfo(prj="ds", info=f"mlist={mlist}")

        return mlist

    def supported(model):
        unsupported = ['codegen', 'deberta', 'flaubert', 'fsmt', 'gpt2', 'led', 'longformer', 'xlm', 'xlnet']
        model = str(model)
        gd.debuginfo(prj="ds", info=f"model={model}")
        key = re.search(r": (.*?)Model", model)
        if key is None:
            key = re.search(r": (.*?)Stack", model)
        if key is None:
            key = re.match(r"(.*?)Model", model)
        assert key is not None, "Not able to determine model policy automatically. Please provide policy."
        if key.group(1).lower() in unsupported:
            return False
        return True

    def get_layers(parent, module):
        layer_list = []
        for key, submodule in module._modules.items():
            gd.debuginfo(prj="ds", info=f"key={key} +++ submodule={submodule}")
            if isinstance(submodule, nn.Linear):
                layer_list = layer_list + [parent + "." + key]
            elif isinstance(submodule, nn.LayerNorm) or key == 'LayerNorm' or key == 'layer_norm':
                layer_list = layer_list + ["ln"]
            else:
                layer_list = layer_list + AutoTP.get_layers(key, submodule)

        gd.debuginfo(prj="ds", info=f"layer_list={layer_list}")

        return layer_list

    def update_policy_list(policy_list, new_module, new_gems):
        gd.debuginfo(prj="ds", info=f"policy_list={policy_list}")
        if len(policy_list):
            for i, policy in enumerate(policy_list):
                gd.debuginfo(prj="ds", info=f"i={i} +++ policy={policy}")
                # if module already exists in policy, combine gems and remove duplicates
                if policy[0] == type(new_module):
                    new_gems = set(new_gems + policy[1])
                    policy_list[i] = tuple([type(new_module), new_gems])
                    return policy_list
        policy_list.append(tuple([type(new_module), new_gems]))

        gd.debuginfo(prj="ds", info=f"update_policy_list policy_list={policy_list}")

        return policy_list

    def kernel_supported(module_list):
        policy = []
        for plcy in replace_policies:
            gd.debuginfo(prj="ds", info=f"plcy={plcy}")

            # instantiate a throw-away policy in order to populate the _orig_layer_class
            _ = plcy(None)
            if isinstance(plcy._orig_layer_class, list):
                for orig_layer_class in plcy._orig_layer_class:
                    policy.append(orig_layer_class)
            elif plcy._orig_layer_class is not None:
                policy.append(plcy._orig_layer_class)

        gd.debuginfo(prj="ds", info=f"policy={policy}")

        for child in module_list:
            gd.debuginfo(prj="ds", info=f"child={child}")
            if child.__class__ in policy:
                return True
        return False

    def tp_parser(model):
        policy_list = []
        module_list = []
        layer_list = []
        gem_list = []

        module_list = AutoTP.get_module_list(model)
        gd.debuginfo(prj="ds", info=f"module_list={module_list}")

        assert AutoTP.supported(model), "AutoTP not supported for model. Please use kernel injection since container policy for model exists." \
        if AutoTP.kernel_supported(module_list) else "AutoTP not supported for model. Please provide policy."
        for module in module_list:
            gd.debuginfo(prj="ds", info=f"module={module}")
            for key, submodule in module._modules.items():
                gd.debuginfo(prj="ds", info=f"key={key} +++ submodule={submodule}")
                if isinstance(submodule, nn.Linear):
                    layer_list = layer_list + ["." + key]
                elif isinstance(submodule, nn.LayerNorm) or key == 'LayerNorm' or key == 'layer_norm':
                    layer_list = layer_list + ["ln"]
                else:
                    layer_list = layer_list + AutoTP.get_layers(key, submodule)

            gd.debuginfo(prj="ds", info=f"layer_list={layer_list}")

            for i, layer in enumerate(layer_list):
                gd.debuginfo(prj="ds", info=f"i={i} +++ layer={layer}")
                if layer == 'ln':
                    if layer_list[i - 1] != 'ln':
                        gem_list = gem_list + [layer_list[i - 1]]
                elif 'out_proj' in layer:
                    gem_list = gem_list + [layer]
                elif 'o_proj' in layer:
                    gem_list = gem_list + [layer]
                elif 'down_proj' in layer:
                    gem_list = gem_list + [layer]
                elif 'attention.dense' in layer and 'GPTNeoX' in str(model):
                    gem_list = gem_list + [layer]
                elif 'self_attention.dense' in layer and 'falcon' in str(
                        type(module)):  # this is a hack to get the right linear layer for this model!
                    gem_list = gem_list + [layer]

            layer_list = []
            if gem_list != []:
                gem_list = list(set(gem_list))
                policy_list = AutoTP.update_policy_list(policy_list, module, gem_list)
                gem_list = []
        assert len(policy_list), "AutoTP not supported for model. Please use kernel injection since container policy for model exists." \
        if AutoTP.kernel_supported(module_list) else "Not able to determine model policy automatically. Please provide policy."

        gd.debuginfo(prj="ds", info=f"policy_list={policy_list}")

        return policy_list
