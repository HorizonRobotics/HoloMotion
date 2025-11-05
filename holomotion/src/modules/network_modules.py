# Project HoloMotion
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.


import inspect
import math
from types import SimpleNamespace
from typing import Optional

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from loguru import logger


class MLP(nn.Module):
    def __init__(self, obs_serializer, module_config_dict):
        super(MLP, self).__init__()
        self.obs_serializer = obs_serializer
        self.obs_dim_dict = obs_serializer.obs_dim_dict
        self.obs_seq_len_dict = obs_serializer.obs_seq_len_dict
        self.module_config_dict = module_config_dict
        self.input_dim = obs_serializer.obs_flat_dim

        self.use_layernorm = module_config_dict.get("use_layernorm", False)

        self._calculate_output_dim()
        self._build_mlp(self.module_config_dict.layer_config)

    def _calculate_output_dim(self):
        output_dim = 0
        for each_output in self.module_config_dict["output_dim"]:
            if isinstance(each_output, (int, float)):
                output_dim += each_output
            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(
                    f"{current_function_name} - Unknown output type: {each_output}"
                )
        self.output_dim = output_dim

    def _build_mlp(self, layer_config):
        layers = []
        hidden_dims = layer_config["hidden_dims"]
        output_dim = self.output_dim
        activation = getattr(nn, layer_config["activation"])()

        layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        layers.append(activation)

        for i in range(len(hidden_dims)):
            if i == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[i], output_dim))
            else:
                if self.use_layernorm:
                    layers.append(nn.LayerNorm(hidden_dims[i]))
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                layers.append(activation)

        self.module = nn.Sequential(*layers)

    def forward(self, input):
        output = self.module(input)
        return output
