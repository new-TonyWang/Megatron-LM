# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import warnings
from typing import Optional, Tuple
from megatron.core.extensions.metis import BitLinear,BitLiearQKV
from megatron.core.extensions.metis_te import MetisColumnParallelLinear,MetisRowParallelLinear,MetisQKVLinear

from megatron.core.models.backends import LocalSpecProvider,BackendSpecProvider
from .transformer_engine_spec_provider import TESpecProvider

class MetisSpecProviderBase(BackendSpecProvider):
    def ffn_linear(self) -> type:
        """Which column parallel linear module TE backend uses"""
        return BitLinear



class MetisSpecProvider(LocalSpecProvider, MetisSpecProviderBase):
    """A protocol for providing the submodules used in Spec building."""
    pass


class MetisPersudoTeSpecProvider(TESpecProvider, MetisSpecProviderBase):
    """A protocol for providing the submodules used in Spec building."""
    def column_parallel_linear(self) -> type:
        """Which column parallel linear module the backend uses"""
        return MetisColumnParallelLinear

    def row_parallel_linear(self) -> type:
        """Which row parallel linear module the backend uses"""
        return MetisRowParallelLinear

    def fuse_layernorm_and_linear(self)-> bool:
        """Whether to fuse layernorm and linear"""
        return False

    def qkv_linear(self)->type:
        return BitLiearQKV

class MetisTeSpecProvider(TESpecProvider, MetisSpecProviderBase):
    """A protocol for providing the submodules used in Spec building."""
    def column_parallel_linear(self) -> type:
        """Which column parallel linear module the backend uses"""
        return MetisColumnParallelLinear

    def row_parallel_linear(self) -> type:
        """Which row parallel linear module the backend uses"""
        return MetisRowParallelLinear

    def fuse_layernorm_and_linear(self)-> bool:
        """Whether to fuse layernorm and linear"""
        return False

    def qkv_linear(self)->type:
        return MetisQKVLinear