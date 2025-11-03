# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import dataclasses
import inspect
import io
import os
import pickle
import warnings
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from packaging.version import Version as PkgVersion
from torch import Tensor
from torch.nn.parameter import Parameter

from megatron.core.dist_checkpointing.utils import replace_prefix_for_sharding
from megatron.core.extensions.transformer_engine import _get_extra_te_kwargs
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_expert_data_parallel_rank,
    get_expert_model_parallel_rank,
    get_expert_model_parallel_world_size,
    get_hierarchical_context_parallel_groups,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    set_tensor_model_parallel_attributes,
)
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    get_data_parallel_rng_tracker_name,
    get_expert_parallel_rng_tracker_name,
)
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import (
    is_layer_window_attention,
    make_sharded_tensors_for_checkpoint,
)
from megatron.core.utils import (
    get_pg_rank,
    get_pg_size,
    get_te_version,
    get_tensor_model_parallel_group_if_none,
    is_te_min_version,
    is_torch_min_version,
)

try:
    import transformer_engine as te

    HAVE_TE = True
except ImportError:
    from unittest.mock import MagicMock

    te = MagicMock()
    HAVE_TE = False

def condition_init_method(config, init_method):
    """Condition TE init_method on config.perform_initialization."""
    return init_method if config.perform_initialization else (lambda w: None)

class MetisTELinear(te.pytorch.MetisLinear):
    """Wrapper for the Transformer-Engine's `Linear` layer.

    Note that if Megatron's parallel_state has not been initialized
    yet, the tp_group passed to TE will be None and must be set later
    via set_tensor_parallel_group().

    parallel_mode currently supports 3 different values:
        - "column": Split the weight matrix along output dimension (used in TEColumnParallelLinear)
        - "row": Split the weight matrix along input dimension (used in TERowParallelLinear)
        - "duplicated": No tensor parallelism and weight is duplicated across TP ranks
        - Note: For expert linear layers, we will disable communication logic here
                as TP communication is handled in token_dispatcher.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        parallel_mode: Optional[str],
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        skip_bias_add: bool,
        skip_weight_param_allocation: bool=False,
        tp_comm_buffer_name: Optional[str] = None,
        is_expert: bool = False,
        symmetric_ar_type: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        if not HAVE_TE:
            raise ImportError(
                "Transformer Engine is not installed. "
                "Please install it with `pip install transformer-engine`."
            )
        assert config.tensor_model_parallel_size == 1,"Metis TELinear only supports tensor_model_parallel_size=1 for now."
        
        self.config = config

        # TE returns a zero length Tensor when bias=False and
        # return_bias=True, but we prefer None.  So in that case we
        # tell TE to not return the bias, and return None
        # ourselves. This way our forward always returns two values
        # and we don't have to deal with the zero length Tensor.
        self.te_return_bias = skip_bias_add and bias
        self.is_first_microbatch = True
        self.disable_parameter_transpose_cache = self.config.disable_parameter_transpose_cache
        self.symmetric_ar_type = symmetric_ar_type
        if skip_weight_param_allocation:
            raise ValueError(
                "Transformer Engine linear layers do not support skip_weight_param_allocation"
            )

        extra_kwargs = _get_extra_te_kwargs(config)

        if self.config.delay_wgrad_compute:
            if is_te_min_version("2.3.0"):
                extra_kwargs["delay_wgrad_compute"] = self.config.delay_wgrad_compute
            else:
                raise RuntimeError("Only TE with version >=2.3.0 supports delay_wgrad_compute now.")
        if (
            self.config.tp_comm_overlap
            and tp_comm_buffer_name
            and tp_comm_buffer_name not in ["qkv", "proj", "fc1", "fc2"]
        ):
            self.config.tp_comm_overlap = False
            warnings.warn(
                f"The user buffer name {tp_comm_buffer_name} is not supported in"
                "Transformer Engine. Disabling TP communication overlap "
                "for this layer."
            )

        if is_te_min_version("0.8.0"):
            if self.config.tp_comm_overlap:
                if is_te_min_version("1.5.0"):
                    # Use old overlap flags if they were supplied instead
                    extra_kwargs["ub_overlap_ag"] = (
                        self.config.tp_comm_overlap_ag
                        if hasattr(self.config, "tp_comm_overlap_ag")
                        else self.config.tp_comm_split_ag or self.config.tp_comm_atomic_ag
                    )
                    extra_kwargs["ub_overlap_rs"] = (
                        self.config.tp_comm_overlap_rs
                        if hasattr(self.config, "tp_comm_overlap_rs")
                        else self.config.tp_comm_split_rs or self.config.tp_comm_atomic_rs
                    )
                    # Disable ub overlap for experts.
                    if is_expert:
                        extra_kwargs["ub_overlap_ag"] = False
                        extra_kwargs["ub_overlap_rs"] = False
                else:
                    extra_kwargs["ub_split_ag"] = self.config.tp_comm_split_ag
                    extra_kwargs["ub_atomic_gemm_ag"] = self.config.tp_comm_atomic_ag
                    extra_kwargs["ub_split_rs"] = self.config.tp_comm_split_rs
                    extra_kwargs["ub_atomic_gemm_rs"] = self.config.tp_comm_atomic_rs
                    # Disable ub overlap for experts.
                    if is_expert:
                        extra_kwargs["ub_split_ag"] = False
                        extra_kwargs["ub_atomic_gemm_ag"] = False
                        extra_kwargs["ub_split_rs"] = False
                        extra_kwargs["ub_atomic_gemm_rs"] = False
                if is_te_min_version("1.0.0", check_equality=False):
                    assert (
                        tp_comm_buffer_name is not None
                    ), "Buffer name should be set to configure communication overlap settings"
                    extra_kwargs["ub_name"] = tp_comm_buffer_name

        if symmetric_ar_type is not None:
            assert is_torch_min_version("2.7.0a0"), "Must have at least torch version 2.7 or higher"
            assert is_te_min_version("2.3.0") or get_te_version() == PkgVersion(
                "2.3.0.dev0+39c0e70"
            ), "Must have at least TE version 2.3 or higher to use symmetric memory all reduce"
            extra_kwargs["symmetric_ar_type"] = symmetric_ar_type
        if parallel_mode == "duplicated":
            assert tp_group is None, "duplicated linear should not have tp_group set"
            tp_size = 1
        else:
            tp_size = get_pg_size(tp_group)

        self.expert_parallel = self.config.expert_model_parallel_size > 1
        if is_expert:
            rng_tracker_name = get_expert_parallel_rng_tracker_name()
        else:
            if parallel_mode == "duplicated":
                rng_tracker_name = get_data_parallel_rng_tracker_name()
            else:
                rng_tracker_name = None
        if is_te_min_version("1.7.0"):
            extra_kwargs["rng_tracker_name"] = rng_tracker_name

        te_parallel_mode = parallel_mode
        if parallel_mode == "duplicated":
            # Handle non-parallel case
            tp_group = None
            tp_size = 1
            explicit_expert_comm = False
            te_parallel_mode = None
        else:
            # Disable communications in TE when using TP or EP by
            explicit_expert_comm = is_expert and (tp_size > 1 or self.expert_parallel)

            if explicit_expert_comm:
                if parallel_mode == "column":
                    output_size = divide(output_size, tp_size)
                elif parallel_mode == "row":
                    input_size = divide(input_size, tp_size)
                te_parallel_mode = None
                tp_size = 1
                tp_group = None

        super().__init__(
            in_features=input_size,
            out_features=output_size,
            sequence_parallel=self.config.sequence_parallel,
            fuse_wgrad_accumulation=self.config.gradient_accumulation_fusion,
            # Pass None if not initialized for backward compatibility with the ckpt converter.
            tp_group=tp_group if torch.distributed.is_initialized() else None,
            tp_size=tp_size,
            get_rng_state_tracker=(
                get_cuda_rng_tracker if get_cuda_rng_tracker().is_initialized() else None
            ),
            init_method=condition_init_method(config, init_method),
            bias=bias,
            return_bias=self.te_return_bias,
            parallel_mode=te_parallel_mode,
            **extra_kwargs,
        )

        for param in self.parameters():
            if is_expert:
                # Reduce the gradient on the expert_data_parallel group for expert linear layers
                setattr(param, "allreduce", not self.expert_parallel)
            else:
                # Reduce the gradient on DP group
                setattr(param, "allreduce", True)
                if parallel_mode == "duplicated":
                    # Reduce the gradient further on the TP group since the weight is
                    # duplicated across TP ranks
                    setattr(param, "sequence_parallel", self.config.sequence_parallel)

    def forward(self, x):
        """Forward."""
        _is_first_microbatch = (
            None if self.disable_parameter_transpose_cache else self.is_first_microbatch
        )
        out = super().forward(x, is_first_microbatch=_is_first_microbatch)
        self.is_first_microbatch = False

        # TE only returns a tuple when return_bias is True, otherwise
        # it returns a single Tensor, we always want to return two
        # values regardless of the arguments.
        if self.te_return_bias:
            return out
        return out, None

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Replicate cross TP/DP."""

        # Provide the dist-ckpt support when TELinear is directly used
        # It can only happen with duplicated parallel mode
        assert (
            self.parallel_mode is None
        ), "TELinear sharded_state_dict can only be used with duplicated parallel mode"
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(state_dict, prefix, None, sharded_offsets)

    def backward_dw(self):
        """Compute weight gradients during the backward pass if delay_wgrad_compute is enabled."""
        if self.config.delay_wgrad_compute:
            super().backward_dw()

class MetisColumnParallelLinear(MetisTELinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        if not HAVE_TE:
            raise ImportError(
                "Transformer Engine is not installed. "
                "Please install it with `pip install transformer-engine`."
            )

        if gather_output:
            raise ValueError("Transformer Engine linear layers do not support gather_output = True")
        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        world_size = get_pg_size(tp_group)
        rank = get_pg_rank(tp_group)

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode="column",
            config=config,
            init_method=(
                condition_init_method(config, init_method)
                if not config.use_cpu_initialization
                else lambda w: None
            ),
            bias=bias,
            skip_bias_add=skip_bias_add,
            is_expert=is_expert,
            skip_weight_param_allocation=skip_weight_param_allocation,
            tp_comm_buffer_name=tp_comm_buffer_name,
            symmetric_ar_type=config.symmetric_ar_type,
            tp_group=tp_group,
        )

        if config.use_cpu_initialization:
            output_size_per_partition = divide(output_size, world_size)
            _ = _initialize_affine_weight_cpu(
                self.weight,
                output_size,
                input_size,
                output_size_per_partition,
                0,
                init_method=condition_init_method(config, init_method),
                stride=1,
                return_master_weight=False,
                rank=rank,
                world_size=world_size,
                skip_set_tensor_parallel_attributes=True,
            )
            if bias:
                self.bias = Parameter(
                    torch.empty(output_size_per_partition, dtype=config.params_dtype)
                )
                set_tensor_model_parallel_attributes(self.bias, True, 0, 1)
                with torch.no_grad():
                    self.bias.zero_()
                setattr(self.bias, "allreduce", True)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias sharded"""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {"weight": 0, "bias": 0}, sharded_offsets
        )

    def __repr__(self):
        base_repr = super().__repr__()
        child_repr =  (
            f"{type(self).__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.use_bias}, TP={self.tp_size})"
        )
        return f"{base_repr}\n\n{child_repr}"

    def backward_dw(self):
        """Compute weight gradients during the backward pass if delay_wgrad_compute is enabled."""
        if self.config.delay_wgrad_compute:
            super().backward_dw()

class MetisRowParallelLinear(MetisTELinear):
    """Wrapper for the Transformer-Engine's `Linear` layer
    but specialized similar to megatron's `RowParallelLinear` layer."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        if not HAVE_TE:
            raise ImportError(
                "Transformer Engine is not installed. "
                "Please install it with `pip install transformer-engine`."
            )

        if not input_is_parallel:
            raise ValueError(
                "Transformer Engine linear layers do not support input_is_parallel = False"
            )
        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            parallel_mode="row",
            config=config,
            init_method=(
                condition_init_method(config, init_method)
                if not config.use_cpu_initialization
                else lambda w: None
            ),
            bias=bias,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=False,
            # We don't currently use this for row parallel layers # pylint: disable=line-too-long
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            symmetric_ar_type=config.symmetric_ar_type,
            tp_group=tp_group,
        )
        if config.use_cpu_initialization:
            world_size = get_pg_size(tp_group)
            rank = get_pg_rank(tp_group)
            input_size_per_partition = divide(input_size, world_size)
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight,
                output_size,
                input_size,
                input_size_per_partition,
                1,
                init_method=condition_init_method(config, init_method),
                stride=1,
                return_master_weight=False,
                params_dtype=config.params_dtype,
                rank=rank,
                world_size=world_size,
                skip_set_tensor_parallel_attributes=True,
            )
            if bias:
                self.bias = Parameter(torch.empty(output_size, dtype=config.params_dtype))
                # Always initialize bias to zero.
                with torch.no_grad():
                    self.bias.zero_()
                setattr(self.bias, "allreduce", True)
                setattr(self.bias, "sequence_parallel", config.sequence_parallel)

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 1, bias not sharded"""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {"weight": 1}, sharded_offsets
        )

    def __repr__(self):
        base_repr = super().__repr__()
        child_repr =  (
            f"{type(self).__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.use_bias}, TP={self.tp_size})"
        )
        return f"{base_repr}\n\n{child_repr}"

    def backward_dw(self):
        """Compute weight gradients during the backward pass if delay_wgrad_compute is enabled."""
        if self.config.delay_wgrad_compute:
            super().backward_dw()

class MetisQKVLinear(torch.nn.Module):
    def __init__(self,
        input_size: int,
        output_size: int,
        *,
        config: TransformerConfig,
        init_method: Callable,
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: Optional[str] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__()
        self.query_projection_size = config.kv_channels * config.num_attention_heads
        self.kv_projection_size = config.kv_channels * config.num_query_groups
        print("config.query_projection_size==,",self.query_projection_size)
        print("config.kv_projection_size==,", self.kv_projection_size)
        self.metis_linear_params = {
            "input_size": input_size,
            "config": config,
            "init_method": init_method,
            "bias": bias,
            "gather_output": gather_output,
            "skip_bias_add": skip_bias_add,
            "skip_weight_param_allocation": skip_weight_param_allocation,
            "is_expert": is_expert,
            "tp_comm_buffer_name": tp_comm_buffer_name,
            "tp_group": tp_group
        }
        self.q_linear = MetisColumnParallelLinear(
            output_size=self.query_projection_size,
            **self.metis_linear_params)
        self.k_linear = MetisColumnParallelLinear(
            output_size=self.kv_projection_size,
            **self.metis_linear_params)
        self.v_linear = MetisColumnParallelLinear(
            output_size=self.kv_projection_size,
            **self.metis_linear_params)

    def forward(self, x):
        q, _ = self.q_linear(x)
        k, _ = self.k_linear(x)
        v, _ = self.v_linear(x)
        out = torch.cat((q, k, v), dim=-1)
        # print("out.shape==",out.shape)
        return out, None


    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias sharded"""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {"weight": 0, "bias": 0}, sharded_offsets
        )

    def set_extra_state(self, state: Any):
        """Extra state is ignored"""
        pass

    def get_extra_state(self) -> None:
        """Keep compatibility with TE state dict."""
        return None