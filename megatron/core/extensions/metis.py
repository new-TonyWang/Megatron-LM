from abc import abstractmethod
from collections.abc import Callable
from typing import Any, List, Optional

from torch.distributed.distributed_c10d import ProcessGroup
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from .quant import *
import torch.nn as nn
import torch.nn.init as init
import transformer_engine.pytorch  as te
from functools import partial
from megatron.core.tensor_parallel.layers import RowParallelLinear, ColumnParallelLinear
import math
from megatron.core.tensor_parallel.layers import _initialize_affine_weight_gpu
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from .transformer_engine import TELinear,HAVE_TE

import torch

import debugpy
from megatron.core.extensions.quant import quant_func, Cast2Fp4e2m1
from transformer_engine.pytorch.module.metis.metis_linear import LinearLowbitContext

class LinearLowbitFunction(torch.autograd.Function):

    @staticmethod
    @torch.no_grad()
    def svd_quant(input_:torch.Tensor, quant_func, rank=60, niter=0, adaptive_schedule="none", broadcast_dim=-1):
        
        if broadcast_dim >= 0:
            cinput = input_.select(broadcast_dim, 0)
        else:
            cinput = input_
        
        original_shape = cinput.shape
        # print("cinput.shape1===",cinput.shape)
        if len(original_shape) == 3:
            cinput = cinput.view(-1, original_shape[-1])
            input_ = input_.view(-1, original_shape[-1])

        cinput = cinput.to(torch.float32)

        ug, sg, vg = torch.svd_lowrank(
            cinput, 
            q=rank, 
            niter=niter
        )
        
        vg = vg.T.to(input_.dtype)
        ug = ug.T.to(input_.dtype)
        sg = sg.to(input_.dtype)
        
        sg, res_scalar = LinearLowbitContext.schedule_list[adaptive_schedule](sg)

        ker = (ug.T @ torch.diag(sg) @ vg)
        if broadcast_dim >= 0:
            ker = ker.unsqueeze(broadcast_dim)

        input_res = input_ - ker
        input_res_scalar = quant_func.get_scalar(input_res)
        input_res = quant_func.quant(input_res, input_res_scalar)
        input_res = quant_func.rquant(input_res, input_res_scalar)

        ug_scalar = quant_func.get_scalar(ug)
        vg_scalar = quant_func.get_scalar(vg)
        ug = quant_func.quant(ug, ug_scalar)
        ug = quant_func.rquant(ug, ug_scalar)
        
        vg = quant_func.quant(vg, vg_scalar)
        vg = quant_func.rquant(vg, vg_scalar)

        
        input_ = ug.T @ torch.diag(sg) @ vg
        if broadcast_dim >= 0:
            input_ = input_.unsqueeze(broadcast_dim)

        input_ = input_ + input_res * res_scalar
        
        if len(original_shape) == 3:
            input_ = input_.view(original_shape[0], original_shape[1], -1)
        return input_

    @staticmethod
    @torch.no_grad()
    def svd_quant_te_fp4(input_:torch.Tensor, rank=60, niter=0, adaptive_schedule="none", broadcast_dim=-1):
        
        if broadcast_dim >= 0:
            cinput = input_.select(broadcast_dim, 0)
        else:
            cinput = input_
        
        original_shape = cinput.shape
        if len(original_shape) == 3:
            cinput = cinput.view(-1, original_shape[-1])
            input_ = input_.view(-1, original_shape[-1])

        cinput.to(torch.float32)
        ug, sg, vg = torch.svd_lowrank(
            cinput.to(torch.float32), 
            q=rank, 
            niter=niter
        )
        
        vg = vg.T.to(input_.dtype)
        ug = ug.T.to(input_.dtype)
        sg = sg.to(input_.dtype)
        
        sg, res_scalar = LinearLowbitContext.schedule_list[adaptive_schedule](sg)

        ker = (ug.T @ torch.diag(sg) @ vg)
        if broadcast_dim >= 0:
            ker = ker.unsqueeze(broadcast_dim)

        input_res = input_ - ker

        # input_res = # quant_to_fp4

        # ug = # quant_to_fp4
        # vg = # quant_to_fp4

        input_ = ug.T @ torch.diag(sg) @ vg
        if broadcast_dim >= 0:
            input_ = input_.unsqueeze(broadcast_dim)

        input_ = input_ + input_res * res_scalar
        
        if len(original_shape) == 3:
            input_ = input_.view(original_shape[0], original_shape[1], -1)
        return input_    

    @staticmethod
    def forward(ctx, input_: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,local_linear_lobit_context:LinearLowbitContext  ):

        input_scalar = quant_func[local_linear_lobit_context.q_forward_input].get_scalar(input_) * local_linear_lobit_context.q_scalar
        weight_scalar = quant_func[local_linear_lobit_context.q_forward_input].get_scalar(weight) * local_linear_lobit_context.q_scalar
        

        if local_linear_lobit_context.enable_activation_svd:
            input_ = LinearLowbitFunction.svd_quant(
                input_, 
                quant_func=quant_func[local_linear_lobit_context.q_forward_input],
                rank=local_linear_lobit_context.activation_lowrank_svd,
                niter=local_linear_lobit_context.activation_lowrank_niter,
                adaptive_schedule=local_linear_lobit_context.activation_longtail_schedule,
                broadcast_dim=local_linear_lobit_context.activation_broadcast_dim
            )
        else:
            input_ = quant_func[local_linear_lobit_context.q_forward_input].quant(input_, input_scalar)
            input_ = quant_func[local_linear_lobit_context.q_forward_input].rquant(input_, input_scalar)
        
        ctx.save_for_backward(
            input_, 
            weight, 
            input_scalar, 
            weight_scalar, 
            bias,
        )
        import copy
        ctx.forward_context = local_linear_lobit_context
        # print("LinearLowbitContext.q_backward_weight==",ctx.forward_context.q_backward_weight)
        # print("backward local_linear_lobit_context==",local_linear_lobit_context)
        weight = quant_func[local_linear_lobit_context.q_forward_weight].quant(weight, weight_scalar)
        weight = quant_func[local_linear_lobit_context.q_forward_weight].rquant(weight, weight_scalar)
        
        output = torch.matmul(input_, weight.T)
        
        if bias is not None:
            output += bias
        
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        local_linear_lobit_context = ctx.forward_context 
        # print("backward local_linear_lobit_context==",local_linear_lobit_context)
        input_, weight, input_scalar, weight_scalar, bias = ctx.saved_tensors
        # print("LinearLowbitContext.q_backward_weight=",local_linear_lobit_context.q_backward_weight)

        # input_ = LinearLowbitFunction.q_backward_input.quant(input_, input_scalar)
        weight = quant_func[local_linear_lobit_context.q_backward_weight].quant(weight, weight_scalar)
        # input_ = LinearLowbitFunction.q_backward_input.rquant(input_, input_scalar)
        weight = quant_func[local_linear_lobit_context.q_backward_weight].rquant(weight, weight_scalar)
        
        
        grad_bias = grad_output.sum(dim=(0, 1)) if bias is not None else None
        
        grad_output_shape0 = grad_output.shape[0]
        grad_output_shape1 = grad_output.shape[1]
        grad_output_shape2 = grad_output.shape[2]

        grad_output = grad_output.reshape(-1, grad_output.shape[-1]).T
        if local_linear_lobit_context.enable_backward_svd:
            if local_linear_lobit_context.backward_lowrank_svd > 0:
                grad_output = LinearLowbitFunction.svd_quant(
                    grad_output, 
                    quant_func=quant_func[local_linear_lobit_context.q_backward_outputgrad],
                    rank=local_linear_lobit_context.backward_lowrank_svd,
                    niter=local_linear_lobit_context.backward_lowrank_niter,
                    adaptive_schedule=local_linear_lobit_context.backward_longtail_schedule,
                    broadcast_dim=local_linear_lobit_context.backward_broadcast_dim,
                )

            else:
                ug, sg, vg = torch.linalg.svd(grad_output, full_matrices=False)
                ug_scalar = ug.abs().mean() * local_linear_lobit_context.q_scalar
                vg_scalar = vg.abs().mean() * local_linear_lobit_context.q_scalar
                
                grad_output = \
                    quant_func[local_linear_lobit_context.q_backward_outputgrad](ug / ug_scalar) @ \
                    torch.diag(sg) @ \
                    quant_func[local_linear_lobit_context.q_backward_outputgrad](vg / vg_scalar)

                grad_output *= ug_scalar * vg_scalar
        else:
            grad_output_scalar = quant_func[local_linear_lobit_context.q_backward_outputgrad].get_scalar(grad_output) * local_linear_lobit_context.q_scalar
            
            grad_output = quant_func[local_linear_lobit_context.q_backward_outputgrad].quant(grad_output, grad_output_scalar)
            
        grad_weight = torch.matmul(
            grad_output,
            input_.reshape(-1, input_.shape[-1])
        )
    
        grad_output = grad_output.T.reshape(grad_output_shape0, grad_output_shape1, grad_output_shape2)
        grad_input = torch.matmul(grad_output, weight)                    
        
        return grad_input, grad_weight, grad_bias, None # for context

class LinearLowbit(torch.nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        config: ModelParallelConfig,
        bias=True,
    ) -> None:
        super().__init__()
        self.config = config
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device=torch.cuda.current_device(), dtype=self.config.params_dtype)
        )
        setattr(self.weight,"allreduce",True)
        if bias:
            self.bias = torch.nn.Parameter(
                torch.empty((out_features,), device=torch.cuda.current_device(), dtype=self.config.params_dtype)
            )
            setattr(self.bias, "allreduce", True)
        else:
            self.register_parameter("bias", None)
            self.bias = None
        self.linear_lobit_context = LinearLowbitContext().clone()
        # self.reset_parameters()
        # debugpy.breakpoint()
        # LinearLowbitFunction.ctx = LinearLowbitContext
        print(f"LinearLowbit ==",LinearLowbitContext())
    # def reset_parameters(self):
    #     init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    #     if self.bias is not None:
    #         fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    #         bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    #         init.uniform_(self.bias, -bound, bound)
            # Hook adding a default empty _extra_state for state dict
        self._register_load_state_dict_pre_hook(
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(
                f"{prefix}_extra_state"
            )
        )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias sharded"""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {"weight": 0, "bias": 0}, sharded_offsets
        )
    def set_extra_state(self, state: Any):
        """Extra state is ignored"""

    def get_extra_state(self) -> None:
        """Keep compatibility with TE state dict."""
        return None

    def _forward_persudo(self,input):
        pass

    def forward(self, input):
        return LinearLowbitFunction.apply(input, self.weight, self.bias,self.linear_lobit_context)

    def __repr__(self):
        tp = 1
        use_bias = self.bias is not None and self.bias is True
        return (
            f"{type(self).__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={use_bias}, TP={tp})"
        )

class BitLinear(torch.nn.Module):
    def __init__(self,
        input_size,
        output_size,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias=True,
        gather_output=False,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        skip_weight_param_allocation: bool = False,
        embedding_activation_buffer: Optional[List[torch.Tensor]] = None,
        grad_output_buffer: Optional[List[torch.Tensor]] = None,
        is_expert: bool = False,
        tp_comm_buffer_name: str = None,  # Not used
        disable_grad_reduce: bool = False,
        input_is_parallel: bool = False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,):
        super().__init__()
        print(f"entering BitLinear with input_size:{input_size}, output_size:{output_size}, bias:{bias}")
        assert tp_group.size() == 1,"BitLinear only support single process group"
        
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        self.skip_bias_add = skip_bias_add
        self.is_expert = is_expert
        self.expert_parallel = config.expert_model_parallel_size > 1
        self.embedding_activation_buffer = embedding_activation_buffer
        self.grad_output_buffer = grad_output_buffer
        self.config = config
        self.disable_grad_reduce = disable_grad_reduce
        self.tp_group = tp_group
        if LinearLowbitContext.enable_lowbit:
            self.linear_residual = LinearLowbit(input_size, output_size, config,bias=bias)
        else:
            self.linear_residual = nn.Linear(input_size, output_size, bias=bias, device=torch.cuda.current_device(),dtype=self.config.params_dtype)
            # init.kaiming_uniform_(self.linear_residual.weight, a=math.sqrt(5))
            if bias:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.linear_residual.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.linear_residual.bias, -bound, bound)
        if config.perform_initialization:
            _initialize_affine_weight_gpu(
                self.linear_residual.weight,
                init_method,
                partition_dim=0,
                stride=stride,
                is_expert=self.is_expert,
            )
        # self.ulinear:Optional[Any[nn.Linear, LinearLowbit]] = None
        # self.vlinear:Optional[Any[nn.Linear, LinearLowbit]] = None
        # self.s = None
        self.weight_svd_has_initialized = False
        self.weight_svd_decomposition()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,Optional[torch.Tensor]]:
        y = self.vlinear(x)
        y = torch.mul(self.s, y)
        y = self.ulinear(y)
        if LinearLowbitContext.forward_svd_rank > 0:
            y += self.linear_residual(x)
        # print(f"x.shape=={x.shape},,y.shape=={y.shape}")
        return y, None #bias is None

    @torch.no_grad()
    def initialize_weight_svd_decomposition(self):
        device = self.linear_residual.weight.device
        weight_fp32 = self.linear_residual.weight.float()
        u, s, v = torch.linalg.svd(weight_fp32, full_matrices=False)
        u = u.to(device = self.linear_residual.weight.get_device(),dtype=torch.bfloat16)
        s = s.to(device = self.linear_residual.weight.get_device(),dtype=torch.bfloat16)
        v = v.to(device = self.linear_residual.weight.get_device(),dtype=torch.bfloat16)
        if self.linear_residual.bias :
            bias = self.linear_residual.bias.to(device=device)
        else:
            bias = None
        w = self.linear_residual.weight.to(device=device)
        # forward svd low rank
        if LinearLowbitContext.forward_svd_rank > 0:
            self.linear_residual = LinearLowbit(
                self.linear_residual.weight.shape[1], 
                self.linear_residual.weight.shape[0],
                self.config,
                bias=True if bias else False, 
            )
                # device=device
            if not bias is None:
                self.linear_residual.bias.copy_(bias)
            self.linear_residual.weight.copy_(
                w - \
                u[:,LinearLowbitContext.forward_svd_rank:] @ \
                torch.diag(s[LinearLowbitContext.forward_svd_rank:]) @ \
                v[LinearLowbitContext.forward_svd_rank:]
                )
        self.weight_svd_has_initialized = True
        return u,s,v,bias

    @torch.no_grad()
    def update_weight_svd_decomposition(self):
        assert self.weight_svd_has_initialized
        weight_fp32 = (self.ulinear.weight @ torch.diag(self.s) @  self.vlinear.weight).float()
        u, s, v = torch.linalg.svd(
            weight_fp32, full_matrices=False)
        u = u.to(device = self.linear_residual.weight.get_device(),dtype=torch.bfloat16)
        s = s.to(device = self.linear_residual.weight.get_device(),dtype=torch.bfloat16)
        v = v.to(device = self.linear_residual.weight.get_device(),dtype=torch.bfloat16)
        bias = self.ulinear.bias
        return u,s,v,bias

    @torch.no_grad()
    def weight_svd_decomposition(self):

        if not self.weight_svd_has_initialized:
          u,s,v,bias = self.initialize_weight_svd_decomposition()
        else:
          u,s,v,bias = self.update_weight_svd_decomposition()
        if LinearLowbitContext.enable_lowbit: 
            # nv fp8
            # ******************************************************************
            # self.ss = u @ s @ u.transpose()
            # with fp8_model_init(enabled=True):
            #     self.uvlinear = te.Linear(
            #         self.linear_residual.weight.shape[1], 
            #         self.linear_residual.weight.shape[0], 
            #         init_method=partial(BitLinear._init_telinear, u @ v), 
            #         bias=False, 
            #         device=self.device
            #     )
            if LinearLowbitContext.forward_svd_rank > 0:
                self.vlinear = LinearLowbit(
                    v.shape[1], 
                    LinearLowbitContext.forward_svd_rank, # v.shape[0] // 30, 
                    self.config,
                    bias=False, 
                    )
                self.ulinear = nn.Linear(
                    LinearLowbitContext.forward_svd_rank, # u.shape[1] // 30, 
                    u.shape[0], 
                    device=torch.cuda.current_device(),
                    dtype=self.config.params_dtype
                )
                self.vlinear.weight.copy_(v[: LinearLowbitContext.forward_svd_rank, :])
                self.ulinear.weight.copy_(u[:, : LinearLowbitContext.forward_svd_rank])
            else:
                self.vlinear = LinearLowbit(
                    v.shape[1], 
                    v.shape[0], # v.shape[0] // 30, 
                    self.config,
                    bias=False, 
                    )
                self.ulinear = nn.Linear(
                    u.shape[1], # u.shape[1] // 30, 
                    u.shape[0], 
                    device=torch.cuda.current_device(),
                    bias=True if bias else False,
                    dtype=self.config.params_dtype
                )
                self.vlinear.weight.copy_(v)
                self.ulinear.weight.copy_(u)


            # forward svd low rank
            if LinearLowbitContext.forward_svd_rank > 0 and bias :
                self.ulinear.bias.copy_(bias)
        else:
            self.vlinear = nn.Linear(v.shape[1], v.shape[0], bias=False)
            self.ulinear = nn.Linear(u.shape[1], u.shape[0])

            
            self.vlinear.weight = nn.Parameter(v)
            self.ulinear.weight = nn.Parameter(u)
            if (not bias is None):
                self.ulinear.bias = nn.Parameter(
                    self.linear_residual.bias.clone().cuda(self.linear_residual.weight.get_device())
                )
        
        if LinearLowbitContext.forward_svd_rank > 0:
            self.s = torch.nn.Parameter(s[:LinearLowbitContext.forward_svd_rank])
            
        else:
            self.s = torch.nn.Parameter(s)
            # self.linear_residual = None

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Sharding along axis 0, bias sharded"""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict, prefix, {"weight": 0, "bias": 0}, sharded_offsets
        )
    def set_extra_state(self, state: Any):
        """Extra state is ignored"""

    def get_extra_state(self) -> None:
        """Keep compatibility with TE state dict."""
        return None

class BitLiearQKV(nn.Module):
    def __init__(self,
    input_size,
    output_size,
    *,
    config: TransformerConfig,
    init_method: Callable,
    bias=True,
    gather_output=False,
    stride=1,
    keep_master_weight_for_test=False,
    skip_bias_add=False,
    skip_weight_param_allocation: bool = False,
    embedding_activation_buffer: Optional[List[torch.Tensor]] = None,
    grad_output_buffer: Optional[List[torch.Tensor]] = None,
    is_expert: bool = False,
    tp_comm_buffer_name: str = None,  # Not used
    disable_grad_reduce: bool = False,
    input_is_parallel: bool = False,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,):
        super().__init__() 

        self.query_projection_size = config.kv_channels * config.num_attention_heads
        self.kv_projection_size = config.kv_channels * config.num_query_groups
        print("config.query_projection_size==,",self.query_projection_size)
        print("config.kv_projection_size==,", self.kv_projection_size)
        bitlinear_kwargs = {
            "config": config,
            "init_method": init_method,
            "bias": bias,
            "gather_output": gather_output,
            "stride": stride,
            "keep_master_weight_for_test": keep_master_weight_for_test,
            "skip_bias_add": skip_bias_add,
            "skip_weight_param_allocation": skip_weight_param_allocation,
            "embedding_activation_buffer": embedding_activation_buffer,
            "grad_output_buffer": grad_output_buffer,
            "is_expert": is_expert,
            "tp_comm_buffer_name": tp_comm_buffer_name,
            "disable_grad_reduce": disable_grad_reduce,
            "input_is_parallel": input_is_parallel,
            "tp_group": tp_group,
        }

        self.q_linear = BitLinear(input_size, self.query_projection_size, **bitlinear_kwargs)
        self.k_linear = BitLinear(input_size, self.kv_projection_size, **bitlinear_kwargs)
        self.v_linear = BitLinear(input_size, self.kv_projection_size, **bitlinear_kwargs)
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,Optional[torch.Tensor],Optional[torch.Tensor],Optional[torch.Tensor]]:
        sq,b,h = x.shape
        q,_ = self.q_linear(x)
        k,_ = self.k_linear(x)
        v,_ = self.v_linear(x)
        # print("q.shape==",q.shape)
        # print("k.shape==",k.shape)
        # print("v.shape==",v.shape)
        out = torch.cat((q, k, v), dim=-1)
        # print("out.shape==",out.shape)
        return out, None

# class BitLinearRowParallelLinear(RowParallelLinear):
#         super().__init__(input_size, output_size, config=config, init_method=init_method, bias=bias, input_is_parallel=input_is_parallel, skip_bias_add=skip_bias_add, stride=stride, keep_master_weight_for_test=keep_master_weight_for_test, is_expert=is_expert, tp_comm_buffer_name=tp_comm_buffer_name, tp_group=tp_group)
#         self.forward_svd_rank = 6
#         pass
# class BitLinearColumnParallelLinear(ColumnParallelLinear):
#     pass