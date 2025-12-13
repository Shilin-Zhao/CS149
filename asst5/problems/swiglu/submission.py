import torch
import triton
import triton.language as tl
import os
from task import input_t, output_t

# ============================================================================
# Triton GEMM for FP32 - 基于官方教程，但为 fp32 输入/输出修改
# https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
# ============================================================================

def get_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # 使用 ieee 精度而非 tf32，确保与 PyTorch 结果一致
        accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 输出 fp32（不转换成 fp16）
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    2D GEMM: (M, K) x (K, N) -> (M, N)，fp32 输入输出。
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K2, N = b.shape
    b = b.contiguous()
    # 分配 fp32 输出
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


# ============================================================================
# 方案2: 双 Kernel 融合 (SWIGLU_BACKEND=dual_fused)
# ============================================================================

@triton.autotune(configs=get_autotune_config(), key=['M', 'N', 'K'])
@triton.jit
def fused_matmul_bias_swish_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    beta,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Fused kernel: out = Swish(X @ W + bias)"""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(x_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        w = tl.load(w_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(x, w, accumulator, input_precision="ieee")
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk

    # 融合: bias + Swish
    offs_n_actual = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    bias_vals = tl.load(bias_ptr + offs_n_actual, mask=offs_n_actual < N, other=0.0)
    gate = accumulator + bias_vals[None, :]
    swish_gate = gate * tl.sigmoid(beta * gate)

    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + stride_om * offs_om[:, None] + stride_on * offs_on[None, :]
    out_mask = (offs_om[:, None] < M) & (offs_on[None, :] < N)
    tl.store(out_ptrs, swish_gate, mask=out_mask)


@triton.autotune(configs=get_autotune_config(), key=['M', 'N', 'K'])
@triton.jit
def fused_matmul_bias_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Fused kernel: out = X @ W + bias"""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(x_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        w = tl.load(w_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(x, w, accumulator, input_precision="ieee")
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk

    # 融合: 加 bias
    offs_n_actual = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    bias_vals = tl.load(bias_ptr + offs_n_actual, mask=offs_n_actual < N, other=0.0)
    result = accumulator + bias_vals[None, :]

    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + stride_om * offs_om[:, None] + stride_on * offs_on[None, :]
    out_mask = (offs_om[:, None] < M) & (offs_on[None, :] < N)
    tl.store(out_ptrs, result, mask=out_mask)


def fused_matmul_bias_swish(x, w, bias, beta):
    """Fused: Swish(X @ W + bias)"""
    assert x.is_contiguous()
    M, K = x.shape
    K2, N = w.shape
    w = w.contiguous()
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    fused_matmul_bias_swish_kernel[grid](
        x, w, bias, out, M, N, K,
        x.stride(0), x.stride(1), w.stride(0), w.stride(1), out.stride(0), out.stride(1),
        beta,
    )
    return out


def fused_matmul_bias(x, w, bias):
    """Fused: X @ W + bias"""
    assert x.is_contiguous()
    M, K = x.shape
    K2, N = w.shape
    w = w.contiguous()
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    fused_matmul_bias_kernel[grid](
        x, w, bias, out, M, N, K,
        x.stride(0), x.stride(1), w.stride(0), w.stride(1), out.stride(0), out.stride(1),
    )
    return out


# ============================================================================
# 方案3: 完全融合 - 单 Kernel (SWIGLU_BACKEND=fully_fused)
# 在一个 Kernel 中完成: X@W+b -> Swish -> X@V+c -> 逐元素乘 -> output
# X 只读一次，中间结果全在寄存器，只写一次 DRAM
# ============================================================================

def get_fused_autotune_config():
    # 完全融合需要更小的 block size，因为寄存器压力翻倍
    return [
        # 你实测最快
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        # # 邻域变体（保持 K=32，调 M/N 与 group/warp/stage）
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},  num_stages=4, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},  num_stages=4, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},  num_stages=4, num_warps=4),
        # # 稍减 stage/warp，压低寄存器与 SMEM 压力
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},  num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},  num_stages=3, num_warps=4),
        # # 小 tile 兜底，防止 OOM/溢出
        # triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},  num_stages=4, num_warps=2),
        # triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},  num_stages=4, num_warps=2),
    ]


@triton.autotune(configs=get_fused_autotune_config(), key=['M', 'N', 'K'])
@triton.jit
def fully_fused_swiglu_kernel(
    # Pointers
    x_ptr, w_ptr, v_ptr, bias_b_ptr, bias_c_ptr, out_ptr,
    # Dimensions: X is (M, K), W/V are (K, N), output is (M, N)
    M, N, K,
    # Strides for X (row, col)
    stride_xm, stride_xk,
    # Strides for W and V (row, col)
    stride_wk, stride_wn,
    # Strides for output (row, col)
    stride_om, stride_on,
    # Swish beta
    beta,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    完全融合的 SwiGLU kernel (参考开源实现):
    output = Swish(X @ W + b) * (X @ V + c)
    
    使用 block_ptr + allow_tf32=False 保证精度。
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # 累加器
    gate_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    value_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # X: (M, K), 每次取 (BLOCK_SIZE_M, BLOCK_SIZE_K)
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, K),
        strides=(stride_xm, stride_xk),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )
    # W: (K, N), 每次取 (BLOCK_SIZE_K, BLOCK_SIZE_N)
    w_block_ptr = tl.make_block_ptr(
        base=w_ptr,
        shape=(K, N),
        strides=(stride_wk, stride_wn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(1, 0),
    )
    # V: (K, N), 每次取 (BLOCK_SIZE_K, BLOCK_SIZE_N)
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr,
        shape=(K, N),
        strides=(stride_wk, stride_wn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(1, 0),
    )

    # K 循环累加
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x_tile = tl.load(x_block_ptr, boundary_check=(0, 1))
        w_tile = tl.load(w_block_ptr, boundary_check=(0, 1))
        v_tile = tl.load(v_block_ptr, boundary_check=(0, 1))


        # allow_tf32=False 保证 FP32 精度（与参考实现一致）
        gate_acc += tl.dot(x_tile, w_tile, allow_tf32=False)
        value_acc += tl.dot(x_tile, v_tile, allow_tf32=False)

        x_block_ptr = tl.advance(x_block_ptr, (0, BLOCK_SIZE_K))
        w_block_ptr = tl.advance(w_block_ptr, (BLOCK_SIZE_K, 0))
        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_SIZE_K, 0))

    # 加 bias
    offs_n_actual = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = offs_n_actual < N
    bias_b_vals = tl.load(bias_b_ptr + offs_n_actual, mask=n_mask, other=0.0)
    bias_c_vals = tl.load(bias_c_ptr + offs_n_actual, mask=n_mask, other=0.0)

    gate = gate_acc + bias_b_vals[None, :]
    # Swish: gate * sigmoid(beta * gate)
    swish_gate = gate * tl.sigmoid(beta * gate)
    value = value_acc + bias_c_vals[None, :]
    output = swish_gate * value

    # 写回 output
    out_block_ptr = tl.make_block_ptr(
        base=out_ptr,
        shape=(M, N),
        strides=(stride_om, stride_on),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    tl.store(out_block_ptr, output)


def fully_fused_swiglu(x, W, V, b, c, beta):
    """完全融合的 SwiGLU: output = Swish(X @ W + b) * (X @ V + c)"""
    assert x.is_contiguous()
    M, K = x.shape
    K2, N = W.shape
    W = W.contiguous()
    V = V.contiguous()
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    fully_fused_swiglu_kernel[grid](
        x, W, V, b, c, out,
        M, N, K,
        x.stride(0), x.stride(1),
        W.stride(0), W.stride(1),
        out.stride(0), out.stride(1),
        beta,
    )
    return out


# ============================================================================
# 主入口
# SWIGLU_BACKEND=pytorch: PyTorch baseline (与 reference.py 完全一致)
# SWIGLU_BACKEND=unfused: Triton 非融合
# SWIGLU_BACKEND=dual_fused: Triton 双 Kernel 融合
# SWIGLU_BACKEND=fully_fused: Triton 完全融合 (单 Kernel, fp32 输入)
# ============================================================================

def custom_kernel(data: input_t) -> output_t:
    """
    SwiGLU 实现。
    
    Environment Variables:
    - SWIGLU_BACKEND=pytorch: PyTorch baseline (与 reference.py 完全一致, 默认)
    - SWIGLU_BACKEND=unfused: Triton 非融合 matmul
    - SWIGLU_BACKEND=dual_fused: Triton 双 Kernel 融合
    - SWIGLU_BACKEND=fully_fused: Triton 完全融合 (单 Kernel, fp32 输入)
    """
    backend = os.getenv("SWIGLU_BACKEND", "pytorch").lower()
    x, W, V, b, c, beta = data
    
    if backend == "pytorch":
        # PyTorch baseline - 与 reference.py 完全一致
        gate = x @ W + b
        swish_gate = gate * torch.sigmoid(beta * gate)
        value = x @ V + c
        return swish_gate * value
    
    # Triton 模式需要 reshape 成 2D
    bsz, seqlen, in_feat = x.shape
    x2d = x.reshape(bsz * seqlen, in_feat).contiguous()
    
    if backend == "fully_fused":
        # 完全融合 - 单 Kernel
        out2d = fully_fused_swiglu(x2d, W, V, b, c, beta)
    elif backend == "dual_fused":
        # 双 Kernel 融合
        swish_gate = fused_matmul_bias_swish(x2d, W, b, beta)
        value = fused_matmul_bias(x2d, V, c)
        out2d = swish_gate * value
    elif backend == "unfused":
        # Triton 非融合
        gate = triton_matmul(x2d, W) + b
        value = triton_matmul(x2d, V) + c
        swish_gate = gate * torch.sigmoid(beta * gate)
        out2d = swish_gate * value
    else:
        # fallback to PyTorch
        gate = x @ W + b
        swish_gate = gate * torch.sigmoid(beta * gate)
        value = x @ V + c
        return swish_gate * value

    return out2d.reshape(bsz, seqlen, -1)
