"""
FlashAttention implementation using Triton and TileLang.

This implements the forward pass of FlashAttention with online softmax,
which avoids materializing the full attention matrix in HBM.

Key optimizations:
1. Tiling: Process Q, K, V in blocks that fit in SRAM
2. Online Softmax: Compute softmax incrementally without full attention matrix
3. Memory efficiency: O(N) memory instead of O(N^2)

Environment Variables:
- FLASH_ATTN_BACKEND=pytorch: Use PyTorch reference implementation
- FLASH_ATTN_BACKEND=triton: Use Triton FlashAttention implementation (default)
- FLASH_ATTN_BACKEND=fa3: Use FlashAttention-3 library
- FLASH_ATTN_BACKEND=tilelang: Use TileLang FlashAttention implementation
"""

import os
import math
import torch
import triton
import triton.language as tl

from task import input_t, output_t


# ============================================================================
# TileLang FlashAttention Implementation
# ============================================================================
# Cache for compiled TileLang kernels to avoid recompilation
_tilelang_kernel_cache = {}


def _get_tilelang_kernel(batch, heads, seq_len, dim):
    """
    Get or create a compiled TileLang FlashAttention kernel.
    
    This kernel directly supports (batch, heads, seq_len, dim) layout,
    avoiding the need for transpose operations.
    """
    import tilelang
    import tilelang.language as T
    
    @tilelang.jit(
        out_idx=[3],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    )
    def flashattn(batch, heads, seq_len, dim, is_causal,
                  block_M=128, block_N=128, num_stages=2, threads=256):
        scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
        # Shape: (batch, heads, seq_len, dim) - BHSD layout
        shape = [batch, heads, seq_len, dim]
        dtype = "float16"
        accum_dtype = "float"

        @T.macro
        def MMA0(
            K: T.Tensor(shape, dtype),
            Q_shared: T.SharedBuffer([block_M, dim], dtype),
            K_shared: T.SharedBuffer([block_N, dim], dtype),
            acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
            k: T.int32,
            bx: T.int32,
            by: T.int32,
            bz: T.int32,
        ):
            # K[batch, head, seq, dim] -> K[bz, by, k*block_N:(k+1)*block_N, :]
            T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], K_shared)
            if is_causal:
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0,
                                                 -T.infinity(acc_s.dtype))
            else:
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(k * block_N + j >= seq_len, -T.infinity(acc_s.dtype), 0)
            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def MMA1(
            V: T.Tensor(shape, dtype),
            V_shared: T.SharedBuffer([block_N, dim], dtype),
            acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
            acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
            k: T.int32,
            by: T.int32,
            bz: T.int32,
        ):
            # V[batch, head, seq, dim] -> V[bz, by, k*block_N:(k+1)*block_N, :]
            T.copy(V[bz, by, k * block_N:(k + 1) * block_N, :], V_shared)
            T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def Softmax(
                acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
                acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
                scores_max: T.FragmentBuffer([block_M], accum_dtype),
                scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),
                scores_scale: T.FragmentBuffer([block_M], accum_dtype),
                scores_sum: T.FragmentBuffer([block_M], accum_dtype),
                logsum: T.FragmentBuffer([block_M], accum_dtype),
        ):
            T.copy(scores_max, scores_max_prev)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            for i in T.Parallel(block_M):
                scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
            for i in T.Parallel(block_M):
                scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
            T.reduce_sum(acc_s, scores_sum, dim=1)
            for i in T.Parallel(block_M):
                logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            T.copy(acc_s, acc_s_cast)

        @T.macro
        def Rescale(
                acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
                scores_scale: T.FragmentBuffer([block_M], accum_dtype),
        ):
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] *= scores_scale[i]

        @T.prim_func
        def main(
                Q: T.Tensor(shape, dtype),
                K: T.Tensor(shape, dtype),
                V: T.Tensor(shape, dtype),
                Output: T.Tensor(shape, dtype),
        ):
            # Grid: (seq_blocks, heads, batch)
            with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([block_M, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)

                # Q[batch, head, seq, dim] -> Q[bz, by, bx*block_M:(bx+1)*block_M, :]
                T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = (
                    T.min(T.ceildiv(seq_len, block_N), T.ceildiv((bx + 1) * block_M, block_N))
                    if is_causal else T.ceildiv(seq_len, block_N)
                )

                for k in T.Pipelined(
                        loop_range,
                        num_stages=num_stages,
                        order=[-1, 0, 3, 1, -1, 2],
                        stage=[-1, 0, 0, 1, -1, 1],
                        group=[[0], [1, 2], [3, 4, 5, 6, 7, 8, 9, 10, 11], [12], [13], [14]]):
                    MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                    Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale, scores_sum, logsum)
                    Rescale(acc_o, scores_scale)
                    MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, O_shared)
                # Output[batch, head, seq, dim] -> Output[bz, by, bx*block_M:(bx+1)*block_M, :]
                T.copy(O_shared, Output[bz, by, bx * block_M:(bx + 1) * block_M, :])

        return main

    # Call the decorated function to get compiled kernel
    return flashattn(batch, heads, seq_len, dim, is_causal=False,
                     block_M=128, block_N=128, num_stages=2, threads=256)


def tilelang_flash_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    TileLang FlashAttention forward pass.
    
    Args:
        Q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        K: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        V: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
    
    Returns:
        Output tensor of shape (batch_size, num_heads, seq_len, head_dim)
    """
    BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
    
    # Ensure inputs are contiguous (no transpose needed - kernel supports BHSD layout directly)
    # Q_tl = Q.contiguous()
    # K_tl = K.contiguous()
    # V_tl = V.contiguous()
    
    # Create cache key
    cache_key = (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    
    # Get or create compiled kernel
    if cache_key not in _tilelang_kernel_cache:
        _tilelang_kernel_cache[cache_key] = _get_tilelang_kernel(
            BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
        )
    
    kernel = _tilelang_kernel_cache[cache_key]
    
    # Run kernel - output is already in (B, H, S, D) format
    return kernel(Q_tl, K_tl, V_tl)


@triton.jit
def _flash_attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    """
    Inner loop for FlashAttention forward pass.
    Iterates over K, V blocks and accumulates the output using online softmax.
    """
    # Loop over all K, V blocks
    for start_kv in range(0, SEQ_LEN, BLOCK_SIZE_KV):
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)
        
        # Load K block and compute QK^T
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)
        
        # Apply softmax scale and compute row-wise max
        # m_ij = max(m_i, rowmax(QK * scale))
        m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
        
        # Compute P = exp(QK * scale - m_ij)
        QK_block = QK_block * softmax_scale - m_ij[:, None]
        P_block = tl.math.exp(QK_block)
        
        # Compute row sum of P
        l_ij = tl.sum(P_block, 1)
        
        # Correction factor for previous accumulator
        alpha = tl.math.exp(m_i - m_ij)
        
        # Update running sum: l_i = l_i * alpha + l_ij
        l_i = l_i * alpha + l_ij
        
        # Load V block
        V_block = tl.load(V_block_ptr)
        
        # Update output accumulator: O = O * alpha + P @ V
        P_block = P_block.to(tl.float16)
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)
        
        # Update running max
        m_i = m_ij
        
        # Advance pointers to next K, V blocks
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
    
    return O_block, l_i, m_i


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [128]
        for BLOCK_SIZE_KV in [128]
        for num_stages in [3]
        for num_warps in [8]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _flash_attn_fwd(
    Q,  # (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    K,  # (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    V,  # (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    O,  # (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    softmax_scale,
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
):
    """
    FlashAttention forward kernel.
    
    Each program instance handles one block of queries for one head in one batch.
    """
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)
    
    # Get program IDs
    block_idx_q = tl.program_id(0)  # Which Q block
    idx_batch_head = tl.program_id(1)  # Which batch and head
    
    # Compute batch and head indices
    idx_batch = idx_batch_head // NUM_HEADS
    idx_head = idx_batch_head % NUM_HEADS
    
    # Compute base offset for this batch and head
    qkv_offset = (
        idx_batch.to(tl.int64) * stride_Q_batch
        + idx_head.to(tl.int64) * stride_Q_head
    )
    
    # Create block pointers for Q, K, V, O
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qkv_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(block_idx_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )
    
    # K is transposed for matmul: (HEAD_DIM, SEQ_LEN)
    # We invert the strides to transpose K implicitly
    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_offset,
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(stride_K_dim, stride_K_seq),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1),
    )
    
    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )
    
    O_block_ptr = tl.make_block_ptr(
        base=O + qkv_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_idx_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )
    
    # Initialize accumulators
    # m_i: running max for each query (for numerical stability)
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    # l_i: running sum of exp for each query (for normalization)
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    # O_block: output accumulator
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)
    
    # Load Q block (stays in SRAM throughout)
    Q_block = tl.load(Q_block_ptr)
    
    # Process all K, V blocks (non-causal attention)
    O_block, l_i, m_i = _flash_attn_fwd_inner(
        O_block,
        l_i,
        m_i,
        Q_block,
        K_block_ptr,
        V_block_ptr,
        softmax_scale,
        BLOCK_SIZE_Q,
        BLOCK_SIZE_KV,
        HEAD_DIM,
        SEQ_LEN,
    )
    
    # Normalize output by the sum of exponentials
    O_block = O_block / l_i[:, None]
    
    # Store output
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))


def flash_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    FlashAttention forward pass.
    
    Args:
        Q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        K: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        V: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
    
    Returns:
        Output tensor of shape (batch_size, num_heads, seq_len, head_dim)
    """
    BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
    
    assert Q.shape == K.shape == V.shape
    assert HEAD_DIM in {16, 32, 64, 128, 256}
    
    # Softmax scale factor: 1 / sqrt(d_k)
    softmax_scale = 1.0 / math.sqrt(HEAD_DIM)
    
    # Allocate output tensor
    O = torch.empty_like(Q)
    
    # Launch kernel
    grid = lambda META: (
        triton.cdiv(SEQ_LEN, META["BLOCK_SIZE_Q"]),
        BATCH_SIZE * NUM_HEADS,
    )
    
    _flash_attn_fwd[grid](
        Q, K, V, O,
        softmax_scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        BATCH_SIZE,
        NUM_HEADS,
        SEQ_LEN,
        HEAD_DIM,
    )
    
    return O


def pytorch_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    PyTorch reference implementation of Scaled Dot Product Attention.
    Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
    
    Args:
        Q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        K: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        V: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
    
    Returns:
        Output tensor of shape (batch_size, num_heads, seq_len, head_dim)
    """
    d_k = Q.size(-1)
    
    # 1. Compute attention scores: QK^T / sqrt(d)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 2. Apply softmax
    attn_probs = torch.softmax(scores, dim=-1)
    
    # 3. Compute output: softmax(scores) @ V
    output = torch.matmul(attn_probs, V)
    return output


def custom_kernel(data: input_t) -> output_t:
    """
    FlashAttention implementation.
    
    Environment Variables:
    - FLASH_ATTN_BACKEND=pytorch: Use PyTorch reference implementation
    - FLASH_ATTN_BACKEND=triton: Use Triton FlashAttention implementation (default)
    - FLASH_ATTN_BACKEND=fa3: Use FlashAttention-3 library
    - FLASH_ATTN_BACKEND=tilelang: Use TileLang FlashAttention implementation
    
    Args:
        data: tuple of (Q, K, V)
            Q: (batch_size, num_heads, seq_len, head_dim)
            K: (batch_size, num_heads, seq_len, head_dim)
            V: (batch_size, num_heads, seq_len, head_dim)
    
    Returns:
        Output tensor of shape (batch_size, num_heads, seq_len, head_dim)
    """
    backend = os.getenv("FLASH_ATTN_BACKEND", "triton").lower()
    Q, K, V = data
    
    if backend == "pytorch":
        # PyTorch reference implementation
        return pytorch_attention(Q, K, V)
    elif backend == "fa3":
        # FlashAttention-3 library
        import flash_attn_interface
        
        # 维度转换 & 内存连续化: (B, N, S, D) -> (B, S, N, D)
        # FA3 强制要求输入布局为 (Batch, Seq, Head, Dim)
        q_in = Q.transpose(1, 2).contiguous()
        k_in = K.transpose(1, 2).contiguous()
        v_in = V.transpose(1, 2).contiguous()

        output = flash_attn_interface.flash_attn_func(
            q_in, k_in, v_in,
            causal=False,
            softmax_scale=None
        )
        return output.transpose(1, 2).contiguous()
    elif backend == "tilelang":
        # TileLang FlashAttention implementation
        return tilelang_flash_attention_forward(Q, K, V)
    else:
        # Triton FlashAttention implementation (default)
        return flash_attention_forward(Q, K, V)
