from __future__ import annotations

from collections import OrderedDict

import torch


def fused_adazero(x: torch.Tensor, eps: float, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    if not _can_use_fused_adazero(x, scale, shift):
        return torch.nn.functional.rms_norm(x, (x.shape[-1],), eps=eps) * (1 + scale) + shift

    import triton
    import triton.language as tl

    y = torch.empty_like(x)
    rows = x.numel() // x.shape[-1]
    cols = x.shape[-1]
    block_cols = triton.next_power_of_2(cols)
    scale_flat = scale.reshape(-1, cols)
    shift_flat = shift.reshape(-1, cols)
    scale_rows = scale_flat.shape[0]
    _adazero_kernel[(rows,)](
        x,
        scale_flat,
        shift_flat,
        y,
        rows,
        cols,
        scale_rows,
        scale_flat.stride(0),
        scale_flat.stride(1),
        shift_flat.stride(0),
        shift_flat.stride(1),
        eps,
        BLOCK_COLS=block_cols,
    )
    return y


def fused_adazero_from_ada(
    x: torch.Tensor,
    eps: float,
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
    *,
    shift_index: int,
    scale_index: int,
    counter: str = "video",
) -> torch.Tensor:
    if not _can_use_ada_from_table(x, scale_shift_table, timestep):
        shift, scale = _torch_ada_values_from_table(
            scale_shift_table,
            timestep,
            x.shape[0],
            x.shape[1],
            x.shape[-1],
            shift_index,
            scale_index,
        )
        return fused_adazero(x, eps, scale, shift)

    import triton

    _increment_ada_counter(counter)

    ts_view = timestep.reshape(x.shape[0], timestep.shape[1], scale_shift_table.shape[0], x.shape[-1])
    y = torch.empty_like(x)
    rows = x.numel() // x.shape[-1]
    cols = x.shape[-1]
    _adazero_from_ada_kernel[(rows,)](
        x,
        scale_shift_table,
        ts_view,
        y,
        rows,
        cols,
        x.shape[1],
        scale_shift_table.stride(0),
        scale_shift_table.stride(1),
        ts_view.stride(0),
        ts_view.stride(1),
        ts_view.stride(2),
        ts_view.stride(3),
        shift_index,
        scale_index,
        scale_shift_table.dtype != timestep.dtype,
        timestep.dtype is torch.bfloat16,
        eps,
        BLOCK_COLS=triton.next_power_of_2(cols),
    )
    return y


def fused_adaln_affine_from_ada(
    x: torch.Tensor,
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
    *,
    shift_index: int,
    scale_index: int,
) -> torch.Tensor:
    if not _can_use_ada_from_table(x, scale_shift_table, timestep):
        shift, scale = _torch_ada_values_from_table(
            scale_shift_table,
            timestep,
            x.shape[0],
            x.shape[1],
            x.shape[-1],
            shift_index,
            scale_index,
        )
        return x * (1 + scale) + shift

    import triton

    global _VIDEO_TEXT_ADALN_CALLS
    _VIDEO_TEXT_ADALN_CALLS += 1

    ts_view = timestep.reshape(x.shape[0], timestep.shape[1], scale_shift_table.shape[0], x.shape[-1])
    y = torch.empty_like(x)
    _adaln_affine_from_ada_kernel[(triton.cdiv(x.numel(), 1024),)](
        x,
        scale_shift_table,
        ts_view,
        y,
        x.numel(),
        x.shape[-1],
        x.shape[1],
        scale_shift_table.stride(0),
        scale_shift_table.stride(1),
        ts_view.stride(0),
        ts_view.stride(1),
        ts_view.stride(2),
        ts_view.stride(3),
        shift_index,
        scale_index,
        scale_shift_table.dtype != timestep.dtype,
        timestep.dtype is torch.bfloat16,
        BLOCK=1024,
    )
    return y


def fused_fp8_quantize_e4m3(x: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
    if not _can_use_fused_fp8_quantize(x, input_scale):
        fp8_min = torch.finfo(torch.float8_e4m3fn).min
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        return torch.clamp(x * input_scale.reciprocal(), fp8_min, fp8_max).to(torch.float8_e4m3fn)

    import triton

    q = torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=torch.float8_e4m3fn)
    n = x.numel()
    _fp8_quantize_e4m3_kernel[(triton.cdiv(n, 1024),)](
        x,
        input_scale,
        q,
        n,
        BLOCK=1024,
    )
    return q


def fused_gelu_tanh_fp8_quantize_e4m3(x: torch.Tensor, input_scale: torch.Tensor) -> torch.Tensor:
    if not _can_use_fused_fp8_quantize(x, input_scale):
        gelu = torch.nn.functional.gelu(x, approximate="tanh")
        fp8_min = torch.finfo(torch.float8_e4m3fn).min
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        return torch.clamp(gelu * input_scale.reciprocal(), fp8_min, fp8_max).to(torch.float8_e4m3fn)

    import triton

    q = torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=torch.float8_e4m3fn)
    n = x.numel()
    _gelu_tanh_fp8_quantize_e4m3_kernel[(triton.cdiv(n, 1024),)](
        x,
        input_scale,
        q,
        n,
        BLOCK=1024,
    )
    return q


def fused_bias_add_bf16(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    if not _can_use_fused_bias_add(x, bias):
        return x + bias.to(x.dtype)

    import triton

    out = torch.empty_like(x)
    _bias_add_bf16_kernel[(triton.cdiv(x.numel(), 1024),)](
        x,
        bias,
        out,
        x.numel(),
        x.shape[-1],
        bias.dtype != x.dtype,
        BLOCK=1024,
    )
    return out


def fused_residual_gate_bias_from_ada(
    x: torch.Tensor,
    y_no_bias: torch.Tensor,
    bias: torch.Tensor,
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
    *,
    gate_index: int,
    count_video_ada: bool = True,
) -> torch.Tensor:
    if (
        not _can_use_ada_from_table(x, scale_shift_table, timestep)
        or not y_no_bias.is_contiguous()
        or y_no_bias.shape != x.shape
        or not _can_use_fused_bias_add(y_no_bias, bias)
    ):
        y = y_no_bias + bias.to(y_no_bias.dtype)
        return fused_residual_gate_from_ada(
            x,
            y,
            scale_shift_table,
            timestep,
            gate_index=gate_index,
            count_video_ada=count_video_ada,
        )

    import triton

    global _VIDEO_ADA_VALUES_CALLS, _VIDEO_TEXT_ADALN_CALLS
    if count_video_ada:
        _VIDEO_ADA_VALUES_CALLS += 1
    else:
        _VIDEO_TEXT_ADALN_CALLS += 1

    ts_view = timestep.reshape(x.shape[0], timestep.shape[1], scale_shift_table.shape[0], x.shape[-1])
    out = torch.empty_like(x)
    _residual_gate_bias_from_ada_kernel[(triton.cdiv(x.numel(), 1024),)](
        x,
        y_no_bias,
        bias,
        scale_shift_table,
        ts_view,
        out,
        x.numel(),
        x.shape[-1],
        x.shape[1],
        bias.dtype != y_no_bias.dtype,
        scale_shift_table.stride(0),
        scale_shift_table.stride(1),
        ts_view.stride(0),
        ts_view.stride(1),
        ts_view.stride(2),
        ts_view.stride(3),
        gate_index,
        scale_shift_table.dtype != timestep.dtype,
        timestep.dtype is torch.bfloat16,
        BLOCK=1024,
    )
    return out


def fused_fp8_quantize_e4m3_per_head_4d(
    x: torch.Tensor,
    *,
    reduce_block: int = 1024,
    quant_block: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize FA3 Q/K/V tensors with one E4M3 scale per batch/head.

    The hot LTX-2.3 video self-attention shape family is [1, Nv, 32, 128].
    FlashAttention-3 FP8 accepts Q/K/V in [B, N, H, D] plus descale tensors in
    [B, H], so this keeps the scale contract narrow and benchmarkable.
    """

    if not _can_use_fused_fp8_quantize_per_head_4d(x):
        return _torch_fp8_quantize_e4m3_per_head_4d(x)

    import triton

    batch, tokens, heads, head_dim = x.shape
    elems_per_head = tokens * head_dim
    blocks_per_head = triton.cdiv(elems_per_head, reduce_block)
    partial = torch.empty((batch, heads, blocks_per_head), device=x.device, dtype=torch.float32)
    scale = torch.empty((batch, heads), device=x.device, dtype=torch.float32)
    q = torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=torch.float8_e4m3fn)

    _fp8_per_head_amax_kernel[(batch * heads * blocks_per_head,)](
        x,
        partial,
        tokens,
        heads,
        head_dim,
        elems_per_head,
        blocks_per_head,
        BLOCK=reduce_block,
    )
    _fp8_per_head_scale_kernel[(batch * heads,)](
        partial,
        scale,
        blocks_per_head,
        BLOCK=triton.next_power_of_2(blocks_per_head),
    )
    _fp8_per_head_quantize_kernel[(batch * heads * triton.cdiv(elems_per_head, quant_block),)](
        x,
        scale,
        q,
        tokens,
        heads,
        head_dim,
        elems_per_head,
        triton.cdiv(elems_per_head, quant_block),
        BLOCK=quant_block,
    )
    return q, scale


def fused_video_qk_rmsnorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse exact video self-attention Q/K RMSNorm and split RoPE.

    Contract: q/k are [1, Nv, 4096] BF16, cos/sin are [B, 32, Nv, 64].
    The math mirrors ltx_core.model.transformer.ops.PytorchPreAttention for the
    LTXRopeType.SPLIT path and keeps BF16 outputs for FA3.
    """

    if not _can_use_fused_video_qk_rmsnorm_rope(q, k, q_weight, k_weight, cos, sin):
        return _torch_video_qk_rmsnorm_rope(q, k, q_weight, k_weight, eps, cos, sin)

    import triton

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    rows = q.shape[0] * q.shape[1]
    cols = q.shape[-1]
    heads = cos.shape[1]
    tokens = q.shape[1]
    head_dim = cols // heads
    half_dim = head_dim // 2
    block_cols = triton.next_power_of_2(cols)

    grid = (rows,)
    _video_qk_rmsnorm_rope_kernel[grid](
        q,
        k,
        q_weight,
        k_weight,
        cos,
        sin,
        q_out,
        k_out,
        rows,
        cols,
        tokens,
        heads,
        head_dim,
        half_dim,
        cos.stride(0),
        cos.stride(1),
        cos.stride(2),
        cos.stride(3),
        sin.stride(0),
        sin.stride(1),
        sin.stride(2),
        sin.stride(3),
        eps,
        BLOCK_COLS=block_cols,
    )
    return q_out, k_out


def fused_video_qkv_rmsnorm_rope_pack(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor | None:
    """Fuse exact Q/K preattention and pack Q/K/V for FA3 qkvpacked forward."""

    if not _can_use_fused_video_qkv_rmsnorm_rope_pack(q, k, v, q_weight, k_weight, cos, sin):
        return None

    import triton

    batch, tokens, cols = q.shape
    heads = cos.shape[1]
    head_dim = cols // heads
    qkv = torch.empty((batch, tokens, 3, heads, head_dim), device=q.device, dtype=q.dtype)

    _video_qkv_rmsnorm_rope_pack_kernel[(batch * tokens,)](
        q,
        k,
        v,
        q_weight,
        k_weight,
        cos,
        sin,
        qkv,
        batch * tokens,
        cols,
        tokens,
        heads,
        head_dim,
        head_dim // 2,
        cos.stride(0),
        cos.stride(1),
        cos.stride(2),
        cos.stride(3),
        sin.stride(0),
        sin.stride(1),
        sin.stride(2),
        sin.stride(3),
        eps,
        BLOCK_COLS=triton.next_power_of_2(cols),
    )
    return qkv


def fused_video_qk_rmsnorm_rope_separate(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Old two-launch Q/K path, kept only for paired kernel benchmarking."""

    if not _can_use_fused_video_qk_rmsnorm_rope(q, k, q_weight, k_weight, cos, sin):
        return _torch_video_qk_rmsnorm_rope(q, k, q_weight, k_weight, eps, cos, sin)

    import triton

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    rows = q.shape[0] * q.shape[1]
    cols = q.shape[-1]
    heads = cos.shape[1]
    tokens = q.shape[1]
    head_dim = cols // heads
    half_dim = head_dim // 2
    block_cols = triton.next_power_of_2(cols)

    grid = (rows,)
    _video_rmsnorm_rope_kernel[grid](
        q,
        q_weight,
        cos,
        sin,
        q_out,
        rows,
        cols,
        tokens,
        heads,
        head_dim,
        half_dim,
        cos.stride(0),
        cos.stride(1),
        cos.stride(2),
        cos.stride(3),
        sin.stride(0),
        sin.stride(1),
        sin.stride(2),
        sin.stride(3),
        eps,
        BLOCK_COLS=block_cols,
    )
    _video_rmsnorm_rope_kernel[grid](
        k,
        k_weight,
        cos,
        sin,
        k_out,
        rows,
        cols,
        tokens,
        heads,
        head_dim,
        half_dim,
        cos.stride(0),
        cos.stride(1),
        cos.stride(2),
        cos.stride(3),
        sin.stride(0),
        sin.stride(1),
        sin.stride(2),
        sin.stride(3),
        eps,
        BLOCK_COLS=block_cols,
    )
    return q_out, k_out


def fused_video_qk_bias_rmsnorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _can_use_fused_video_qk_bias_rmsnorm_rope(q, k, q_bias, k_bias, q_weight, k_weight, cos, sin):
        q = q + q_bias.to(q.dtype)
        k = k + k_bias.to(k.dtype)
        return _torch_video_qk_rmsnorm_rope(q, k, q_weight, k_weight, eps, cos, sin)

    import triton

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    rows = q.shape[0] * q.shape[1]
    cols = q.shape[-1]
    heads = cos.shape[1]
    tokens = q.shape[1]
    head_dim = cols // heads
    half_dim = head_dim // 2
    block_cols = triton.next_power_of_2(cols)

    grid = (rows,)
    _video_bias_rmsnorm_rope_kernel[grid](
        q,
        q_bias,
        q_weight,
        cos,
        sin,
        q_out,
        rows,
        cols,
        tokens,
        heads,
        head_dim,
        half_dim,
        q_bias.dtype != q.dtype,
        cos.stride(0),
        cos.stride(1),
        cos.stride(2),
        cos.stride(3),
        sin.stride(0),
        sin.stride(1),
        sin.stride(2),
        sin.stride(3),
        eps,
        BLOCK_COLS=block_cols,
    )
    _video_bias_rmsnorm_rope_kernel[grid](
        k,
        k_bias,
        k_weight,
        cos,
        sin,
        k_out,
        rows,
        cols,
        tokens,
        heads,
        head_dim,
        half_dim,
        k_bias.dtype != k.dtype,
        cos.stride(0),
        cos.stride(1),
        cos.stride(2),
        cos.stride(3),
        sin.stride(0),
        sin.stride(1),
        sin.stride(2),
        sin.stride(3),
        eps,
        BLOCK_COLS=block_cols,
    )
    return q_out, k_out


def fused_residual_gate_bf16(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    if not _can_use_fused_residual_gate(x, y, gate):
        return x + y * gate

    import triton

    out = torch.empty_like(x)
    cols = x.shape[-1]
    gate_flat = gate.reshape(-1, cols)
    _residual_gate_kernel[(triton.cdiv(x.numel(), 1024),)](
        x,
        y,
        gate_flat,
        out,
        x.numel(),
        cols,
        gate_flat.shape[0],
        gate_flat.stride(0),
        gate_flat.stride(1),
        BLOCK=1024,
    )
    return out


def fused_head_gate_mul_bf16(out: torch.Tensor, gates: torch.Tensor, heads: int, dim_head: int) -> torch.Tensor:
    if not _can_use_fused_head_gate_mul(out, gates, heads, dim_head):
        batch, tokens, _ = out.shape
        return (out.view(batch, tokens, heads, dim_head) * gates.unsqueeze(-1)).view(
            batch, tokens, heads * dim_head
        )

    y = torch.empty_like(out)
    rows = out.numel() // out.shape[-1]
    cols = out.shape[-1]
    gates_flat = gates.reshape(-1, heads)
    _head_gate_mul_kernel[(triton.cdiv(out.numel(), 1024),)](
        out,
        gates_flat,
        y,
        out.numel(),
        cols,
        heads,
        dim_head,
        gates_flat.stride(0),
        gates_flat.stride(1),
        BLOCK=1024,
    )
    return y


def fused_residual_gate_from_ada(
    x: torch.Tensor,
    y: torch.Tensor,
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
    *,
    gate_index: int,
    count_video_ada: bool = True,
    counter: str | None = None,
) -> torch.Tensor:
    if not _can_use_ada_from_table(x, scale_shift_table, timestep) or not y.is_contiguous() or y.shape != x.shape:
        (gate,) = _torch_ada_values_from_table(
            scale_shift_table,
            timestep,
            x.shape[0],
            x.shape[1],
            x.shape[-1],
            gate_index,
        )
        return _simple_residual_gate(x, y, gate)

    import triton

    if counter is not None:
        _increment_ada_counter(counter)
    elif count_video_ada:
        _increment_ada_counter("video")
    else:
        _increment_ada_counter("video_text")

    ts_view = timestep.reshape(x.shape[0], timestep.shape[1], scale_shift_table.shape[0], x.shape[-1])
    out = torch.empty_like(x)
    _residual_gate_from_ada_kernel[(triton.cdiv(x.numel(), 1024),)](
        x,
        y,
        scale_shift_table,
        ts_view,
        out,
        x.numel(),
        x.shape[-1],
        x.shape[1],
        scale_shift_table.stride(0),
        scale_shift_table.stride(1),
        ts_view.stride(0),
        ts_view.stride(1),
        ts_view.stride(2),
        ts_view.stride(3),
        gate_index,
        scale_shift_table.dtype != timestep.dtype,
        timestep.dtype is torch.bfloat16,
        BLOCK=1024,
    )
    return out


def fused_mul_from_ada(
    x: torch.Tensor,
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
    *,
    gate_index: int,
) -> torch.Tensor:
    if not _can_use_ada_from_table(x, scale_shift_table, timestep):
        (gate,) = _torch_ada_values_from_table(
            scale_shift_table,
            timestep,
            x.shape[0],
            x.shape[1],
            x.shape[-1],
            gate_index,
        )
        return x * gate

    import triton

    global _VIDEO_TEXT_ADALN_CALLS
    _VIDEO_TEXT_ADALN_CALLS += 1

    ts_view = timestep.reshape(x.shape[0], timestep.shape[1], scale_shift_table.shape[0], x.shape[-1])
    out = torch.empty_like(x)
    _mul_from_ada_kernel[(triton.cdiv(x.numel(), 1024),)](
        x,
        scale_shift_table,
        ts_view,
        out,
        x.numel(),
        x.shape[-1],
        x.shape[1],
        scale_shift_table.stride(0),
        scale_shift_table.stride(1),
        ts_view.stride(0),
        ts_view.stride(1),
        ts_view.stride(2),
        ts_view.stride(3),
        gate_index,
        scale_shift_table.dtype != timestep.dtype,
        timestep.dtype is torch.bfloat16,
        BLOCK=1024,
    )
    return out


def fp8_linear_from_quantized_e4m3(linear: torch.nn.Module, qinput: torch.Tensor, *, out_dtype: torch.dtype) -> torch.Tensor:
    origin_shape = qinput.shape
    qflat = qinput.reshape(-1, qinput.shape[-1]) if qinput.dim() == 3 else qinput
    output = torch._scaled_mm(
        qflat,
        linear.weight.t(),
        scale_a=linear.input_scale,
        scale_b=linear.weight_scale,
        out_dtype=out_dtype,
        use_fast_accum=True,
    )
    if linear.bias is not None:
        output = output + linear.bias.to(output.dtype)
    if output.dim() != len(origin_shape):
        output_shape = list(origin_shape)
        output_shape[-1] = output.shape[-1]
        output = output.reshape(output_shape)
    return output


def fp8_linear_without_bias(linear: torch.nn.Module, x: torch.Tensor, *, out_dtype: torch.dtype) -> torch.Tensor | None:
    if not _can_use_fp8_linear_without_bias(linear, x):
        return None

    origin_shape = x.shape
    qinput = fused_fp8_quantize_e4m3(x.contiguous(), linear.input_scale)
    qflat = qinput.reshape(-1, qinput.shape[-1]) if qinput.dim() == 3 else qinput
    output = torch._scaled_mm(
        qflat,
        linear.weight.t(),
        scale_a=linear.input_scale,
        scale_b=linear.weight_scale,
        out_dtype=out_dtype,
        use_fast_accum=True,
    )
    if output.dim() != len(origin_shape):
        output_shape = list(origin_shape)
        output_shape[-1] = output.shape[-1]
        output = output.reshape(output_shape)
    return output


def _bf16_linear_without_bias(linear: torch.nn.Module, x: torch.Tensor) -> torch.Tensor | None:
    if not _can_use_bf16_linear_without_bias(linear, x):
        return None
    return torch.nn.functional.linear(x, linear.weight, bias=None)


def _linear_without_bias_for_video_msa(
    linear: torch.nn.Module,
    x: torch.Tensor,
    *,
    out_dtype: torch.dtype,
    allow_bf16_linear: bool,
) -> torch.Tensor | None:
    projected = fp8_linear_without_bias(linear, x, out_dtype=out_dtype)
    if projected is not None:
        return projected
    if allow_bf16_linear:
        return _bf16_linear_without_bias(linear, x)
    return None


def fp8_qkv_from_shared_quantized_input(
    attn: torch.nn.Module,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Run video self-attention Q/K/V FP8 linears from one exact quantized input."""

    if not _can_use_video_qkv_quant_reuse(attn, x):
        return None

    qinput = fused_fp8_quantize_e4m3(x.contiguous(), attn.to_q.input_scale)
    return (
        fp8_linear_from_quantized_e4m3(attn.to_q, qinput, out_dtype=x.dtype),
        fp8_linear_from_quantized_e4m3(attn.to_k, qinput, out_dtype=x.dtype),
        fp8_linear_from_quantized_e4m3(attn.to_v, qinput, out_dtype=x.dtype),
    )


def fp8_qkv_from_packed_linear(
    attn: torch.nn.Module,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Run exact video self-attention Q/K/V projections as one packed FP8 matmul."""

    if not _can_use_video_qkv_packed_linear(attn, x):
        return None

    try:
        packed = _get_video_qkv_packed_cache(attn)
        qinput = fused_fp8_quantize_e4m3(x.contiguous(), attn.to_q.input_scale)
        qflat = qinput.reshape(-1, qinput.shape[-1]) if qinput.dim() == 3 else qinput
        output = torch._scaled_mm(
            qflat,
            packed["weight"].t(),
            scale_a=attn.to_q.input_scale,
            scale_b=attn.to_q.weight_scale,
            out_dtype=x.dtype,
            use_fast_accum=True,
        )
        output = output + packed["bias"].to(output.dtype)
        if output.dim() != len(x.shape):
            output = output.reshape(x.shape[0], x.shape[1], output.shape[-1])
        return output.split(attn.to_q.out_features, dim=-1)
    except Exception as exc:
        _record_video_qkv_packed_linear_fallback(type(exc).__name__)
        return None


def fp8_qkv_from_packed_requant(
    attn: torch.nn.Module,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Run Q/K/V as one packed FP8 matmul after common-scale requantization.

    This is not exact to the checkpoint weights because torch._scaled_mm accepts
    one weight scale for the packed matrix. Keep it behind an explicit
    quality-risk flag and real-output checks.
    """

    if not _can_use_video_qkv_packed_requant(attn, x):
        return None

    try:
        packed = _get_video_qkv_packed_requant_cache(attn)
        qinput = fused_fp8_quantize_e4m3(x.contiguous(), attn.to_q.input_scale)
        qflat = qinput.reshape(-1, qinput.shape[-1]) if qinput.dim() == 3 else qinput
        output = torch._scaled_mm(
            qflat,
            packed["weight"].t(),
            scale_a=attn.to_q.input_scale,
            scale_b=packed["weight_scale"],
            out_dtype=x.dtype,
            use_fast_accum=True,
        )
        output = output + packed["bias"].to(output.dtype)
        if output.dim() != len(x.shape):
            output = output.reshape(x.shape[0], x.shape[1], output.shape[-1])
        q, k, v = output.split(attn.to_q.out_features, dim=-1)
        _record_video_qkv_packed_requant_check(attn, x, q, k, v)
        return q, k, v
    except Exception as exc:
        _record_video_qkv_packed_requant_fallback(type(exc).__name__)
        return None


def fp8_qkv_from_grouped_mm(
    attn: torch.nn.Module,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Run exact video self-attention Q/K/V projections with grouped scaled-mm."""

    if not _can_use_video_qkv_grouped_mm(attn, x):
        return None

    try:
        packed = _get_video_qkv_grouped_mm_cache(attn, x.shape[0] * x.shape[1])
        qinput = fused_fp8_quantize_e4m3(x.contiguous(), attn.to_q.input_scale)
        qflat = qinput.reshape(-1, qinput.shape[-1]) if qinput.dim() == 3 else qinput
        grouped_input = qflat.unsqueeze(0).expand(3, -1, -1)
        output = torch._scaled_grouped_mm(
            grouped_input,
            packed["weight"],
            packed["scale_a"],
            packed["scale_b"],
            out_dtype=x.dtype,
            use_fast_accum=True,
        )
        output = output + packed["bias"].to(output.dtype).unsqueeze(1)
        q, k, v = (output[0], output[1], output[2])
        if x.dim() == 3:
            q = q.reshape(x.shape[0], x.shape[1], q.shape[-1])
            k = k.reshape(x.shape[0], x.shape[1], k.shape[-1])
            v = v.reshape(x.shape[0], x.shape[1], v.shape[-1])
        _record_video_qkv_grouped_mm_check(attn, x, q, k, v)
        return q, k, v
    except Exception as exc:
        _record_video_qkv_grouped_mm_fallback(type(exc).__name__)
        return None


def fp8_qk_from_grouped_mm(
    attn: torch.nn.Module,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Run exact video self-attention Q/K projections with grouped scaled-mm."""

    if not _can_use_video_qk_grouped_mm(attn, x):
        return None

    try:
        packed = _get_video_qk_grouped_mm_cache(attn, x.shape[0] * x.shape[1])
        qinput = fused_fp8_quantize_e4m3(x.contiguous(), attn.to_q.input_scale)
        qflat = qinput.reshape(-1, qinput.shape[-1]) if qinput.dim() == 3 else qinput
        grouped_input = qflat.unsqueeze(0).expand(2, -1, -1)
        output = torch._scaled_grouped_mm(
            grouped_input,
            packed["weight"],
            packed["scale_a"],
            packed["scale_b"],
            out_dtype=x.dtype,
            use_fast_accum=True,
        )
        output = output + packed["bias"].to(output.dtype).unsqueeze(1)
        q, k = output[0], output[1]
        if x.dim() == 3:
            q = q.reshape(x.shape[0], x.shape[1], q.shape[-1])
            k = k.reshape(x.shape[0], x.shape[1], k.shape[-1])
        _record_video_qk_grouped_mm_check(attn, x, q, k)
        return q, k
    except Exception as exc:
        _record_video_qk_grouped_mm_fallback(type(exc).__name__)
        return None


def _torch_video_qk_rmsnorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from ltx_core.model.transformer.rope import LTXRopeType, apply_rotary_emb

    q = torch.nn.functional.rms_norm(q, (q.shape[-1],), weight=q_weight, eps=eps)
    k = torch.nn.functional.rms_norm(k, (k.shape[-1],), weight=k_weight, eps=eps)
    q = apply_rotary_emb(q, (cos, sin), LTXRopeType.SPLIT)
    k = apply_rotary_emb(k, (cos, sin), LTXRopeType.SPLIT)
    return q, k


def _torch_fp8_quantize_e4m3_per_head_4d(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = x.float().abs().amax(dim=(1, 3)).clamp_min(1e-12) / fp8_max
    q = torch.clamp(x.float() / scale[:, None, :, None], -fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return q, scale.to(torch.float32).contiguous()


def _torch_ada_values_from_table(
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
    batch_size: int,
    tokens: int,
    cols: int,
    *indices: int,
) -> tuple[torch.Tensor, ...]:
    ts_view = timestep.reshape(batch_size, timestep.shape[1], scale_shift_table.shape[0], -1)
    if ts_view.shape[1] != tokens or ts_view.shape[-1] != cols:
        raise ValueError(
            f"Unexpected Ada timestep shape {tuple(timestep.shape)} for x shape "
            f"({batch_size}, {tokens}, {cols}) and table {tuple(scale_shift_table.shape)}"
        )
    table = scale_shift_table.to(device=timestep.device, dtype=timestep.dtype)
    return tuple(table[index].view(1, 1, cols) + ts_view[:, :, index, :] for index in indices)


def _can_use_ada_from_table(x: torch.Tensor, scale_shift_table: torch.Tensor, timestep: torch.Tensor) -> bool:
    if triton is None:
        return False
    if not x.is_cuda or not x.is_contiguous() or x.dtype != torch.bfloat16:
        return False
    if x.ndim != 3 or x.shape[-1] not in (2048, 4096):
        return False
    if not scale_shift_table.is_cuda or scale_shift_table.dtype not in (torch.bfloat16, torch.float32):
        return False
    if scale_shift_table.ndim != 2 or scale_shift_table.shape[-1] != x.shape[-1]:
        return False
    if not timestep.is_cuda or timestep.dtype not in (torch.bfloat16, torch.float32) or timestep.ndim < 2:
        return False
    if timestep.shape[0] != x.shape[0] or timestep.shape[1] != x.shape[1]:
        return False
    if timestep.numel() != x.shape[0] * x.shape[1] * scale_shift_table.shape[0] * x.shape[-1]:
        return False
    return True


def _can_use_fused_fp8_quantize(x: torch.Tensor, input_scale: torch.Tensor) -> bool:
    return x.is_cuda and x.is_contiguous() and input_scale.numel() == 1 and x.dtype in (torch.bfloat16, torch.float16)


def _can_use_fused_bias_add(x: torch.Tensor, bias: torch.Tensor) -> bool:
    if triton is None:
        return False
    if not x.is_cuda or not x.is_contiguous() or x.dtype != torch.bfloat16:
        return False
    if x.ndim < 2 or bias.ndim != 1 or bias.shape[0] != x.shape[-1]:
        return False
    if not bias.is_cuda or bias.dtype not in (torch.bfloat16, torch.float32):
        return False
    return True


def _can_use_fp8_linear_without_bias(linear: torch.nn.Module, x: torch.Tensor) -> bool:
    if triton is None:
        return False
    try:
        from ltx_core.quantization.fp8_scaled_mm import FP8Linear
        from ltx_core.quantization.trtllm_scaled_usable import trtllm_scaled_mm_usable
    except Exception:
        return False
    if trtllm_scaled_mm_usable():
        return False
    if not isinstance(linear, FP8Linear) or linear.bias is None:
        return False
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16):
        return False
    if x.shape[-1] != linear.in_features:
        return False
    return True


def _can_use_bf16_linear_without_bias(linear: torch.nn.Module, x: torch.Tensor) -> bool:
    if not isinstance(linear, torch.nn.Linear) or linear.bias is None:
        return False
    if not x.is_cuda or x.dtype != torch.bfloat16:
        return False
    if x.shape[-1] != linear.in_features:
        return False
    if linear.weight.dtype != torch.bfloat16 or linear.bias.dtype not in (torch.bfloat16, torch.float32):
        return False
    if linear.weight.shape != (linear.out_features, linear.in_features):
        return False
    return True


def _can_use_video_qkv_quant_reuse(attn: torch.nn.Module, x: torch.Tensor) -> bool:
    if triton is None:
        return False
    try:
        from ltx_core.quantization.fp8_scaled_mm import FP8Linear
        from ltx_core.quantization.trtllm_scaled_usable import trtllm_scaled_mm_usable
    except Exception:
        return False
    if trtllm_scaled_mm_usable():
        return False
    if not x.is_cuda or not x.is_contiguous() or x.dtype not in (torch.bfloat16, torch.float16):
        return False
    if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] < 1 or x.shape[-1] != 4096:
        return False
    linears = (getattr(attn, "to_q", None), getattr(attn, "to_k", None), getattr(attn, "to_v", None))
    if not all(isinstance(linear, FP8Linear) for linear in linears):
        return False
    if not all(getattr(linear, "bias", None) is not None for linear in linears):
        return False
    cached = getattr(attn, "_ltx_qkv_quant_reuse_ok", None)
    if cached is None:
        cached = bool(
            torch.equal(attn.to_q.input_scale, attn.to_k.input_scale)
            and torch.equal(attn.to_q.input_scale, attn.to_v.input_scale)
        )
        setattr(attn, "_ltx_qkv_quant_reuse_ok", cached)
    return cached


def _can_use_video_qkv_packed_linear(attn: torch.nn.Module, x: torch.Tensor) -> bool:
    if triton is None:
        _record_video_qkv_packed_linear_fallback("no_triton")
        return False
    try:
        from ltx_core.quantization.fp8_scaled_mm import FP8Linear
        from ltx_core.quantization.trtllm_scaled_usable import trtllm_scaled_mm_usable
    except Exception:
        _record_video_qkv_packed_linear_fallback("import")
        return False
    if trtllm_scaled_mm_usable():
        _record_video_qkv_packed_linear_fallback("trtllm")
        return False
    if not x.is_cuda or not x.is_contiguous() or x.dtype not in (torch.bfloat16, torch.float16):
        _record_video_qkv_packed_linear_fallback("input")
        return False
    if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] < 1 or x.shape[-1] != 4096:
        _record_video_qkv_packed_linear_fallback("shape")
        return False
    linears = (getattr(attn, "to_q", None), getattr(attn, "to_k", None), getattr(attn, "to_v", None))
    if not all(isinstance(linear, FP8Linear) for linear in linears):
        _record_video_qkv_packed_linear_fallback("linear_type")
        return False
    if not all(getattr(linear, "bias", None) is not None for linear in linears):
        _record_video_qkv_packed_linear_fallback("bias")
        return False
    if not (
        torch.equal(attn.to_q.input_scale, attn.to_k.input_scale)
        and torch.equal(attn.to_q.input_scale, attn.to_v.input_scale)
    ):
        _record_video_qkv_packed_linear_fallback("input_scale")
        return False
    if not (
        torch.equal(attn.to_q.weight_scale, attn.to_k.weight_scale)
        and torch.equal(attn.to_q.weight_scale, attn.to_v.weight_scale)
    ):
        _record_video_qkv_packed_linear_fallback("weight_scale")
        return False
    if not (
        attn.to_q.in_features == attn.to_k.in_features == attn.to_v.in_features == x.shape[-1]
        and attn.to_q.out_features == attn.to_k.out_features == attn.to_v.out_features == x.shape[-1]
    ):
        _record_video_qkv_packed_linear_fallback("features")
        return False
    return True


def _can_use_video_qkv_packed_requant(attn: torch.nn.Module, x: torch.Tensor) -> bool:
    if triton is None:
        _record_video_qkv_packed_requant_fallback("no_triton")
        return False
    try:
        from ltx_core.quantization.fp8_scaled_mm import FP8Linear
        from ltx_core.quantization.trtllm_scaled_usable import trtllm_scaled_mm_usable
    except Exception:
        _record_video_qkv_packed_requant_fallback("import")
        return False
    if trtllm_scaled_mm_usable():
        _record_video_qkv_packed_requant_fallback("trtllm")
        return False
    if not x.is_cuda or not x.is_contiguous() or x.dtype not in (torch.bfloat16, torch.float16):
        _record_video_qkv_packed_requant_fallback("input")
        return False
    if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] < 1 or x.shape[-1] != 4096:
        _record_video_qkv_packed_requant_fallback("shape")
        return False
    linears = (getattr(attn, "to_q", None), getattr(attn, "to_k", None), getattr(attn, "to_v", None))
    if not all(isinstance(linear, FP8Linear) for linear in linears):
        _record_video_qkv_packed_requant_fallback("linear_type")
        return False
    if not all(getattr(linear, "bias", None) is not None for linear in linears):
        _record_video_qkv_packed_requant_fallback("bias")
        return False
    if not (
        torch.equal(attn.to_q.input_scale, attn.to_k.input_scale)
        and torch.equal(attn.to_q.input_scale, attn.to_v.input_scale)
    ):
        _record_video_qkv_packed_requant_fallback("input_scale")
        return False
    if not (
        attn.to_q.in_features == attn.to_k.in_features == attn.to_v.in_features == x.shape[-1]
        and attn.to_q.out_features == attn.to_k.out_features == attn.to_v.out_features == x.shape[-1]
    ):
        _record_video_qkv_packed_requant_fallback("features")
        return False
    return True


def _can_use_video_qkv_grouped_mm(attn: torch.nn.Module, x: torch.Tensor) -> bool:
    if not hasattr(torch, "_scaled_grouped_mm"):
        _record_video_qkv_grouped_mm_fallback("no_scaled_grouped_mm")
        return False
    try:
        from ltx_core.quantization.fp8_scaled_mm import FP8Linear
        from ltx_core.quantization.trtllm_scaled_usable import trtllm_scaled_mm_usable
    except Exception:
        _record_video_qkv_grouped_mm_fallback("import")
        return False
    if trtllm_scaled_mm_usable():
        _record_video_qkv_grouped_mm_fallback("trtllm")
        return False
    if not x.is_cuda or not x.is_contiguous() or x.dtype not in (torch.bfloat16, torch.float16):
        _record_video_qkv_grouped_mm_fallback("input")
        return False
    if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] < 1 or x.shape[-1] != 4096:
        _record_video_qkv_grouped_mm_fallback("shape")
        return False
    linears = (getattr(attn, "to_q", None), getattr(attn, "to_k", None), getattr(attn, "to_v", None))
    if not all(isinstance(linear, FP8Linear) for linear in linears):
        _record_video_qkv_grouped_mm_fallback("linear_type")
        return False
    if not all(getattr(linear, "bias", None) is not None for linear in linears):
        _record_video_qkv_grouped_mm_fallback("bias")
        return False
    if not (
        torch.equal(attn.to_q.input_scale, attn.to_k.input_scale)
        and torch.equal(attn.to_q.input_scale, attn.to_v.input_scale)
    ):
        _record_video_qkv_grouped_mm_fallback("input_scale")
        return False
    if not (
        attn.to_q.in_features == attn.to_k.in_features == attn.to_v.in_features == x.shape[-1]
        and attn.to_q.out_features == attn.to_k.out_features == attn.to_v.out_features == x.shape[-1]
    ):
        _record_video_qkv_grouped_mm_fallback("features")
        return False
    try:
        _scale_row(attn.to_q.input_scale, x.shape[0] * x.shape[1])
        _scale_row(attn.to_q.weight_scale, attn.to_q.out_features)
        _scale_row(attn.to_k.weight_scale, attn.to_k.out_features)
        _scale_row(attn.to_v.weight_scale, attn.to_v.out_features)
    except ValueError:
        _record_video_qkv_grouped_mm_fallback("scale_shape")
        return False
    return True


def _can_use_video_qk_grouped_mm(attn: torch.nn.Module, x: torch.Tensor) -> bool:
    if not hasattr(torch, "_scaled_grouped_mm"):
        _record_video_qk_grouped_mm_fallback("no_scaled_grouped_mm")
        return False
    try:
        from ltx_core.quantization.fp8_scaled_mm import FP8Linear
        from ltx_core.quantization.trtllm_scaled_usable import trtllm_scaled_mm_usable
    except Exception:
        _record_video_qk_grouped_mm_fallback("import")
        return False
    if trtllm_scaled_mm_usable():
        _record_video_qk_grouped_mm_fallback("trtllm")
        return False
    if not x.is_cuda or not x.is_contiguous() or x.dtype not in (torch.bfloat16, torch.float16):
        _record_video_qk_grouped_mm_fallback("input")
        return False
    if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] < 1 or x.shape[-1] != 4096:
        _record_video_qk_grouped_mm_fallback("shape")
        return False
    linears = (getattr(attn, "to_q", None), getattr(attn, "to_k", None))
    if not all(isinstance(linear, FP8Linear) for linear in linears):
        _record_video_qk_grouped_mm_fallback("linear_type")
        return False
    if not all(getattr(linear, "bias", None) is not None for linear in linears):
        _record_video_qk_grouped_mm_fallback("bias")
        return False
    if not torch.equal(attn.to_q.input_scale, attn.to_k.input_scale):
        _record_video_qk_grouped_mm_fallback("input_scale")
        return False
    if not (
        attn.to_q.in_features == attn.to_k.in_features == x.shape[-1]
        and attn.to_q.out_features == attn.to_k.out_features == x.shape[-1]
    ):
        _record_video_qk_grouped_mm_fallback("features")
        return False
    try:
        _scale_row(attn.to_q.input_scale, x.shape[0] * x.shape[1])
        _scale_row(attn.to_q.weight_scale, attn.to_q.out_features)
        _scale_row(attn.to_k.weight_scale, attn.to_k.out_features)
    except ValueError:
        _record_video_qk_grouped_mm_fallback("scale_shape")
        return False
    return True


def _scale_row(scale: torch.Tensor, length: int) -> torch.Tensor:
    scale_f = scale.detach().float()
    if scale_f.numel() == 1:
        return scale_f.reshape(1, 1).expand(1, length)
    if scale_f.numel() == length:
        return scale_f.reshape(1, length)
    raise ValueError(f"Cannot expand scale with {scale_f.numel()} values to length {length}")


def _get_video_qkv_packed_cache(attn: torch.nn.Module) -> dict[str, torch.Tensor]:
    key = (
        attn.to_q.weight.data_ptr(),
        attn.to_k.weight.data_ptr(),
        attn.to_v.weight.data_ptr(),
        attn.to_q.bias.data_ptr(),
        attn.to_k.bias.data_ptr(),
        attn.to_v.bias.data_ptr(),
    )
    cached = getattr(attn, "_ltx_qkv_packed_linear_cache", None)
    if cached is not None and cached.get("key") == key:
        return cached
    packed = {
        "key": key,
        "weight": torch.cat((attn.to_q.weight, attn.to_k.weight, attn.to_v.weight), dim=0).contiguous(),
        "bias": torch.cat((attn.to_q.bias, attn.to_k.bias, attn.to_v.bias), dim=0).contiguous(),
    }
    setattr(attn, "_ltx_qkv_packed_linear_cache", packed)
    return packed


def _get_video_qkv_packed_requant_cache(attn: torch.nn.Module) -> dict[str, torch.Tensor]:
    scales = (attn.to_q.weight_scale, attn.to_k.weight_scale, attn.to_v.weight_scale)
    key = (
        attn.to_q.weight.data_ptr(),
        attn.to_k.weight.data_ptr(),
        attn.to_v.weight.data_ptr(),
        attn.to_q.bias.data_ptr(),
        attn.to_k.bias.data_ptr(),
        attn.to_v.bias.data_ptr(),
        tuple(float(scale.detach().float().item()) for scale in scales),
    )
    cached = getattr(attn, "_ltx_qkv_packed_requant_cache", None)
    if cached is not None and cached.get("key") == key:
        return cached

    fp8_min = torch.finfo(torch.float8_e4m3fn).min
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    common_scale = torch.stack([scale.detach().float() for scale in scales]).amax().contiguous()

    def _requant(linear: torch.nn.Module) -> torch.Tensor:
        dequant = linear.weight.float() * linear.weight_scale.detach().float()
        return torch.clamp(dequant / common_scale, fp8_min, fp8_max).to(torch.float8_e4m3fn)

    packed = {
        "key": key,
        "weight": torch.cat((_requant(attn.to_q), _requant(attn.to_k), _requant(attn.to_v)), dim=0).contiguous(),
        "weight_scale": common_scale,
        "bias": torch.cat((attn.to_q.bias, attn.to_k.bias, attn.to_v.bias), dim=0).contiguous(),
    }
    setattr(attn, "_ltx_qkv_packed_requant_cache", packed)
    return packed


def _get_video_qkv_grouped_mm_cache(attn: torch.nn.Module, rows: int) -> dict[str, torch.Tensor]:
    key = (
        attn.to_q.weight.data_ptr(),
        attn.to_k.weight.data_ptr(),
        attn.to_v.weight.data_ptr(),
        attn.to_q.bias.data_ptr(),
        attn.to_k.bias.data_ptr(),
        attn.to_v.bias.data_ptr(),
        attn.to_q.input_scale.data_ptr(),
        attn.to_q.weight_scale.data_ptr(),
        attn.to_k.weight_scale.data_ptr(),
        attn.to_v.weight_scale.data_ptr(),
        rows,
    )
    cached = getattr(attn, "_ltx_qkv_grouped_mm_cache", None)
    if cached is not None and cached.get("key") == key:
        return cached

    packed = {
        "key": key,
        "weight": torch.stack((attn.to_q.weight, attn.to_k.weight, attn.to_v.weight), dim=0).transpose(1, 2),
        "scale_a": _scale_row(attn.to_q.input_scale, rows).expand(3, rows).contiguous(),
        "scale_b": torch.cat(
            (
                _scale_row(attn.to_q.weight_scale, attn.to_q.out_features),
                _scale_row(attn.to_k.weight_scale, attn.to_k.out_features),
                _scale_row(attn.to_v.weight_scale, attn.to_v.out_features),
            ),
            dim=0,
        ).contiguous(),
        "bias": torch.stack((attn.to_q.bias, attn.to_k.bias, attn.to_v.bias), dim=0).contiguous(),
    }
    setattr(attn, "_ltx_qkv_grouped_mm_cache", packed)
    return packed


def _get_video_qk_grouped_mm_cache(attn: torch.nn.Module, rows: int) -> dict[str, torch.Tensor]:
    key = (
        attn.to_q.weight.data_ptr(),
        attn.to_k.weight.data_ptr(),
        attn.to_q.bias.data_ptr(),
        attn.to_k.bias.data_ptr(),
        attn.to_q.input_scale.data_ptr(),
        attn.to_q.weight_scale.data_ptr(),
        attn.to_k.weight_scale.data_ptr(),
        rows,
    )
    cached = getattr(attn, "_ltx_qk_grouped_mm_cache", None)
    if cached is not None and cached.get("key") == key:
        return cached

    packed = {
        "key": key,
        "weight": torch.stack((attn.to_q.weight, attn.to_k.weight), dim=0).transpose(1, 2),
        "scale_a": _scale_row(attn.to_q.input_scale, rows).expand(2, rows).contiguous(),
        "scale_b": torch.cat(
            (
                _scale_row(attn.to_q.weight_scale, attn.to_q.out_features),
                _scale_row(attn.to_k.weight_scale, attn.to_k.out_features),
            ),
            dim=0,
        ).contiguous(),
        "bias": torch.stack((attn.to_q.bias, attn.to_k.bias), dim=0).contiguous(),
    }
    setattr(attn, "_ltx_qk_grouped_mm_cache", packed)
    return packed


def _record_video_qkv_packed_requant_check(
    attn: torch.nn.Module,
    x: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    if len(_VIDEO_QKV_PACKED_REQUANT_CHECKS) >= _VIDEO_QKV_PACKED_REQUANT_CHECK_LIMIT:
        return
    with torch.no_grad():
        refs = (attn.to_q(x), attn.to_k(x), attn.to_v(x))
        outs = (q, k, v)
        for name, ref, out in zip(("q", "k", "v"), refs, outs, strict=True):
            diff = (out.float() - ref.float()).abs()
            denom = ref.float().abs().clamp_min(1e-6)
            _VIDEO_QKV_PACKED_REQUANT_CHECKS.append(
                {
                    "tensor": name,
                    "max_abs": float(diff.max().item()),
                    "mean_abs": float(diff.mean().item()),
                    "rms": float(torch.sqrt(torch.mean(diff.square())).item()),
                    "mean_rel": float((diff / denom).mean().item()),
                }
            )


def _record_video_qkv_grouped_mm_check(
    attn: torch.nn.Module,
    x: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    if len(_VIDEO_QKV_GROUPED_MM_CHECKS) >= _VIDEO_QKV_GROUPED_MM_CHECK_LIMIT:
        return
    with torch.no_grad():
        refs = (attn.to_q(x), attn.to_k(x), attn.to_v(x))
        outs = (q, k, v)
        for name, ref, out in zip(("q", "k", "v"), refs, outs, strict=True):
            diff = (out.float() - ref.float()).abs()
            denom = ref.float().abs().clamp_min(1e-6)
            _VIDEO_QKV_GROUPED_MM_CHECKS.append(
                {
                    "tensor": name,
                    "max_abs": float(diff.max().item()),
                    "mean_abs": float(diff.mean().item()),
                    "rms": float(torch.sqrt(torch.mean(diff.square())).item()),
                    "mean_rel": float((diff / denom).mean().item()),
                }
            )


def _record_video_qk_grouped_mm_check(
    attn: torch.nn.Module,
    x: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
) -> None:
    if len(_VIDEO_QK_GROUPED_MM_CHECKS) >= _VIDEO_QK_GROUPED_MM_CHECK_LIMIT:
        return
    with torch.no_grad():
        refs = (attn.to_q(x), attn.to_k(x))
        outs = (q, k)
        for name, ref, out in zip(("q", "k"), refs, outs, strict=True):
            diff = (out.float() - ref.float()).abs()
            denom = ref.float().abs().clamp_min(1e-6)
            _VIDEO_QK_GROUPED_MM_CHECKS.append(
                {
                    "tensor": name,
                    "max_abs": float(diff.max().item()),
                    "mean_abs": float(diff.mean().item()),
                    "rms": float(torch.sqrt(torch.mean(diff.square())).item()),
                    "mean_rel": float((diff / denom).mean().item()),
                }
            )


def _can_use_fused_fp8_quantize_per_head_4d(x: torch.Tensor) -> bool:
    if triton is None:
        return False
    if not x.is_cuda or not x.is_contiguous() or x.dtype not in (torch.bfloat16, torch.float16):
        return False
    if x.ndim != 4:
        return False
    batch, tokens, heads, head_dim = x.shape
    if batch < 1 or tokens < 1 or heads < 1 or head_dim not in (64, 128):
        return False
    return True


def _can_use_fused_video_qk_rmsnorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> bool:
    return _fused_video_qk_rmsnorm_rope_guard_reason(q, k, q_weight, k_weight, cos, sin) is None


def _fused_video_qk_rmsnorm_rope_guard_reason(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> str | None:
    if triton is None:
        return "triton_missing"
    if not q.is_cuda or not k.is_cuda or q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        return "qk_device_or_dtype"
    if q.shape != k.shape or q.ndim != 3 or q.shape[0] != 1 or q.shape[-1] != 4096:
        return "qk_shape"
    if q.shape[1] < 1:
        return "empty_tokens"
    if not q.is_contiguous() or not k.is_contiguous():
        return "qk_noncontiguous"
    if q_weight.shape != (4096,) or k_weight.shape != (4096,):
        return "qk_weight_shape"
    if cos.ndim != 4 or sin.ndim != 4 or cos.shape != sin.shape:
        return "rope_shape"
    if cos.shape[0] not in (1, q.shape[0]) or cos.shape[1] != 32 or cos.shape[2] != q.shape[1] or cos.shape[3] != 64:
        return "rope_shape"
    return None


def _can_use_fused_video_qkv_rmsnorm_rope_pack(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> bool:
    return _fused_video_qkv_rmsnorm_rope_pack_guard_reason(q, k, v, q_weight, k_weight, cos, sin) is None


def _fused_video_qkv_rmsnorm_rope_pack_guard_reason(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> str | None:
    reason = _fused_video_qk_rmsnorm_rope_guard_reason(q, k, q_weight, k_weight, cos, sin)
    if reason is not None:
        return reason
    if v.shape != q.shape:
        return "v_shape"
    if not v.is_cuda:
        return "v_device"
    if not v.is_contiguous():
        return "v_noncontiguous"
    if v.dtype != q.dtype:
        return "v_dtype"
    return None


def _can_use_fused_video_qk_bias_rmsnorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> bool:
    if not _can_use_fused_video_qk_rmsnorm_rope(q, k, q_weight, k_weight, cos, sin):
        return False
    if q_bias.shape != (4096,) or k_bias.shape != (4096,):
        return False
    if not q_bias.is_cuda or not k_bias.is_cuda:
        return False
    if q_bias.dtype not in (torch.bfloat16, torch.float32) or k_bias.dtype not in (torch.bfloat16, torch.float32):
        return False
    return True


def _can_use_fused_residual_gate(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor) -> bool:
    if triton is None:
        return False
    if not x.is_cuda or not y.is_cuda or x.dtype != torch.bfloat16 or y.dtype != torch.bfloat16:
        return False
    if x.shape != y.shape or x.ndim < 2 or x.shape[-1] not in (2048, 4096):
        return False
    if not x.is_contiguous() or not y.is_contiguous():
        return False
    if not gate.is_cuda or gate.dtype != torch.bfloat16:
        return False
    if gate.shape[-1] != x.shape[-1] or gate.numel() % x.shape[-1] != 0:
        return False
    return True


def _can_use_fused_head_gate_mul(out: torch.Tensor, gates: torch.Tensor, heads: int, dim_head: int) -> bool:
    if triton is None:
        return False
    if not out.is_cuda or out.dtype != torch.bfloat16 or not out.is_contiguous():
        return False
    if out.ndim != 3 or heads < 1 or dim_head < 1 or out.shape[-1] != heads * dim_head:
        return False
    if not gates.is_cuda or gates.dtype != torch.bfloat16:
        return False
    if gates.shape != (out.shape[0], out.shape[1], heads):
        return False
    return True


def _can_use_fused_adazero(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> bool:
    if not x.is_cuda or x.dtype != torch.bfloat16:
        return False
    if x.ndim < 2 or x.shape[-1] not in (2048, 4096):
        return False
    if scale.shape[-1] != x.shape[-1] or shift.shape[-1] != x.shape[-1]:
        return False
    if scale.numel() % x.shape[-1] != 0 or shift.numel() % x.shape[-1] != 0:
        return False
    if scale.numel() != shift.numel():
        return False
    return True


def _can_use_ffn_gelu_fp8_quant(project_in: torch.nn.Module, linear_out: torch.nn.Module) -> bool:
    if triton is None:
        return False
    try:
        from ltx_core.model.transformer.gelu_approx import GELUApprox
        from ltx_core.quantization.fp8_scaled_mm import FP8Linear
        from ltx_core.quantization.trtllm_scaled_usable import trtllm_scaled_mm_usable
    except Exception:
        return False
    if trtllm_scaled_mm_usable():
        return False
    return (
        isinstance(project_in, GELUApprox)
        and isinstance(getattr(project_in, "proj", None), FP8Linear)
        and isinstance(linear_out, FP8Linear)
    )


_FFN_GELU_FP8_QUANT_CALLS = 0


@torch.compiler.disable
def patch_ltx_adazero() -> None:
    from ltx_core.model.transformer.ops import PytorchAdaZeroFunction

    def _call(self: object, x: torch.Tensor, eps: float, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return fused_adazero(x, eps, scale, shift)

    PytorchAdaZeroFunction.__call__ = _call


@torch.compiler.disable
def patch_ltx_fp8_quantize(*, bias_epilogue: bool = False, triton_bias_add: bool = False) -> None:
    from ltx_core.quantization.fp8_scaled_mm import FP8Linear
    from ltx_core.quantization.trtllm_scaled_usable import trtllm_scaled_mm_usable

    def _forward(self: object, x: torch.Tensor) -> torch.Tensor:
        origin_shape = x.shape

        if trtllm_scaled_mm_usable():
            qinput, cur_input_scale = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x, self.input_scale)
            if qinput.dim() == 3:
                qinput = qinput.reshape(-1, qinput.shape[-1])
            output = torch.ops.trtllm.cublas_scaled_mm(
                qinput,
                self.weight.t(),
                scale_a=cur_input_scale,
                scale_b=self.weight_scale,
                bias=None,
                out_dtype=x.dtype,
            )
        else:
            qinput = fused_fp8_quantize_e4m3(x.contiguous(), self.input_scale)
            if qinput.dim() == 3:
                qinput = qinput.reshape(-1, qinput.shape[-1])
            output = torch._scaled_mm(
                qinput,
                self.weight.t(),
                scale_a=self.input_scale,
                scale_b=self.weight_scale,
                bias=self.bias.to(x.dtype) if (bias_epilogue and self.bias is not None) else None,
                out_dtype=x.dtype,
                use_fast_accum=True,
            )

        if self.bias is not None and not bias_epilogue:
            if triton_bias_add:
                output = fused_bias_add_bf16(output, self.bias)
            else:
                output = output + self.bias.to(output.dtype)

        if output.dim() != len(origin_shape):
            output_shape = list(origin_shape)
            output_shape[-1] = output.shape[-1]
            output = output.reshape(output_shape)

        return output

    FP8Linear.forward = _forward


@torch.compiler.disable
def patch_ltx_ffn_gelu_fp8_quant() -> None:
    global _FFN_GELU_FP8_QUANT_CALLS
    _FFN_GELU_FP8_QUANT_CALLS = 0
    from ltx_core.model.transformer.feed_forward import FeedForward

    original_forward = FeedForward.forward

    def _forward(self: object, x: torch.Tensor) -> torch.Tensor:
        global _FFN_GELU_FP8_QUANT_CALLS
        net = getattr(self, "net", None)
        if (
            net is not None
            and len(net) == 3
            and _can_use_ffn_gelu_fp8_quant(net[0], net[2])
            and x.is_cuda
            and x.dtype in (torch.bfloat16, torch.float16)
        ):
            hidden = net[0].proj(x)
            if hidden.is_contiguous():
                qhidden = fused_gelu_tanh_fp8_quantize_e4m3(hidden, net[2].input_scale)
                _FFN_GELU_FP8_QUANT_CALLS += 1
                return fp8_linear_from_quantized_e4m3(net[2], qhidden, out_dtype=hidden.dtype)
        return original_forward(self, x)

    FeedForward.forward = _forward


def collect_ffn_gelu_fp8_quant_calls() -> int:
    return _FFN_GELU_FP8_QUANT_CALLS


_CROSS_ATTENTION_ADALN_CALLS = 0


@torch.compiler.disable
def patch_ltx_cross_attention_adaln() -> None:
    global _CROSS_ATTENTION_ADALN_CALLS
    _CROSS_ATTENTION_ADALN_CALLS = 0

    import ltx_core.model.transformer.transformer as transformer_mod

    def _apply_cross_attention_adaln(
        x: torch.Tensor,
        context: torch.Tensor,
        attn: torch.nn.Module,
        q_shift: torch.Tensor,
        q_scale: torch.Tensor,
        q_gate: torch.Tensor,
        prompt_scale_shift_table: torch.Tensor,
        prompt_timestep: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        norm_eps: float = 1e-6,
    ) -> torch.Tensor:
        global _CROSS_ATTENTION_ADALN_CALLS
        batch_size = x.shape[0]
        shift_kv, scale_kv = (
            prompt_scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
            + prompt_timestep.reshape(batch_size, prompt_timestep.shape[1], 2, -1)
        ).unbind(dim=2)
        attn_input = fused_adazero(x, norm_eps, q_scale, q_shift)
        encoder_hidden_states = context * (1 + scale_kv) + shift_kv
        _CROSS_ATTENTION_ADALN_CALLS += 1
        return attn(attn_input, context=encoder_hidden_states, mask=context_mask) * q_gate

    transformer_mod.apply_cross_attention_adaln = _apply_cross_attention_adaln


def collect_cross_attention_adaln_calls() -> int:
    return _CROSS_ATTENTION_ADALN_CALLS


_VIDEO_PREATTENTION_CHECK_LIMIT = 0
_VIDEO_PREATTENTION_CHECKS: list[dict[str, float]] = []
_VIDEO_PREATTENTION_MODE = "dual"


@torch.compiler.disable
def patch_ltx_video_preattention(checks: int = 0, *, mode: str = "dual") -> None:
    global _VIDEO_PREATTENTION_CHECK_LIMIT, _VIDEO_PREATTENTION_MODE
    if mode not in {"dual", "separate"}:
        raise ValueError(f"unknown video preattention mode: {mode}")
    _VIDEO_PREATTENTION_CHECK_LIMIT = checks
    _VIDEO_PREATTENTION_MODE = mode
    _VIDEO_PREATTENTION_CHECKS.clear()

    from ltx_core.model.transformer.ops import PytorchPreAttention
    from ltx_core.model.transformer.rope import LTXRopeType

    def _call(
        self: object,
        q: torch.Tensor,
        k: torch.Tensor,
        attn_module: torch.nn.Module,
        mask: torch.Tensor | None,
        pe: tuple[torch.Tensor, torch.Tensor] | None,
        k_pe: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            mask is None
            and pe is not None
            and k_pe is None
            and getattr(attn_module, "rope_type", None) is LTXRopeType.SPLIT
            and _can_use_fused_video_qk_rmsnorm_rope(
                q,
                k,
                attn_module.q_norm.weight,
                attn_module.k_norm.weight,
                pe[0],
                pe[1],
            )
        ):
            if _VIDEO_PREATTENTION_MODE == "separate":
                out = fused_video_qk_rmsnorm_rope_separate(
                    q,
                    k,
                    attn_module.q_norm.weight,
                    attn_module.k_norm.weight,
                    attn_module.q_norm.eps,
                    pe[0],
                    pe[1],
                )
            else:
                out = fused_video_qk_rmsnorm_rope(
                    q,
                    k,
                    attn_module.q_norm.weight,
                    attn_module.k_norm.weight,
                    attn_module.q_norm.eps,
                    pe[0],
                    pe[1],
                )
            if _VIDEO_PREATTENTION_CHECK_LIMIT > len(_VIDEO_PREATTENTION_CHECKS):
                ref = _torch_video_qk_rmsnorm_rope(
                    q,
                    k,
                    attn_module.q_norm.weight,
                    attn_module.k_norm.weight,
                    attn_module.q_norm.eps,
                    pe[0],
                    pe[1],
                )
                _record_video_preattention_check(ref, out)
            return out

        q = attn_module.q_norm(q)
        k = attn_module.k_norm(k)
        if pe is not None:
            from ltx_core.model.transformer.rope import apply_rotary_emb

            q = apply_rotary_emb(q, pe, attn_module.rope_type)
            k = apply_rotary_emb(k, pe if k_pe is None else k_pe, attn_module.rope_type)
        return q, k

    PytorchPreAttention.__call__ = _call


def _record_video_preattention_check(
    ref: tuple[torch.Tensor, torch.Tensor],
    out: tuple[torch.Tensor, torch.Tensor],
) -> None:
    ref_q, ref_k = ref
    out_q, out_k = out
    diff_q = (out_q.float() - ref_q.float()).abs()
    diff_k = (out_k.float() - ref_k.float()).abs()
    diff = torch.maximum(diff_q, diff_k)
    denom = torch.maximum(ref_q.float().abs(), ref_k.float().abs()).clamp_min(1e-6)
    _VIDEO_PREATTENTION_CHECKS.append(
        {
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "rms": float(torch.sqrt(torch.mean(diff.square())).item()),
            "mean_rel": float((diff / denom).mean().item()),
        }
    )


def collect_video_preattention_checks() -> list[dict[str, float]]:
    return list(_VIDEO_PREATTENTION_CHECKS)


_BLOCK_RESIDUAL_MODE = "addcmul"
_FUSE_VIDEO_ADA_VALUES = False
_VIDEO_ADA_VALUES_CALLS = 0
_FUSE_AUDIO_ADA_VALUES = False
_AUDIO_ADA_VALUES_CALLS = 0
_FUSE_VIDEO_TEXT_ADALN = False
_VIDEO_TEXT_ADALN_CALLS = 0
_FUSE_VIDEO_TEXT_CONTEXT_ADALN = False
_FUSE_VIDEO_OUT_BIAS_RESIDUAL = False
_FUSE_VIDEO_FFN_OUT_BIAS_RESIDUAL = False
_FUSE_VIDEO_QK_BIAS_PREATTENTION = False
_FUSE_VIDEO_QKV_QUANT_REUSE = False
_VIDEO_QKV_QUANT_REUSE_CALLS = 0
_FUSE_VIDEO_QKV_PACKED_LINEAR = False
_VIDEO_QKV_PACKED_LINEAR_CALLS = 0
_VIDEO_QKV_PACKED_LINEAR_FALLBACKS = 0
_VIDEO_QKV_PACKED_LINEAR_FALLBACK_REASONS: dict[str, int] = {}
_FUSE_VIDEO_QKV_PACKED_REQUANT = False
_VIDEO_QKV_PACKED_REQUANT_CALLS = 0
_VIDEO_QKV_PACKED_REQUANT_FALLBACKS = 0
_VIDEO_QKV_PACKED_REQUANT_FALLBACK_REASONS: dict[str, int] = {}
_VIDEO_QKV_PACKED_REQUANT_CHECK_LIMIT = 0
_VIDEO_QKV_PACKED_REQUANT_CHECKS: list[dict[str, float]] = []
_FUSE_VIDEO_QKV_GROUPED_MM = False
_VIDEO_QKV_GROUPED_MM_CALLS = 0
_VIDEO_QKV_GROUPED_MM_FALLBACKS = 0
_VIDEO_QKV_GROUPED_MM_FALLBACK_REASONS: dict[str, int] = {}
_VIDEO_QKV_GROUPED_MM_CHECK_LIMIT = 0
_VIDEO_QKV_GROUPED_MM_CHECKS: list[dict[str, float]] = []
_FUSE_VIDEO_QK_GROUPED_MM = False
_VIDEO_QK_GROUPED_MM_CALLS = 0
_VIDEO_QK_GROUPED_MM_FALLBACKS = 0
_VIDEO_QK_GROUPED_MM_FALLBACK_REASONS: dict[str, int] = {}
_VIDEO_QK_GROUPED_MM_CHECK_LIMIT = 0
_VIDEO_QK_GROUPED_MM_CHECKS: list[dict[str, float]] = []
_FUSE_VIDEO_MSA_BRANCH = False
_FUSE_VIDEO_GATE_MUL = False
_PROFILE_VIDEO_MSA_BRANCH = False
_VIDEO_MSA_BRANCH_TOKEN_COUNTS: set[int] | None = None
_VIDEO_MSA_BRANCH_MODE = "generic"
_VIDEO_MSA_BRANCH_CALLS = 0
_VIDEO_MSA_BRANCH_FALLBACKS = 0
_VIDEO_MSA_BRANCH_FALLBACK_REASONS: dict[str, int] = {}
_VIDEO_MSA_BRANCH_PROFILE_EVENTS: list[tuple[str, object, object]] = []
_VIDEO_MSA_BRANCH_LAST_FALLBACK_REASON: str | None = None


def _residual_addcmul(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    if _BLOCK_RESIDUAL_MODE == "triton_bf16_round":
        return fused_residual_gate_bf16(x, y, gate)
    return torch.addcmul(x, y, gate)


def _simple_residual_gate(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    if _BLOCK_RESIDUAL_MODE in {"triton_bf16_round", "triton_simple_bf16_round"}:
        return fused_residual_gate_bf16(x, y, gate)
    if _BLOCK_RESIDUAL_MODE in {"addcmul", "addcmul_simple"}:
        return torch.addcmul(x, y, gate)
    return x + y * gate


@torch.compiler.disable
def patch_ltx_block_residual_addcmul() -> None:
    global _BLOCK_RESIDUAL_MODE
    _BLOCK_RESIDUAL_MODE = "addcmul"
    _patch_ltx_block_residuals()


@torch.compiler.disable
def patch_ltx_block_residual_triton() -> None:
    global _BLOCK_RESIDUAL_MODE
    _BLOCK_RESIDUAL_MODE = "triton_bf16_round"
    _patch_ltx_block_residuals()


@torch.compiler.disable
def patch_ltx_block_simple_residual_triton() -> None:
    global _BLOCK_RESIDUAL_MODE
    _BLOCK_RESIDUAL_MODE = "triton_simple_bf16_round"
    _patch_ltx_block_residuals()


@torch.compiler.disable
def patch_ltx_video_ada_values() -> None:
    global _FUSE_VIDEO_ADA_VALUES, _VIDEO_ADA_VALUES_CALLS
    _FUSE_VIDEO_ADA_VALUES = True
    _VIDEO_ADA_VALUES_CALLS = 0
    _patch_ltx_block_residuals()


def collect_video_ada_values_calls() -> int:
    return _VIDEO_ADA_VALUES_CALLS


@torch.compiler.disable
def patch_ltx_audio_ada_values() -> None:
    global _FUSE_AUDIO_ADA_VALUES, _AUDIO_ADA_VALUES_CALLS
    _FUSE_AUDIO_ADA_VALUES = True
    _AUDIO_ADA_VALUES_CALLS = 0
    _patch_ltx_block_residuals()


def collect_audio_ada_values_calls() -> int:
    return _AUDIO_ADA_VALUES_CALLS


def _increment_ada_counter(counter: str) -> None:
    global _VIDEO_ADA_VALUES_CALLS, _AUDIO_ADA_VALUES_CALLS, _VIDEO_TEXT_ADALN_CALLS
    if counter == "video":
        _VIDEO_ADA_VALUES_CALLS += 1
    elif counter == "audio":
        _AUDIO_ADA_VALUES_CALLS += 1
    elif counter == "video_text":
        _VIDEO_TEXT_ADALN_CALLS += 1


@torch.compiler.disable
def patch_ltx_video_text_adaln() -> None:
    global _FUSE_VIDEO_TEXT_ADALN, _VIDEO_TEXT_ADALN_CALLS
    _FUSE_VIDEO_TEXT_ADALN = True
    _VIDEO_TEXT_ADALN_CALLS = 0
    _patch_ltx_block_residuals()


@torch.compiler.disable
def patch_ltx_video_text_context_adaln() -> None:
    global _FUSE_VIDEO_TEXT_CONTEXT_ADALN, _VIDEO_TEXT_ADALN_CALLS
    _FUSE_VIDEO_TEXT_CONTEXT_ADALN = True
    _VIDEO_TEXT_ADALN_CALLS = 0
    _patch_ltx_block_residuals()


def collect_video_text_adaln_calls() -> int:
    return _VIDEO_TEXT_ADALN_CALLS


@torch.compiler.disable
def patch_ltx_video_out_bias_residual() -> None:
    global _FUSE_VIDEO_OUT_BIAS_RESIDUAL
    _FUSE_VIDEO_OUT_BIAS_RESIDUAL = True
    _patch_ltx_block_residuals()


@torch.compiler.disable
def patch_ltx_video_ffn_out_bias_residual() -> None:
    global _FUSE_VIDEO_FFN_OUT_BIAS_RESIDUAL
    _FUSE_VIDEO_FFN_OUT_BIAS_RESIDUAL = True
    _patch_ltx_block_residuals()


@torch.compiler.disable
def patch_ltx_video_qk_bias_preattention() -> None:
    global _FUSE_VIDEO_QK_BIAS_PREATTENTION
    _FUSE_VIDEO_QK_BIAS_PREATTENTION = True
    _patch_ltx_block_residuals()


@torch.compiler.disable
def patch_ltx_video_qkv_quant_reuse() -> None:
    global _FUSE_VIDEO_QKV_QUANT_REUSE, _VIDEO_QKV_QUANT_REUSE_CALLS
    _FUSE_VIDEO_QKV_QUANT_REUSE = True
    _VIDEO_QKV_QUANT_REUSE_CALLS = 0
    _patch_ltx_block_residuals()


def collect_video_qkv_quant_reuse_calls() -> int:
    return _VIDEO_QKV_QUANT_REUSE_CALLS


@torch.compiler.disable
def patch_ltx_video_qkv_packed_linear() -> None:
    global _FUSE_VIDEO_QKV_PACKED_LINEAR, _VIDEO_QKV_PACKED_LINEAR_CALLS
    global _VIDEO_QKV_PACKED_LINEAR_FALLBACKS, _VIDEO_QKV_PACKED_LINEAR_FALLBACK_REASONS
    _FUSE_VIDEO_QKV_PACKED_LINEAR = True
    _VIDEO_QKV_PACKED_LINEAR_CALLS = 0
    _VIDEO_QKV_PACKED_LINEAR_FALLBACKS = 0
    _VIDEO_QKV_PACKED_LINEAR_FALLBACK_REASONS = {}
    _patch_ltx_block_residuals()


def collect_video_qkv_packed_linear_stats() -> dict[str, object]:
    return {
        "calls": _VIDEO_QKV_PACKED_LINEAR_CALLS,
        "fallbacks": _VIDEO_QKV_PACKED_LINEAR_FALLBACKS,
        "fallback_reasons": dict(_VIDEO_QKV_PACKED_LINEAR_FALLBACK_REASONS),
    }


@torch.compiler.disable
def patch_ltx_video_qkv_packed_requant(checks: int = 0) -> None:
    global _FUSE_VIDEO_QKV_PACKED_REQUANT, _VIDEO_QKV_PACKED_REQUANT_CALLS
    global _VIDEO_QKV_PACKED_REQUANT_FALLBACKS, _VIDEO_QKV_PACKED_REQUANT_FALLBACK_REASONS
    global _VIDEO_QKV_PACKED_REQUANT_CHECK_LIMIT, _VIDEO_QKV_PACKED_REQUANT_CHECKS
    _FUSE_VIDEO_QKV_PACKED_REQUANT = True
    _VIDEO_QKV_PACKED_REQUANT_CALLS = 0
    _VIDEO_QKV_PACKED_REQUANT_FALLBACKS = 0
    _VIDEO_QKV_PACKED_REQUANT_FALLBACK_REASONS = {}
    _VIDEO_QKV_PACKED_REQUANT_CHECK_LIMIT = max(0, checks) * 3
    _VIDEO_QKV_PACKED_REQUANT_CHECKS = []
    _patch_ltx_block_residuals()


def collect_video_qkv_packed_requant_stats() -> dict[str, object]:
    return {
        "calls": _VIDEO_QKV_PACKED_REQUANT_CALLS,
        "fallbacks": _VIDEO_QKV_PACKED_REQUANT_FALLBACKS,
        "fallback_reasons": dict(_VIDEO_QKV_PACKED_REQUANT_FALLBACK_REASONS),
        "checks": list(_VIDEO_QKV_PACKED_REQUANT_CHECKS),
    }


@torch.compiler.disable
def patch_ltx_video_qkv_grouped_mm(checks: int = 0) -> None:
    global _FUSE_VIDEO_QKV_GROUPED_MM, _VIDEO_QKV_GROUPED_MM_CALLS
    global _VIDEO_QKV_GROUPED_MM_FALLBACKS, _VIDEO_QKV_GROUPED_MM_FALLBACK_REASONS
    global _VIDEO_QKV_GROUPED_MM_CHECK_LIMIT, _VIDEO_QKV_GROUPED_MM_CHECKS
    _FUSE_VIDEO_QKV_GROUPED_MM = True
    _VIDEO_QKV_GROUPED_MM_CALLS = 0
    _VIDEO_QKV_GROUPED_MM_FALLBACKS = 0
    _VIDEO_QKV_GROUPED_MM_FALLBACK_REASONS = {}
    _VIDEO_QKV_GROUPED_MM_CHECK_LIMIT = max(0, checks) * 3
    _VIDEO_QKV_GROUPED_MM_CHECKS = []
    _patch_ltx_block_residuals()


def collect_video_qkv_grouped_mm_stats() -> dict[str, object]:
    return {
        "calls": _VIDEO_QKV_GROUPED_MM_CALLS,
        "fallbacks": _VIDEO_QKV_GROUPED_MM_FALLBACKS,
        "fallback_reasons": dict(_VIDEO_QKV_GROUPED_MM_FALLBACK_REASONS),
        "checks": list(_VIDEO_QKV_GROUPED_MM_CHECKS),
    }


@torch.compiler.disable
def patch_ltx_video_qk_grouped_mm(checks: int = 0) -> None:
    global _FUSE_VIDEO_QK_GROUPED_MM, _VIDEO_QK_GROUPED_MM_CALLS
    global _VIDEO_QK_GROUPED_MM_FALLBACKS, _VIDEO_QK_GROUPED_MM_FALLBACK_REASONS
    global _VIDEO_QK_GROUPED_MM_CHECK_LIMIT, _VIDEO_QK_GROUPED_MM_CHECKS
    _FUSE_VIDEO_QK_GROUPED_MM = True
    _VIDEO_QK_GROUPED_MM_CALLS = 0
    _VIDEO_QK_GROUPED_MM_FALLBACKS = 0
    _VIDEO_QK_GROUPED_MM_FALLBACK_REASONS = {}
    _VIDEO_QK_GROUPED_MM_CHECK_LIMIT = max(0, checks) * 2
    _VIDEO_QK_GROUPED_MM_CHECKS = []
    _patch_ltx_block_residuals()


def collect_video_qk_grouped_mm_stats() -> dict[str, object]:
    return {
        "calls": _VIDEO_QK_GROUPED_MM_CALLS,
        "fallbacks": _VIDEO_QK_GROUPED_MM_FALLBACKS,
        "fallback_reasons": dict(_VIDEO_QK_GROUPED_MM_FALLBACK_REASONS),
        "checks": list(_VIDEO_QK_GROUPED_MM_CHECKS),
    }


def _record_video_qk_grouped_mm_fallback(reason: str) -> None:
    global _VIDEO_QK_GROUPED_MM_FALLBACKS
    if not _FUSE_VIDEO_QK_GROUPED_MM:
        return
    _VIDEO_QK_GROUPED_MM_FALLBACKS += 1
    _VIDEO_QK_GROUPED_MM_FALLBACK_REASONS[reason] = (
        _VIDEO_QK_GROUPED_MM_FALLBACK_REASONS.get(reason, 0) + 1
    )


def _record_video_qkv_grouped_mm_fallback(reason: str) -> None:
    global _VIDEO_QKV_GROUPED_MM_FALLBACKS
    if not _FUSE_VIDEO_QKV_GROUPED_MM:
        return
    _VIDEO_QKV_GROUPED_MM_FALLBACKS += 1
    _VIDEO_QKV_GROUPED_MM_FALLBACK_REASONS[reason] = (
        _VIDEO_QKV_GROUPED_MM_FALLBACK_REASONS.get(reason, 0) + 1
    )


def _record_video_qkv_packed_requant_fallback(reason: str) -> None:
    global _VIDEO_QKV_PACKED_REQUANT_FALLBACKS
    if not _FUSE_VIDEO_QKV_PACKED_REQUANT:
        return
    _VIDEO_QKV_PACKED_REQUANT_FALLBACKS += 1
    _VIDEO_QKV_PACKED_REQUANT_FALLBACK_REASONS[reason] = (
        _VIDEO_QKV_PACKED_REQUANT_FALLBACK_REASONS.get(reason, 0) + 1
    )


def _record_video_qkv_packed_linear_fallback(reason: str) -> None:
    global _VIDEO_QKV_PACKED_LINEAR_FALLBACKS
    if not _FUSE_VIDEO_QKV_PACKED_LINEAR:
        return
    _VIDEO_QKV_PACKED_LINEAR_FALLBACKS += 1
    _VIDEO_QKV_PACKED_LINEAR_FALLBACK_REASONS[reason] = (
        _VIDEO_QKV_PACKED_LINEAR_FALLBACK_REASONS.get(reason, 0) + 1
    )


@torch.compiler.disable
def patch_ltx_video_msa_branch(
    token_counts: tuple[int, ...] | None = None,
    *,
    profile: bool = False,
    mode: str = "generic",
    gate_mul: bool = False,
) -> None:
    """Use a single exact fast branch for video self-attention.

    This does not change the math or enable a quality-risk attention kernel. It
    packages the current production MSA fast path as one shape-gated branch so
    the serving runtime can fail closed and benchmark the whole branch before a
    lower-level custom MSA kernel replaces it.
    """

    if mode not in {"generic", "direct", "direct_bf16_out", "direct_qkvpacked"}:
        raise ValueError(f"unknown video MSA branch mode: {mode}")

    global _FUSE_VIDEO_MSA_BRANCH, _FUSE_VIDEO_GATE_MUL, _PROFILE_VIDEO_MSA_BRANCH
    global _VIDEO_MSA_BRANCH_TOKEN_COUNTS, _VIDEO_MSA_BRANCH_MODE
    global _VIDEO_MSA_BRANCH_CALLS
    global _VIDEO_MSA_BRANCH_FALLBACKS, _VIDEO_MSA_BRANCH_FALLBACK_REASONS
    global _VIDEO_MSA_BRANCH_PROFILE_EVENTS, _VIDEO_MSA_BRANCH_LAST_FALLBACK_REASON
    _FUSE_VIDEO_MSA_BRANCH = True
    _FUSE_VIDEO_GATE_MUL = gate_mul
    _PROFILE_VIDEO_MSA_BRANCH = profile
    _VIDEO_MSA_BRANCH_TOKEN_COUNTS = set(token_counts) if token_counts else None
    _VIDEO_MSA_BRANCH_MODE = mode
    _VIDEO_MSA_BRANCH_CALLS = 0
    _VIDEO_MSA_BRANCH_FALLBACKS = 0
    _VIDEO_MSA_BRANCH_FALLBACK_REASONS = {}
    _VIDEO_MSA_BRANCH_PROFILE_EVENTS = []
    _VIDEO_MSA_BRANCH_LAST_FALLBACK_REASON = None
    _patch_ltx_block_residuals()


def collect_video_msa_branch_stats() -> dict[str, object]:
    profile_rows: list[dict[str, object]] = []
    if _VIDEO_MSA_BRANCH_PROFILE_EVENTS:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        grouped: dict[str, dict[str, object]] = {}
        for name, start, end in _VIDEO_MSA_BRANCH_PROFILE_EVENTS:
            elapsed_ms = float(start.elapsed_time(end))
            row = grouped.setdefault(
                name,
                {"phase": name, "calls": 0, "total_ms": 0.0, "min_ms": None, "max_ms": 0.0},
            )
            row["calls"] = int(row["calls"]) + 1
            row["total_ms"] = float(row["total_ms"]) + elapsed_ms
            row["min_ms"] = elapsed_ms if row["min_ms"] is None else min(float(row["min_ms"]), elapsed_ms)
            row["max_ms"] = max(float(row["max_ms"]), elapsed_ms)
        profile_rows = list(grouped.values())
        for row in profile_rows:
            row["avg_ms"] = float(row["total_ms"]) / int(row["calls"])
        profile_rows.sort(key=lambda item: float(item["total_ms"]), reverse=True)
    return {
        "calls": _VIDEO_MSA_BRANCH_CALLS,
        "fallbacks": _VIDEO_MSA_BRANCH_FALLBACKS,
        "fallback_reasons": dict(_VIDEO_MSA_BRANCH_FALLBACK_REASONS),
        "token_counts": sorted(_VIDEO_MSA_BRANCH_TOKEN_COUNTS) if _VIDEO_MSA_BRANCH_TOKEN_COUNTS else None,
        "mode": _VIDEO_MSA_BRANCH_MODE,
        "profile": profile_rows,
    }


def reset_video_msa_branch_profile_events() -> None:
    _VIDEO_MSA_BRANCH_PROFILE_EVENTS.clear()


def _profile_video_msa_phase(name: str, fn: object) -> object:
    if not _PROFILE_VIDEO_MSA_BRANCH or not torch.cuda.is_available():
        return fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    _VIDEO_MSA_BRANCH_PROFILE_EVENTS.append((name, start, end))
    return result


def _video_msa_gated_attention(
    attn: torch.nn.Module,
    x: torch.Tensor,
    out: torch.Tensor,
    run_phase: object,
) -> torch.Tensor:
    if not _PROFILE_VIDEO_MSA_BRANCH and not _FUSE_VIDEO_GATE_MUL:
        return attn.gated_attention_function(x, out, attn)

    gate_logits = run_phase("gate_logits", lambda: attn.to_gate_logits(x))
    if _FUSE_VIDEO_GATE_MUL:
        gates = run_phase("gate_sigmoid", lambda: 2.0 * torch.sigmoid(gate_logits))
        return run_phase("gate_mul_triton", lambda: fused_head_gate_mul_bf16(out, gates, attn.heads, attn.dim_head))

    def _apply_gate() -> torch.Tensor:
        batch, tokens, _ = out.shape
        shaped = out.view(batch, tokens, attn.heads, attn.dim_head)
        gates = 2.0 * torch.sigmoid(gate_logits)
        return (shaped * gates.unsqueeze(-1)).view(batch, tokens, attn.heads * attn.dim_head)

    return run_phase("gate_sigmoid_mul", _apply_gate)


def _record_video_msa_branch_fallback(reason: str) -> None:
    global _VIDEO_MSA_BRANCH_FALLBACKS
    _VIDEO_MSA_BRANCH_FALLBACKS += 1
    _VIDEO_MSA_BRANCH_FALLBACK_REASONS[reason] = _VIDEO_MSA_BRANCH_FALLBACK_REASONS.get(reason, 0) + 1


_UNIFORM_TIMESTEP_ADALN_CALLS = 0
_ORIGINAL_PREPARE_TIMESTEP = None
_ORIGINAL_PREPARE_CROSS_ATTENTION_TIMESTEP = None


@torch.compiler.disable
def patch_ltx_uniform_timestep_adaln() -> None:
    """Compute AdaLN timestep embeddings once when pure T2V uses a full-denoise mask.

    The serving path asserts the denoise mask is all ones before enabling this
    patch. In that regime every token in a modality has the same timestep, so
    per-token AdaLN MLP work can be replaced by one MLP call per batch row and
    an expanded view.
    """

    global _ORIGINAL_PREPARE_TIMESTEP, _ORIGINAL_PREPARE_CROSS_ATTENTION_TIMESTEP, _UNIFORM_TIMESTEP_ADALN_CALLS

    from ltx_core.model.transformer.transformer_args import (
        MultiModalTransformerArgsPreprocessor,
        TransformerArgsPreprocessor,
    )

    if _ORIGINAL_PREPARE_TIMESTEP is None:
        _ORIGINAL_PREPARE_TIMESTEP = TransformerArgsPreprocessor._prepare_timestep
    if _ORIGINAL_PREPARE_CROSS_ATTENTION_TIMESTEP is None:
        _ORIGINAL_PREPARE_CROSS_ATTENTION_TIMESTEP = (
            MultiModalTransformerArgsPreprocessor._prepare_cross_attention_timestep
        )

    _UNIFORM_TIMESTEP_ADALN_CALLS = 0

    def _prepare_timestep_uniform(
        self: object,
        timestep: torch.Tensor,
        adaln: torch.nn.Module,
        batch_size: int,
        hidden_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global _UNIFORM_TIMESTEP_ADALN_CALLS
        if timestep.ndim == 2 and timestep.shape[1] > 1:
            tokens = timestep.shape[1]
            timestep_scaled = timestep[:, :1] * self.timestep_scale_multiplier
            prepared, embedded = adaln(timestep_scaled.flatten(), hidden_dtype=hidden_dtype)
            _UNIFORM_TIMESTEP_ADALN_CALLS += 1
            return (
                prepared.view(batch_size, 1, prepared.shape[-1]).expand(batch_size, tokens, -1),
                embedded.view(batch_size, 1, embedded.shape[-1]).expand(batch_size, tokens, -1),
            )
        return _ORIGINAL_PREPARE_TIMESTEP(self, timestep, adaln, batch_size, hidden_dtype)

    def _prepare_cross_attention_timestep_uniform(
        self: object,
        modality_timesteps: torch.Tensor,
        cross_modality_sigma: torch.Tensor,
        timestep_scale_multiplier: int,
        batch_size: int,
        hidden_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global _UNIFORM_TIMESTEP_ADALN_CALLS
        if modality_timesteps.ndim == 2 and modality_timesteps.shape[1] > 1:
            tokens = modality_timesteps.shape[1]
            av_ca_factor = self.av_ca_timestep_scale_multiplier / timestep_scale_multiplier
            scale_shift_timestep, _ = self.cross_scale_shift_adaln(
                (modality_timesteps[:, :1] * timestep_scale_multiplier).flatten(),
                hidden_dtype=hidden_dtype,
            )
            scale_shift_timestep = scale_shift_timestep.view(batch_size, 1, scale_shift_timestep.shape[-1]).expand(
                batch_size,
                tokens,
                -1,
            )
            gate_noise_timestep, _ = self.cross_gate_adaln(
                (cross_modality_sigma * timestep_scale_multiplier * av_ca_factor).flatten(),
                hidden_dtype=hidden_dtype,
            )
            gate_noise_timestep = gate_noise_timestep.view(batch_size, -1, gate_noise_timestep.shape[-1])
            _UNIFORM_TIMESTEP_ADALN_CALLS += 1
            return scale_shift_timestep, gate_noise_timestep
        return _ORIGINAL_PREPARE_CROSS_ATTENTION_TIMESTEP(
            self,
            modality_timesteps,
            cross_modality_sigma,
            timestep_scale_multiplier,
            batch_size,
            hidden_dtype,
        )

    TransformerArgsPreprocessor._prepare_timestep = _prepare_timestep_uniform
    MultiModalTransformerArgsPreprocessor._prepare_cross_attention_timestep = _prepare_cross_attention_timestep_uniform


def collect_uniform_timestep_adaln_calls() -> int:
    return _UNIFORM_TIMESTEP_ADALN_CALLS


_ROPE_EMBEDDING_CACHE_MAX_ENTRIES = 12
_ROPE_EMBEDDING_CACHE: "OrderedDict[tuple[object, ...], tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()
_ROPE_EMBEDDING_CACHE_HITS = 0
_ROPE_EMBEDDING_CACHE_MISSES = 0
_ORIGINAL_PREPARE_POSITIONAL_EMBEDDINGS = None


@torch.compiler.disable
def patch_ltx_rope_embedding_cache(max_entries: int = _ROPE_EMBEDDING_CACHE_MAX_ENTRIES) -> None:
    """Reuse deterministic RoPE embeddings for fixed-shape denoise loops.

    Pure T2V requests keep the same position tensors for every denoise step.
    The official preprocessor recomputes the same cos/sin tables each forward;
    this cache returns the exact tensors produced by the first call for that
    position tensor and RoPE configuration. The key is intentionally tied to the
    position tensor pointer/shape/stride, so it only reuses within the same
    concrete latent state instead of guessing based on arbitrary conditioning.
    """

    global _ORIGINAL_PREPARE_POSITIONAL_EMBEDDINGS
    global _ROPE_EMBEDDING_CACHE_MAX_ENTRIES, _ROPE_EMBEDDING_CACHE_HITS, _ROPE_EMBEDDING_CACHE_MISSES

    from ltx_core.model.transformer.transformer_args import TransformerArgsPreprocessor

    if _ORIGINAL_PREPARE_POSITIONAL_EMBEDDINGS is None:
        _ORIGINAL_PREPARE_POSITIONAL_EMBEDDINGS = TransformerArgsPreprocessor._prepare_positional_embeddings

    _ROPE_EMBEDDING_CACHE.clear()
    _ROPE_EMBEDDING_CACHE_MAX_ENTRIES = max_entries
    _ROPE_EMBEDDING_CACHE_HITS = 0
    _ROPE_EMBEDDING_CACHE_MISSES = 0

    def _prepare_positional_embeddings_cached(
        self: object,
        positions: torch.Tensor,
        inner_dim: int,
        max_pos: list[int],
        use_middle_indices_grid: bool,
        num_attention_heads: int,
        x_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global _ROPE_EMBEDDING_CACHE_HITS, _ROPE_EMBEDDING_CACHE_MISSES

        if not isinstance(positions, torch.Tensor) or positions.requires_grad:
            return _ORIGINAL_PREPARE_POSITIONAL_EMBEDDINGS(
                self,
                positions,
                inner_dim,
                max_pos,
                use_middle_indices_grid,
                num_attention_heads,
                x_dtype,
            )

        key = (
            positions.data_ptr(),
            tuple(positions.shape),
            tuple(positions.stride()),
            str(positions.device),
            str(positions.dtype),
            inner_dim,
            tuple(max_pos),
            bool(use_middle_indices_grid),
            num_attention_heads,
            str(x_dtype),
            bool(getattr(self, "double_precision_rope", False)),
            float(getattr(self, "positional_embedding_theta", 0.0)),
            getattr(getattr(self, "rope_type", None), "value", str(getattr(self, "rope_type", None))),
        )
        cached = _ROPE_EMBEDDING_CACHE.get(key)
        if cached is not None:
            _ROPE_EMBEDDING_CACHE.move_to_end(key)
            _ROPE_EMBEDDING_CACHE_HITS += 1
            return cached

        out = _ORIGINAL_PREPARE_POSITIONAL_EMBEDDINGS(
            self,
            positions,
            inner_dim,
            max_pos,
            use_middle_indices_grid,
            num_attention_heads,
            x_dtype,
        )
        _ROPE_EMBEDDING_CACHE_MISSES += 1
        if not any(t.requires_grad for t in out):
            _ROPE_EMBEDDING_CACHE[key] = out
            while len(_ROPE_EMBEDDING_CACHE) > _ROPE_EMBEDDING_CACHE_MAX_ENTRIES:
                _ROPE_EMBEDDING_CACHE.popitem(last=False)
        return out

    TransformerArgsPreprocessor._prepare_positional_embeddings = _prepare_positional_embeddings_cached


def collect_rope_embedding_cache_stats() -> dict[str, int]:
    return {
        "hits": _ROPE_EMBEDDING_CACHE_HITS,
        "misses": _ROPE_EMBEDDING_CACHE_MISSES,
        "entries": len(_ROPE_EMBEDDING_CACHE),
        "max_entries": _ROPE_EMBEDDING_CACHE_MAX_ENTRIES,
    }


def _video_self_attention_without_qk_out_bias(
    attn: torch.nn.Module,
    x: torch.Tensor,
    *,
    pe: tuple[torch.Tensor, torch.Tensor] | None = None,
    mask: torch.Tensor | None = None,
    perturbation_mask: torch.Tensor | None = None,
    all_perturbed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if (
        not _FUSE_VIDEO_QK_BIAS_PREATTENTION
        or pe is None
        or mask is not None
        or all_perturbed
        or not _FUSE_VIDEO_OUT_BIAS_RESIDUAL
    ):
        return None

    out_linear = attn.to_out[0]
    if not _can_use_fp8_linear_without_bias(attn.to_q, x) or not _can_use_fp8_linear_without_bias(attn.to_k, x):
        return None
    if not _can_use_fp8_linear_without_bias(out_linear, x):
        return None

    v = attn.to_v(x)
    q_no_bias = fp8_linear_without_bias(attn.to_q, x, out_dtype=x.dtype)
    k_no_bias = fp8_linear_without_bias(attn.to_k, x, out_dtype=x.dtype)
    if q_no_bias is None or k_no_bias is None:
        return None
    q, k = fused_video_qk_bias_rmsnorm_rope(
        q_no_bias,
        k_no_bias,
        attn.to_q.bias,
        attn.to_k.bias,
        attn.q_norm.weight,
        attn.k_norm.weight,
        attn.q_norm.eps,
        pe[0],
        pe[1],
    )
    out = attn.attention_function(q, k, v, attn.heads)
    if perturbation_mask is not None:
        out = out * perturbation_mask + v * (1 - perturbation_mask)
    if attn.to_gate_logits is not None:
        out = _video_msa_gated_attention(attn, x, out, _run_phase)

    projected = fp8_linear_without_bias(out_linear, out, out_dtype=out.dtype)
    if projected is None:
        return None
    return projected, out_linear.bias


def _attention_without_out_bias(
    attn: torch.nn.Module,
    x: torch.Tensor,
    *,
    context: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    pe: torch.Tensor | None = None,
    k_pe: torch.Tensor | None = None,
    perturbation_mask: torch.Tensor | None = None,
    all_perturbed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    context = x if context is None else context
    out_linear = attn.to_out[0]
    if pe is None:
        return None
    if not _can_use_fp8_linear_without_bias(out_linear, x):
        return None

    use_attention = not all_perturbed
    profile_msa = (
        _PROFILE_VIDEO_MSA_BRANCH
        and context is x
        and mask is None
        and pe is not None
        and x.ndim == 3
        and x.shape[0] == 1
        and x.shape[1] >= 1024
        and x.shape[-1] == 4096
    )

    def _run_phase(name: str, fn: object) -> object:
        return _profile_video_msa_phase(name, fn) if profile_msa else fn()

    qkv = None
    qk = None
    if _FUSE_VIDEO_QKV_GROUPED_MM and use_attention and context is x:
        qkv = _run_phase("qkv_grouped_mm", lambda: fp8_qkv_from_grouped_mm(attn, x))
        if qkv is not None:
            global _VIDEO_QKV_GROUPED_MM_CALLS
            _VIDEO_QKV_GROUPED_MM_CALLS += 1
    elif _FUSE_VIDEO_QKV_PACKED_LINEAR and use_attention and context is x:
        qkv = _run_phase("qkv_packed_linear", lambda: fp8_qkv_from_packed_linear(attn, x))
        if qkv is not None:
            global _VIDEO_QKV_PACKED_LINEAR_CALLS
            _VIDEO_QKV_PACKED_LINEAR_CALLS += 1
    elif _FUSE_VIDEO_QKV_PACKED_REQUANT and use_attention and context is x:
        qkv = _run_phase("qkv_packed_requant", lambda: fp8_qkv_from_packed_requant(attn, x))
        if qkv is not None:
            global _VIDEO_QKV_PACKED_REQUANT_CALLS
            _VIDEO_QKV_PACKED_REQUANT_CALLS += 1
    elif _FUSE_VIDEO_QKV_QUANT_REUSE and use_attention and context is x:
        qkv = _run_phase("qkv_quant_reuse", lambda: fp8_qkv_from_shared_quantized_input(attn, x))
        if qkv is not None:
            global _VIDEO_QKV_QUANT_REUSE_CALLS
            _VIDEO_QKV_QUANT_REUSE_CALLS += 1
    if qkv is None and _FUSE_VIDEO_QK_GROUPED_MM and use_attention and context is x:
        qk = _run_phase("qk_grouped_mm", lambda: fp8_qk_from_grouped_mm(attn, x))
        if qk is not None:
            global _VIDEO_QK_GROUPED_MM_CALLS
            _VIDEO_QK_GROUPED_MM_CALLS += 1
    if qkv is not None:
        q, k, v = qkv
    else:
        v = _run_phase("v_projection", lambda: attn.to_v(context))
    if not use_attention:
        out = v
    else:
        if qkv is None:
            if qk is None:
                q = _run_phase("q_projection", lambda: attn.to_q(x))
                k = _run_phase("k_projection", lambda: attn.to_k(context))
            else:
                q, k = qk
        q, k = _run_phase("preattention", lambda: attn.preattention_function(q, k, attn, mask, pe, k_pe))
        if mask is None:
            out = _run_phase("fa3_attention", lambda: attn.attention_function(q, k, v, attn.heads))
        else:
            out = attn.masked_attention_function(q, k, v, attn.heads, mask)
        if perturbation_mask is not None:
            out = out * perturbation_mask + v * (1 - perturbation_mask)

    if attn.to_gate_logits is not None:
        out = _video_msa_gated_attention(attn, x, out, _run_phase)

    projected = _run_phase("out_projection_without_bias", lambda: fp8_linear_without_bias(out_linear, out, out_dtype=out.dtype))
    if projected is None:
        return None
    return projected, out_linear.bias


def _video_msa_direct_without_out_bias(
    attn: torch.nn.Module,
    x: torch.Tensor,
    *,
    pe: tuple[torch.Tensor, torch.Tensor] | None,
    perturbation_mask: torch.Tensor | None,
    allow_bf16_out_linear: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    out_linear = attn.to_out[0]
    if pe is None:
        return None
    if not (
        _can_use_fp8_linear_without_bias(out_linear, x)
        or (allow_bf16_out_linear and _can_use_bf16_linear_without_bias(out_linear, x))
    ):
        return None
    if not _can_use_fused_video_qk_rmsnorm_rope(
        x,
        x,
        attn.q_norm.weight,
        attn.k_norm.weight,
        pe[0] if pe is not None else None,
        pe[1] if pe is not None else None,
    ):
        return None

    def _run_phase(name: str, fn: object) -> object:
        return _profile_video_msa_phase(name, fn)

    v = _run_phase("v_projection", lambda: attn.to_v(x))
    q = _run_phase("q_projection", lambda: attn.to_q(x))
    k = _run_phase("k_projection", lambda: attn.to_k(x))
    q, k = _run_phase(
        "preattention",
        lambda: fused_video_qk_rmsnorm_rope(
            q,
            k,
            attn.q_norm.weight,
            attn.k_norm.weight,
            attn.q_norm.eps,
            pe[0],
            pe[1],
        ),
    )
    out = _run_phase("fa3_attention", lambda: attn.attention_function(q, k, v, attn.heads))
    if perturbation_mask is not None:
        out = out * perturbation_mask + v * (1 - perturbation_mask)
    if attn.to_gate_logits is not None:
        out = _video_msa_gated_attention(attn, x, out, _run_phase)

    projected = _run_phase(
        "out_projection_without_bias",
        lambda: _linear_without_bias_for_video_msa(
            out_linear,
            out,
            out_dtype=out.dtype,
            allow_bf16_linear=allow_bf16_out_linear,
        ),
    )
    if projected is None:
        return None
    return projected, out_linear.bias


def _video_msa_direct_qkvpacked_without_out_bias(
    attn: torch.nn.Module,
    x: torch.Tensor,
    *,
    pe: tuple[torch.Tensor, torch.Tensor] | None,
    perturbation_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    global _VIDEO_MSA_BRANCH_LAST_FALLBACK_REASON

    _VIDEO_MSA_BRANCH_LAST_FALLBACK_REASON = None
    out_linear = attn.to_out[0]
    if pe is None:
        _VIDEO_MSA_BRANCH_LAST_FALLBACK_REASON = "qkvpacked_missing_rope"
        return None
    if perturbation_mask is not None:
        _VIDEO_MSA_BRANCH_LAST_FALLBACK_REASON = "qkvpacked_perturbation_mask"
        return None
    if not _can_use_fp8_linear_without_bias(out_linear, x):
        _VIDEO_MSA_BRANCH_LAST_FALLBACK_REASON = "qkvpacked_out_linear"
        return None
    input_guard_reason = _fused_video_qk_rmsnorm_rope_guard_reason(
        x,
        x,
        attn.q_norm.weight,
        attn.k_norm.weight,
        pe[0],
        pe[1],
    )
    if input_guard_reason is not None:
        _VIDEO_MSA_BRANCH_LAST_FALLBACK_REASON = f"qkvpacked_input_{input_guard_reason}"
        return None
    try:
        import flash_attn_interface
    except Exception:
        _VIDEO_MSA_BRANCH_LAST_FALLBACK_REASON = "qkvpacked_flash_attn_missing"
        return None

    def _run_phase(name: str, fn: object) -> object:
        return _profile_video_msa_phase(name, fn)

    v = _run_phase("v_projection", lambda: attn.to_v(x))
    q = _run_phase("q_projection", lambda: attn.to_q(x))
    k = _run_phase("k_projection", lambda: attn.to_k(x))
    qkv = _run_phase(
        "preattention_qkvpack",
        lambda: fused_video_qkv_rmsnorm_rope_pack(
            q,
            k,
            v,
            attn.q_norm.weight,
            attn.k_norm.weight,
            attn.q_norm.eps,
            pe[0],
            pe[1],
        ),
    )
    if qkv is None:
        pack_guard_reason = _fused_video_qkv_rmsnorm_rope_pack_guard_reason(
            q,
            k,
            v,
            attn.q_norm.weight,
            attn.k_norm.weight,
            pe[0],
            pe[1],
        )
        _VIDEO_MSA_BRANCH_LAST_FALLBACK_REASON = (
            f"qkvpacked_pack_{pack_guard_reason}" if pack_guard_reason is not None else "qkvpacked_pack"
        )
        return None
    out = _run_phase("fa3_qkvpacked_attention", lambda: flash_attn_interface.flash_attn_qkvpacked_func(qkv))
    out = out.reshape(x.shape)
    if attn.to_gate_logits is not None:
        out = _video_msa_gated_attention(attn, x, out, _run_phase)
    projected = _run_phase("out_projection_without_bias", lambda: fp8_linear_without_bias(out_linear, out, out_dtype=out.dtype))
    if projected is None:
        _VIDEO_MSA_BRANCH_LAST_FALLBACK_REASON = "qkvpacked_out_projection"
        return None
    return projected, out_linear.bias


def _feedforward_without_out_bias(ff: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    net = getattr(ff, "net", None)
    if net is None or len(net) != 3:
        return None
    out_linear = net[2]
    hidden = net[1](net[0](x))
    if not _can_use_fp8_linear_without_bias(out_linear, hidden):
        return None
    projected = fp8_linear_without_bias(out_linear, hidden, out_dtype=hidden.dtype)
    if projected is None:
        return None
    return projected, out_linear.bias


def _apply_video_text_cross_attention_adaln_from_ada(
    block: object,
    x: torch.Tensor,
    context: torch.Tensor,
    attn: torch.nn.Module,
    timestep: torch.Tensor,
    prompt_timestep: torch.Tensor,
    context_mask: torch.Tensor | None,
) -> torch.Tensor:
    from ltx_core.utils import rms_norm

    if not getattr(block, "cross_attention_adaln", False):
        return attn(rms_norm(x, eps=block.norm_eps), context=context, mask=context_mask)

    if not _can_use_ada_from_table(x, block.scale_shift_table, timestep):
        return x + block._apply_text_cross_attention(
            x,
            context,
            attn,
            block.scale_shift_table,
            getattr(block, "prompt_scale_shift_table", None),
            timestep,
            prompt_timestep,
            context_mask,
            cross_attention_adaln=True,
        )

    batch_size = x.shape[0]
    prompt_scale_shift_table = getattr(block, "prompt_scale_shift_table", None)
    if (
        _FUSE_VIDEO_TEXT_CONTEXT_ADALN
        and prompt_scale_shift_table is not None
        and _can_use_ada_from_table(context, prompt_scale_shift_table, prompt_timestep)
    ):
        encoder_hidden_states = fused_adaln_affine_from_ada(
            context,
            prompt_scale_shift_table,
            prompt_timestep,
            shift_index=0,
            scale_index=1,
        )
    else:
        shift_kv, scale_kv = (
            prompt_scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
            + prompt_timestep.reshape(batch_size, prompt_timestep.shape[1], 2, -1)
        ).unbind(dim=2)
        encoder_hidden_states = context * (1 + scale_kv) + shift_kv
    attn_input = fused_adaln_affine_from_ada(
        rms_norm(x, eps=block.norm_eps),
        block.scale_shift_table,
        timestep,
        shift_index=6,
        scale_index=7,
    )
    no_bias = (
        _attention_without_out_bias(attn, attn_input, context=encoder_hidden_states, mask=context_mask)
        if _FUSE_VIDEO_OUT_BIAS_RESIDUAL
        else None
    )
    if no_bias is not None:
        out, bias = no_bias
        return fused_residual_gate_bias_from_ada(
            x,
            out,
            bias,
            block.scale_shift_table,
            timestep,
            gate_index=8,
            count_video_ada=False,
        )
    out = attn(attn_input, context=encoder_hidden_states, mask=context_mask)
    return fused_residual_gate_from_ada(
        x,
        out,
        block.scale_shift_table,
        timestep,
        gate_index=8,
        count_video_ada=False,
    )


def _video_msa_branch_from_ada(block: object, video: object, vx: torch.Tensor) -> torch.Tensor | None:
    global _VIDEO_MSA_BRANCH_CALLS

    if not _FUSE_VIDEO_MSA_BRANCH:
        return None
    if not (_FUSE_VIDEO_ADA_VALUES and _FUSE_VIDEO_OUT_BIAS_RESIDUAL):
        _record_video_msa_branch_fallback("required_fusions_disabled")
        return None
    if vx.ndim != 3 or vx.shape[0] != 1 or vx.shape[-1] != 4096:
        _record_video_msa_branch_fallback("shape")
        return None
    if _VIDEO_MSA_BRANCH_TOKEN_COUNTS is not None and vx.shape[1] not in _VIDEO_MSA_BRANCH_TOKEN_COUNTS:
        _record_video_msa_branch_fallback("token_count")
        return None
    if getattr(video, "self_attention_mask", None) is not None:
        _record_video_msa_branch_fallback("mask")
        return None
    if getattr(video, "self_attn_all_perturbed", False):
        _record_video_msa_branch_fallback("all_perturbed")
        return None
    if getattr(video, "positional_embeddings", None) is None:
        _record_video_msa_branch_fallback("missing_rope")
        return None
    allow_bf16_out_linear = _VIDEO_MSA_BRANCH_MODE == "direct_bf16_out"
    if not (
        _can_use_fp8_linear_without_bias(block.attn1.to_out[0], vx)
        or (allow_bf16_out_linear and _can_use_bf16_linear_without_bias(block.attn1.to_out[0], vx))
    ):
        _record_video_msa_branch_fallback("out_linear")
        return None

    norm_vx = _profile_video_msa_phase(
        "adazero_from_table",
        lambda: fused_adazero_from_ada(
            vx,
            block.norm_eps,
            block.scale_shift_table,
            video.timesteps,
            shift_index=0,
            scale_index=1,
        ),
    )
    if _VIDEO_MSA_BRANCH_MODE in {"direct", "direct_bf16_out"}:
        no_bias = _video_msa_direct_without_out_bias(
            block.attn1,
            norm_vx,
            pe=video.positional_embeddings,
            perturbation_mask=video.self_attn_perturbation_mask,
            allow_bf16_out_linear=allow_bf16_out_linear,
        )
        if no_bias is None:
            _record_video_msa_branch_fallback("direct_attention_without_out_bias")
            return None
    elif _VIDEO_MSA_BRANCH_MODE == "direct_qkvpacked":
        no_bias = _video_msa_direct_qkvpacked_without_out_bias(
            block.attn1,
            norm_vx,
            pe=video.positional_embeddings,
            perturbation_mask=video.self_attn_perturbation_mask,
        )
        if no_bias is None:
            _record_video_msa_branch_fallback(
                _VIDEO_MSA_BRANCH_LAST_FALLBACK_REASON or "direct_qkvpacked_attention_without_out_bias"
            )
            return None
    else:
        no_bias = _attention_without_out_bias(
            block.attn1,
            norm_vx,
            pe=video.positional_embeddings,
            mask=None,
            perturbation_mask=video.self_attn_perturbation_mask,
            all_perturbed=False,
        )
    if no_bias is None:
        _record_video_msa_branch_fallback("attention_without_out_bias")
        return None

    vx_msa_out, vx_msa_bias = no_bias
    _VIDEO_MSA_BRANCH_CALLS += 1
    return _profile_video_msa_phase(
        "bias_gate_residual",
        lambda: fused_residual_gate_bias_from_ada(
            vx,
            vx_msa_out,
            vx_msa_bias,
            block.scale_shift_table,
            video.timesteps,
            gate_index=2,
        ),
    )


def _patch_ltx_block_residuals() -> None:
    from dataclasses import replace

    from ltx_core.model.transformer.transformer import BasicAVTransformerBlock

    def _forward(
        self: object,
        video: object | None,
        audio: object | None,
    ) -> tuple[object | None, object | None]:
        if video is None and audio is None:
            raise ValueError("At least one of video or audio must be provided")

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        run_vx = video is not None and video.enabled and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax.numel() > 0

        run_a2v = run_vx and (audio is not None and ax.numel() > 0)
        run_v2a = run_ax and (video is not None and vx.numel() > 0)

        if run_vx:
            vx_msa_branch = _video_msa_branch_from_ada(self, video, vx)
            if vx_msa_branch is not None:
                vx = vx_msa_branch
                del vx_msa_branch
            else:
                del vx_msa_branch
                if _FUSE_VIDEO_ADA_VALUES:
                    norm_vx = fused_adazero_from_ada(
                        vx,
                        self.norm_eps,
                        self.scale_shift_table,
                        video.timesteps,
                        shift_index=0,
                        scale_index=1,
                    )
                    vgate_msa = None
                else:
                    vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                        self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3)
                    )
                    norm_vx = self.ada_zero_function(vx, self.norm_eps, vscale_msa, vshift_msa)
                    del vshift_msa, vscale_msa

                vx_msa_no_bias = (
                    _video_self_attention_without_qk_out_bias(
                        self.attn1,
                        norm_vx,
                        pe=video.positional_embeddings,
                        mask=video.self_attention_mask,
                        perturbation_mask=video.self_attn_perturbation_mask,
                        all_perturbed=video.self_attn_all_perturbed,
                    )
                    if (_FUSE_VIDEO_ADA_VALUES and _FUSE_VIDEO_OUT_BIAS_RESIDUAL)
                    else None
                )
                if vx_msa_no_bias is None:
                    vx_msa_no_bias = (
                        _attention_without_out_bias(
                            self.attn1,
                            norm_vx,
                            pe=video.positional_embeddings,
                            mask=video.self_attention_mask,
                            perturbation_mask=video.self_attn_perturbation_mask,
                            all_perturbed=video.self_attn_all_perturbed,
                        )
                        if (_FUSE_VIDEO_ADA_VALUES and _FUSE_VIDEO_OUT_BIAS_RESIDUAL)
                        else None
                    )
                if vx_msa_no_bias is not None:
                    vx_msa_out, vx_msa_bias = vx_msa_no_bias
                    vx = fused_residual_gate_bias_from_ada(
                        vx,
                        vx_msa_out,
                        vx_msa_bias,
                        self.scale_shift_table,
                        video.timesteps,
                        gate_index=2,
                    )
                else:
                    vx_msa_out = self.attn1(
                        norm_vx,
                        pe=video.positional_embeddings,
                        mask=video.self_attention_mask,
                        perturbation_mask=video.self_attn_perturbation_mask,
                        all_perturbed=video.self_attn_all_perturbed,
                    )
                    if _FUSE_VIDEO_ADA_VALUES:
                        vx = fused_residual_gate_from_ada(
                            vx,
                            vx_msa_out,
                            self.scale_shift_table,
                            video.timesteps,
                            gate_index=2,
                        )
                    else:
                        vx = _simple_residual_gate(vx, vx_msa_out, vgate_msa)
                del vgate_msa, norm_vx, vx_msa_out
            if _FUSE_VIDEO_TEXT_ADALN:
                vx = _apply_video_text_cross_attention_adaln_from_ada(
                    self,
                    vx,
                    video.context,
                    self.attn2,
                    video.timesteps,
                    video.prompt_timestep,
                    video.context_mask,
                )
            else:
                vx = vx + self._apply_text_cross_attention(
                    vx,
                    video.context,
                    self.attn2,
                    self.scale_shift_table,
                    getattr(self, "prompt_scale_shift_table", None),
                    video.timesteps,
                    video.prompt_timestep,
                    video.context_mask,
                    cross_attention_adaln=self.cross_attention_adaln,
                )

        if run_ax:
            if _FUSE_AUDIO_ADA_VALUES:
                norm_ax = fused_adazero_from_ada(
                    ax,
                    self.norm_eps,
                    self.audio_scale_shift_table,
                    audio.timesteps,
                    shift_index=0,
                    scale_index=1,
                    counter="audio",
                )
                agate_msa = None
            else:
                ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                    self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3)
                )

                norm_ax = self.ada_zero_function(ax, self.norm_eps, ascale_msa, ashift_msa)
                del ashift_msa, ascale_msa
            ax_msa_out = self.audio_attn1(
                norm_ax,
                pe=audio.positional_embeddings,
                mask=audio.self_attention_mask,
                perturbation_mask=audio.self_attn_perturbation_mask,
                all_perturbed=audio.self_attn_all_perturbed,
            )
            if _FUSE_AUDIO_ADA_VALUES:
                ax = fused_residual_gate_from_ada(
                    ax,
                    ax_msa_out,
                    self.audio_scale_shift_table,
                    audio.timesteps,
                    gate_index=2,
                    counter="audio",
                )
            else:
                ax = _simple_residual_gate(ax, ax_msa_out, agate_msa)
            del agate_msa, norm_ax, ax_msa_out
            ax = ax + self._apply_text_cross_attention(
                ax,
                audio.context,
                self.audio_attn2,
                self.audio_scale_shift_table,
                getattr(self, "audio_prompt_scale_shift_table", None),
                audio.timesteps,
                audio.prompt_timestep,
                audio.context_mask,
                cross_attention_adaln=self.cross_attention_adaln,
            )

        if run_a2v or run_v2a:
            vx_pre_av = vx
            ax_pre_av = ax
            if run_a2v and not video.cross_attn_skip_all:
                scale_ca_video_a2v, shift_ca_video_a2v, gate_out_a2v = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                    slice(0, 2),
                )
                a2v_vx_scaled = self.ada_zero_function(vx_pre_av, self.norm_eps, scale_ca_video_a2v, shift_ca_video_a2v)
                del scale_ca_video_a2v, shift_ca_video_a2v

                scale_ca_audio_a2v, shift_ca_audio_a2v, _ = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_audio,
                    ax.shape[0],
                    audio.cross_scale_shift_timestep,
                    audio.cross_gate_timestep,
                    slice(0, 2),
                )
                a2v_ax_scaled = self.ada_zero_function(ax_pre_av, self.norm_eps, scale_ca_audio_a2v, shift_ca_audio_a2v)
                del scale_ca_audio_a2v, shift_ca_audio_a2v
                vx = vx + (
                    self.audio_to_video_attn(
                        a2v_vx_scaled,
                        context=a2v_ax_scaled,
                        pe=video.cross_positional_embeddings,
                        k_pe=audio.cross_positional_embeddings,
                    )
                    * gate_out_a2v
                    * video.cross_attn_perturbation_mask
                )
                del gate_out_a2v, a2v_vx_scaled, a2v_ax_scaled

            if run_v2a and not audio.cross_attn_skip_all:
                scale_ca_audio_v2a, shift_ca_audio_v2a, gate_out_v2a = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_audio,
                    ax.shape[0],
                    audio.cross_scale_shift_timestep,
                    audio.cross_gate_timestep,
                    slice(2, 4),
                )
                v2a_ax_scaled = self.ada_zero_function(ax_pre_av, self.norm_eps, scale_ca_audio_v2a, shift_ca_audio_v2a)
                del scale_ca_audio_v2a, shift_ca_audio_v2a
                scale_ca_video_v2a, shift_ca_video_v2a, _ = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                    slice(2, 4),
                )
                v2a_vx_scaled = self.ada_zero_function(vx_pre_av, self.norm_eps, scale_ca_video_v2a, shift_ca_video_v2a)
                del scale_ca_video_v2a, shift_ca_video_v2a
                ax = ax + (
                    self.video_to_audio_attn(
                        v2a_ax_scaled,
                        context=v2a_vx_scaled,
                        pe=audio.cross_positional_embeddings,
                        k_pe=video.cross_positional_embeddings,
                    )
                    * gate_out_v2a
                    * audio.cross_attn_perturbation_mask
                )
                del gate_out_v2a, v2a_ax_scaled, v2a_vx_scaled
            del vx_pre_av, ax_pre_av

        if run_vx:
            if _FUSE_VIDEO_ADA_VALUES:
                vx_scaled = fused_adazero_from_ada(
                    vx,
                    self.norm_eps,
                    self.scale_shift_table,
                    video.timesteps,
                    shift_index=3,
                    scale_index=4,
                )
                vx_ff_no_bias = (
                    _feedforward_without_out_bias(self.ff, vx_scaled)
                    if _FUSE_VIDEO_FFN_OUT_BIAS_RESIDUAL
                    else None
                )
                if vx_ff_no_bias is not None:
                    vx_ff_out, vx_ff_bias = vx_ff_no_bias
                    vx = fused_residual_gate_bias_from_ada(
                        vx,
                        vx_ff_out,
                        vx_ff_bias,
                        self.scale_shift_table,
                        video.timesteps,
                        gate_index=5,
                    )
                else:
                    vx_ff_out = self.ff(vx_scaled)
                    vx = fused_residual_gate_from_ada(
                        vx,
                        vx_ff_out,
                        self.scale_shift_table,
                        video.timesteps,
                        gate_index=5,
                    )
                del vx_ff_out, vx_scaled
            else:
                vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                    self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, 6)
                )
                vx_scaled = self.ada_zero_function(vx, self.norm_eps, vscale_mlp, vshift_mlp)
                vx = _simple_residual_gate(vx, self.ff(vx_scaled), vgate_mlp)

                del vshift_mlp, vscale_mlp, vgate_mlp, vx_scaled

        if run_ax:
            if _FUSE_AUDIO_ADA_VALUES:
                ax_scaled = fused_adazero_from_ada(
                    ax,
                    self.norm_eps,
                    self.audio_scale_shift_table,
                    audio.timesteps,
                    shift_index=3,
                    scale_index=4,
                    counter="audio",
                )
                ax_ff_out = self.audio_ff(ax_scaled)
                ax = fused_residual_gate_from_ada(
                    ax,
                    ax_ff_out,
                    self.audio_scale_shift_table,
                    audio.timesteps,
                    gate_index=5,
                    counter="audio",
                )
                del ax_ff_out, ax_scaled
            else:
                ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                    self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, 6)
                )
                ax_scaled = self.ada_zero_function(ax, self.norm_eps, ascale_mlp, ashift_mlp)
                ax = _simple_residual_gate(ax, self.audio_ff(ax_scaled), agate_mlp)

                del ashift_mlp, ascale_mlp, agate_mlp, ax_scaled

        return replace(video, x=vx) if video is not None else None, replace(audio, x=ax) if audio is not None else None

    BasicAVTransformerBlock.forward = _forward


try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except Exception:  # pragma: no cover - import fallback for CPU-only local tests
    triton = None
    tl = None
    libdevice = None


if triton is not None:

    @triton.jit
    def _round_f32_to_bf16_f32(x):
        bits = x.to(tl.int32, bitcast=True)
        lsb = (bits >> 16) & 1
        rounded = (bits + 0x7FFF + lsb) & -65536
        return rounded.to(tl.float32, bitcast=True)

    @triton.jit
    def _adazero_kernel(
        x_ptr,
        scale_ptr,
        shift_ptr,
        y_ptr,
        rows: tl.constexpr,
        cols: tl.constexpr,
        scale_rows: tl.constexpr,
        scale_stride_row: tl.constexpr,
        scale_stride_col: tl.constexpr,
        shift_stride_row: tl.constexpr,
        shift_stride_col: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_COLS)
        mask = offsets < cols
        x_offsets = row * cols + offsets
        if scale_rows == 1:
            scale_row = 0
        elif scale_rows == rows:
            scale_row = row
        else:
            scale_row = row // (rows // scale_rows)
        scale_offsets = scale_row * scale_stride_row + offsets * scale_stride_col
        shift_offsets = scale_row * shift_stride_row + offsets * shift_stride_col

        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + scale_offsets, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + shift_offsets, mask=mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(x * x, axis=0) / cols
        inv_rms = tl.rsqrt(mean_square + eps)
        y = x * inv_rms * (1.0 + scale) + shift
        tl.store(y_ptr + x_offsets, y, mask=mask)

    @triton.jit
    def _adazero_from_ada_kernel(
        x_ptr,
        table_ptr,
        timestep_ptr,
        y_ptr,
        rows: tl.constexpr,
        cols: tl.constexpr,
        tokens: tl.constexpr,
        table_stride_row: tl.constexpr,
        table_stride_col: tl.constexpr,
        timestep_stride_b: tl.constexpr,
        timestep_stride_t: tl.constexpr,
        timestep_stride_p: tl.constexpr,
        timestep_stride_c: tl.constexpr,
        shift_index: tl.constexpr,
        scale_index: tl.constexpr,
        round_table: tl.constexpr,
        round_ada: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_COLS)
        mask = offsets < cols
        batch = row // tokens
        token = row - batch * tokens
        x_offsets = row * cols + offsets

        shift_table = tl.load(
            table_ptr + shift_index * table_stride_row + offsets * table_stride_col,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        scale_table = tl.load(
            table_ptr + scale_index * table_stride_row + offsets * table_stride_col,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if round_table:
            shift_table = _round_f32_to_bf16_f32(shift_table)
            scale_table = _round_f32_to_bf16_f32(scale_table)
        shift = shift_table + tl.load(
            timestep_ptr
            + batch * timestep_stride_b
            + token * timestep_stride_t
            + shift_index * timestep_stride_p
            + offsets * timestep_stride_c,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        scale = scale_table + tl.load(
            timestep_ptr
            + batch * timestep_stride_b
            + token * timestep_stride_t
            + scale_index * timestep_stride_p
            + offsets * timestep_stride_c,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if round_ada:
            shift = _round_f32_to_bf16_f32(shift)
            scale = _round_f32_to_bf16_f32(scale)

        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(x * x, axis=0) / cols
        inv_rms = tl.rsqrt(mean_square + eps)
        y = x * inv_rms * (1.0 + scale) + shift
        tl.store(y_ptr + x_offsets, y, mask=mask)

    @triton.jit
    def _fp8_quantize_e4m3_kernel(
        x_ptr,
        scale_ptr,
        q_ptr,
        n: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr).to(tl.float32)
        fp8_min = -448.0
        fp8_max = 448.0
        q = tl.minimum(tl.maximum(x / scale, fp8_min), fp8_max).to(tl.float8e4nv)
        tl.store(q_ptr + offsets, q, mask=mask)

    @triton.jit
    def _bias_add_bf16_kernel(
        x_ptr,
        bias_ptr,
        out_ptr,
        n: tl.constexpr,
        cols: tl.constexpr,
        round_bias: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        col = offsets % cols
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + col, mask=mask, other=0.0).to(tl.float32)
        if round_bias:
            bias = _round_f32_to_bf16_f32(bias)
        out = x + bias
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _adaln_affine_from_ada_kernel(
        x_ptr,
        table_ptr,
        timestep_ptr,
        y_ptr,
        n: tl.constexpr,
        cols: tl.constexpr,
        tokens: tl.constexpr,
        table_stride_row: tl.constexpr,
        table_stride_col: tl.constexpr,
        timestep_stride_b: tl.constexpr,
        timestep_stride_t: tl.constexpr,
        timestep_stride_p: tl.constexpr,
        timestep_stride_c: tl.constexpr,
        shift_index: tl.constexpr,
        scale_index: tl.constexpr,
        round_table: tl.constexpr,
        round_ada: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        col = offsets % cols
        row = offsets // cols
        batch = row // tokens
        token = row - batch * tokens

        shift_table = tl.load(
            table_ptr + shift_index * table_stride_row + col * table_stride_col,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        scale_table = tl.load(
            table_ptr + scale_index * table_stride_row + col * table_stride_col,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if round_table:
            shift_table = _round_f32_to_bf16_f32(shift_table)
            scale_table = _round_f32_to_bf16_f32(scale_table)
        shift = shift_table + tl.load(
            timestep_ptr
            + batch * timestep_stride_b
            + token * timestep_stride_t
            + shift_index * timestep_stride_p
            + col * timestep_stride_c,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        scale = scale_table + tl.load(
            timestep_ptr
            + batch * timestep_stride_b
            + token * timestep_stride_t
            + scale_index * timestep_stride_p
            + col * timestep_stride_c,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if round_ada:
            shift = _round_f32_to_bf16_f32(shift)
            scale = _round_f32_to_bf16_f32(scale)

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        factor = _round_f32_to_bf16_f32(1.0 + scale)
        product = _round_f32_to_bf16_f32(x * factor)
        out = product + shift
        tl.store(y_ptr + offsets, out, mask=mask)

    @triton.jit
    def _gelu_tanh_fp8_quantize_e4m3_kernel(
        x_ptr,
        scale_ptr,
        q_ptr,
        n: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tanh_arg = 0.7978845608028654 * (x + 0.044715 * x * x * x)
        gelu = (0.5 * x * (1.0 + libdevice.tanh(tanh_arg))).to(tl.bfloat16).to(tl.float32)
        scale = tl.load(scale_ptr).to(tl.float32)
        q = tl.minimum(tl.maximum(gelu / scale, -448.0), 448.0).to(tl.float8e4nv)
        tl.store(q_ptr + offsets, q, mask=mask)

    @triton.jit
    def _residual_gate_kernel(
        x_ptr,
        y_ptr,
        gate_ptr,
        out_ptr,
        n: tl.constexpr,
        cols: tl.constexpr,
        gate_rows: tl.constexpr,
        gate_stride_row: tl.constexpr,
        gate_stride_col: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        col = offsets % cols
        row = offsets // cols
        if gate_rows == 1:
            gate_row = 0
        elif gate_rows == n // cols:
            gate_row = row
        else:
            gate_row = row // ((n // cols) // gate_rows)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + gate_row * gate_stride_row + col * gate_stride_col, mask=mask, other=0.0).to(
            tl.float32
        )
        product = _round_f32_to_bf16_f32(y * gate)
        out = x + product
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _head_gate_mul_kernel(
        x_ptr,
        gate_ptr,
        out_ptr,
        n: tl.constexpr,
        cols: tl.constexpr,
        heads: tl.constexpr,
        dim_head: tl.constexpr,
        gate_stride_row: tl.constexpr,
        gate_stride_head: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        col = offsets % cols
        row = offsets // cols
        head = col // dim_head
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(
            gate_ptr + row * gate_stride_row + head * gate_stride_head,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        out = _round_f32_to_bf16_f32(x * gate)
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _residual_gate_from_ada_kernel(
        x_ptr,
        y_ptr,
        table_ptr,
        timestep_ptr,
        out_ptr,
        n: tl.constexpr,
        cols: tl.constexpr,
        tokens: tl.constexpr,
        table_stride_row: tl.constexpr,
        table_stride_col: tl.constexpr,
        timestep_stride_b: tl.constexpr,
        timestep_stride_t: tl.constexpr,
        timestep_stride_p: tl.constexpr,
        timestep_stride_c: tl.constexpr,
        gate_index: tl.constexpr,
        round_table: tl.constexpr,
        round_ada: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        col = offsets % cols
        row = offsets // cols
        batch = row // tokens
        token = row - batch * tokens

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gate_table = tl.load(
            table_ptr + gate_index * table_stride_row + col * table_stride_col,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if round_table:
            gate_table = _round_f32_to_bf16_f32(gate_table)
        gate = gate_table + tl.load(
            timestep_ptr
            + batch * timestep_stride_b
            + token * timestep_stride_t
            + gate_index * timestep_stride_p
            + col * timestep_stride_c,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if round_ada:
            gate = _round_f32_to_bf16_f32(gate)
        product = _round_f32_to_bf16_f32(y * gate)
        out = x + product
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _residual_gate_bias_from_ada_kernel(
        x_ptr,
        y_ptr,
        bias_ptr,
        table_ptr,
        timestep_ptr,
        out_ptr,
        n: tl.constexpr,
        cols: tl.constexpr,
        tokens: tl.constexpr,
        round_bias: tl.constexpr,
        table_stride_row: tl.constexpr,
        table_stride_col: tl.constexpr,
        timestep_stride_b: tl.constexpr,
        timestep_stride_t: tl.constexpr,
        timestep_stride_p: tl.constexpr,
        timestep_stride_c: tl.constexpr,
        gate_index: tl.constexpr,
        round_table: tl.constexpr,
        round_ada: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        col = offsets % cols
        row = offsets // cols
        batch = row // tokens
        token = row - batch * tokens

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + col, mask=mask, other=0.0).to(tl.float32)
        if round_bias:
            bias = _round_f32_to_bf16_f32(bias)
        y_biased = _round_f32_to_bf16_f32(y + bias)

        gate_table = tl.load(
            table_ptr + gate_index * table_stride_row + col * table_stride_col,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if round_table:
            gate_table = _round_f32_to_bf16_f32(gate_table)
        gate = gate_table + tl.load(
            timestep_ptr
            + batch * timestep_stride_b
            + token * timestep_stride_t
            + gate_index * timestep_stride_p
            + col * timestep_stride_c,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if round_ada:
            gate = _round_f32_to_bf16_f32(gate)

        product = _round_f32_to_bf16_f32(y_biased * gate)
        out = x + product
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _mul_from_ada_kernel(
        x_ptr,
        table_ptr,
        timestep_ptr,
        out_ptr,
        n: tl.constexpr,
        cols: tl.constexpr,
        tokens: tl.constexpr,
        table_stride_row: tl.constexpr,
        table_stride_col: tl.constexpr,
        timestep_stride_b: tl.constexpr,
        timestep_stride_t: tl.constexpr,
        timestep_stride_p: tl.constexpr,
        timestep_stride_c: tl.constexpr,
        gate_index: tl.constexpr,
        round_table: tl.constexpr,
        round_ada: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        col = offsets % cols
        row = offsets // cols
        batch = row // tokens
        token = row - batch * tokens

        gate_table = tl.load(
            table_ptr + gate_index * table_stride_row + col * table_stride_col,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if round_table:
            gate_table = _round_f32_to_bf16_f32(gate_table)
        gate = gate_table + tl.load(
            timestep_ptr
            + batch * timestep_stride_b
            + token * timestep_stride_t
            + gate_index * timestep_stride_p
            + col * timestep_stride_c,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if round_ada:
            gate = _round_f32_to_bf16_f32(gate)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        out = _round_f32_to_bf16_f32(x * gate)
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _fp8_per_head_amax_kernel(
        x_ptr,
        partial_ptr,
        tokens: tl.constexpr,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        elems_per_head: tl.constexpr,
        blocks_per_head: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        block_id = pid % blocks_per_head
        head = (pid // blocks_per_head) % heads
        batch = pid // (blocks_per_head * heads)

        offsets = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < elems_per_head
        token = offsets // head_dim
        dim = offsets - token * head_dim
        x_offsets = ((batch * tokens + token) * heads + head) * head_dim + dim
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        amax = tl.max(tl.abs(x), axis=0)
        tl.store(partial_ptr + pid, amax)

    @triton.jit
    def _fp8_per_head_scale_kernel(
        partial_ptr,
        scale_ptr,
        blocks_per_head: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < blocks_per_head
        amax = tl.max(tl.load(partial_ptr + pid * blocks_per_head + offsets, mask=mask, other=0.0), axis=0)
        scale = tl.maximum(amax / 448.0, 1.0e-12)
        tl.store(scale_ptr + pid, scale)

    @triton.jit
    def _fp8_per_head_quantize_kernel(
        x_ptr,
        scale_ptr,
        q_ptr,
        tokens: tl.constexpr,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        elems_per_head: tl.constexpr,
        blocks_per_head: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        block_id = pid % blocks_per_head
        head = (pid // blocks_per_head) % heads
        batch = pid // (blocks_per_head * heads)

        offsets = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < elems_per_head
        token = offsets // head_dim
        dim = offsets - token * head_dim
        tensor_offsets = ((batch * tokens + token) * heads + head) * head_dim + dim
        x = tl.load(x_ptr + tensor_offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + batch * heads + head).to(tl.float32)
        q = tl.minimum(tl.maximum(x / scale, -448.0), 448.0).to(tl.float8e4nv)
        tl.store(q_ptr + tensor_offsets, q, mask=mask)

    @triton.jit
    def _video_rmsnorm_rope_kernel(
        x_ptr,
        weight_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        rows: tl.constexpr,
        cols: tl.constexpr,
        tokens: tl.constexpr,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        half_dim: tl.constexpr,
        cos_stride_b: tl.constexpr,
        cos_stride_h: tl.constexpr,
        cos_stride_t: tl.constexpr,
        cos_stride_d: tl.constexpr,
        sin_stride_b: tl.constexpr,
        sin_stride_h: tl.constexpr,
        sin_stride_t: tl.constexpr,
        sin_stride_d: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_COLS)
        mask = offsets < cols

        row_offsets = row * cols + offsets
        x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(x * x, axis=0) / cols
        inv_rms = tl.rsqrt(mean_square + eps)
        normed = x * inv_rms * weight

        dim = offsets % head_dim
        head = offsets // head_dim
        token = row % tokens
        batch = row // tokens
        half = dim % half_dim
        is_second = dim >= half_dim
        other_offsets = offsets + tl.where(is_second, -half_dim, half_dim)
        other_x = tl.load(x_ptr + row * cols + other_offsets, mask=mask, other=0.0).to(tl.float32)
        other_weight = tl.load(weight_ptr + other_offsets, mask=mask, other=0.0).to(tl.float32)
        other_normed = other_x * inv_rms * other_weight

        pe_offsets = (
            batch * cos_stride_b
            + head * cos_stride_h
            + token * cos_stride_t
            + half * cos_stride_d
        )
        cos = tl.load(cos_ptr + pe_offsets, mask=mask, other=1.0).to(tl.float32)
        sin = tl.load(sin_ptr + (
            batch * sin_stride_b
            + head * sin_stride_h
            + token * sin_stride_t
            + half * sin_stride_d
        ), mask=mask, other=0.0).to(tl.float32)

        rotated = tl.where(is_second, normed * cos + other_normed * sin, normed * cos - other_normed * sin)
        tl.store(out_ptr + row_offsets, rotated, mask=mask)

    @triton.jit
    def _video_qk_rmsnorm_rope_kernel(
        q_ptr,
        k_ptr,
        q_weight_ptr,
        k_weight_ptr,
        cos_ptr,
        sin_ptr,
        q_out_ptr,
        k_out_ptr,
        rows: tl.constexpr,
        cols: tl.constexpr,
        tokens: tl.constexpr,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        half_dim: tl.constexpr,
        cos_stride_b: tl.constexpr,
        cos_stride_h: tl.constexpr,
        cos_stride_t: tl.constexpr,
        cos_stride_d: tl.constexpr,
        sin_stride_b: tl.constexpr,
        sin_stride_h: tl.constexpr,
        sin_stride_t: tl.constexpr,
        sin_stride_d: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_COLS)
        mask = offsets < cols

        row_offsets = row * cols + offsets
        q = tl.load(q_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
        k = tl.load(k_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
        q_weight = tl.load(q_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        k_weight = tl.load(k_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        q_inv_rms = tl.rsqrt(tl.sum(q * q, axis=0) / cols + eps)
        k_inv_rms = tl.rsqrt(tl.sum(k * k, axis=0) / cols + eps)
        q_normed = q * q_inv_rms * q_weight
        k_normed = k * k_inv_rms * k_weight

        dim = offsets % head_dim
        head = offsets // head_dim
        token = row % tokens
        batch = row // tokens
        half = dim % half_dim
        is_second = dim >= half_dim
        other_offsets = offsets + tl.where(is_second, -half_dim, half_dim)
        other_row_offsets = row * cols + other_offsets

        other_q = tl.load(q_ptr + other_row_offsets, mask=mask, other=0.0).to(tl.float32)
        other_k = tl.load(k_ptr + other_row_offsets, mask=mask, other=0.0).to(tl.float32)
        other_q_weight = tl.load(q_weight_ptr + other_offsets, mask=mask, other=0.0).to(tl.float32)
        other_k_weight = tl.load(k_weight_ptr + other_offsets, mask=mask, other=0.0).to(tl.float32)
        other_q_normed = other_q * q_inv_rms * other_q_weight
        other_k_normed = other_k * k_inv_rms * other_k_weight

        pe_offsets = batch * cos_stride_b + head * cos_stride_h + token * cos_stride_t + half * cos_stride_d
        cos = tl.load(cos_ptr + pe_offsets, mask=mask, other=1.0).to(tl.float32)
        sin = tl.load(
            sin_ptr + batch * sin_stride_b + head * sin_stride_h + token * sin_stride_t + half * sin_stride_d,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        q_rotated = tl.where(
            is_second,
            q_normed * cos + other_q_normed * sin,
            q_normed * cos - other_q_normed * sin,
        )
        k_rotated = tl.where(
            is_second,
            k_normed * cos + other_k_normed * sin,
            k_normed * cos - other_k_normed * sin,
        )
        tl.store(q_out_ptr + row_offsets, q_rotated, mask=mask)
        tl.store(k_out_ptr + row_offsets, k_rotated, mask=mask)

    @triton.jit
    def _video_qkv_rmsnorm_rope_pack_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        q_weight_ptr,
        k_weight_ptr,
        cos_ptr,
        sin_ptr,
        qkv_ptr,
        rows: tl.constexpr,
        cols: tl.constexpr,
        tokens: tl.constexpr,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        half_dim: tl.constexpr,
        cos_stride_b: tl.constexpr,
        cos_stride_h: tl.constexpr,
        cos_stride_t: tl.constexpr,
        cos_stride_d: tl.constexpr,
        sin_stride_b: tl.constexpr,
        sin_stride_h: tl.constexpr,
        sin_stride_t: tl.constexpr,
        sin_stride_d: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_COLS)
        mask = offsets < cols

        row_offsets = row * cols + offsets
        q = tl.load(q_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
        k = tl.load(k_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(v_ptr + row_offsets, mask=mask, other=0.0)
        q_weight = tl.load(q_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        k_weight = tl.load(k_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        q_inv_rms = tl.rsqrt(tl.sum(q * q, axis=0) / cols + eps)
        k_inv_rms = tl.rsqrt(tl.sum(k * k, axis=0) / cols + eps)
        q_normed = q * q_inv_rms * q_weight
        k_normed = k * k_inv_rms * k_weight

        dim = offsets % head_dim
        head = offsets // head_dim
        token = row % tokens
        batch = row // tokens
        half = dim % half_dim
        is_second = dim >= half_dim
        other_offsets = offsets + tl.where(is_second, -half_dim, half_dim)
        other_row_offsets = row * cols + other_offsets

        other_q = tl.load(q_ptr + other_row_offsets, mask=mask, other=0.0).to(tl.float32)
        other_k = tl.load(k_ptr + other_row_offsets, mask=mask, other=0.0).to(tl.float32)
        other_q_weight = tl.load(q_weight_ptr + other_offsets, mask=mask, other=0.0).to(tl.float32)
        other_k_weight = tl.load(k_weight_ptr + other_offsets, mask=mask, other=0.0).to(tl.float32)
        other_q_normed = other_q * q_inv_rms * other_q_weight
        other_k_normed = other_k * k_inv_rms * other_k_weight

        pe_offsets = batch * cos_stride_b + head * cos_stride_h + token * cos_stride_t + half * cos_stride_d
        cos = tl.load(cos_ptr + pe_offsets, mask=mask, other=1.0).to(tl.float32)
        sin = tl.load(
            sin_ptr + batch * sin_stride_b + head * sin_stride_h + token * sin_stride_t + half * sin_stride_d,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        q_rotated = tl.where(
            is_second,
            q_normed * cos + other_q_normed * sin,
            q_normed * cos - other_q_normed * sin,
        )
        k_rotated = tl.where(
            is_second,
            k_normed * cos + other_k_normed * sin,
            k_normed * cos - other_k_normed * sin,
        )

        packed_offsets = row * 3 * cols + offsets
        tl.store(qkv_ptr + packed_offsets, q_rotated, mask=mask)
        tl.store(qkv_ptr + packed_offsets + cols, k_rotated, mask=mask)
        tl.store(qkv_ptr + packed_offsets + 2 * cols, v, mask=mask)

    @triton.jit
    def _video_bias_rmsnorm_rope_kernel(
        x_ptr,
        bias_ptr,
        weight_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        rows: tl.constexpr,
        cols: tl.constexpr,
        tokens: tl.constexpr,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        half_dim: tl.constexpr,
        round_bias: tl.constexpr,
        cos_stride_b: tl.constexpr,
        cos_stride_h: tl.constexpr,
        cos_stride_t: tl.constexpr,
        cos_stride_d: tl.constexpr,
        sin_stride_b: tl.constexpr,
        sin_stride_h: tl.constexpr,
        sin_stride_t: tl.constexpr,
        sin_stride_d: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_COLS)
        mask = offsets < cols

        row_offsets = row * cols + offsets
        x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        if round_bias:
            bias = _round_f32_to_bf16_f32(bias)
        x = _round_f32_to_bf16_f32(x + bias)

        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(x * x, axis=0) / cols
        inv_rms = tl.rsqrt(mean_square + eps)
        normed = x * inv_rms * weight

        dim = offsets % head_dim
        head = offsets // head_dim
        token = row % tokens
        batch = row // tokens
        half = dim % half_dim
        is_second = dim >= half_dim
        other_offsets = offsets + tl.where(is_second, -half_dim, half_dim)
        other_x = tl.load(x_ptr + row * cols + other_offsets, mask=mask, other=0.0).to(tl.float32)
        other_bias = tl.load(bias_ptr + other_offsets, mask=mask, other=0.0).to(tl.float32)
        if round_bias:
            other_bias = _round_f32_to_bf16_f32(other_bias)
        other_x = _round_f32_to_bf16_f32(other_x + other_bias)
        other_weight = tl.load(weight_ptr + other_offsets, mask=mask, other=0.0).to(tl.float32)
        other_normed = other_x * inv_rms * other_weight

        pe_offsets = batch * cos_stride_b + head * cos_stride_h + token * cos_stride_t + half * cos_stride_d
        cos = tl.load(cos_ptr + pe_offsets, mask=mask, other=1.0).to(tl.float32)
        sin = tl.load(
            sin_ptr + batch * sin_stride_b + head * sin_stride_h + token * sin_stride_t + half * sin_stride_d,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        rotated = tl.where(is_second, normed * cos + other_normed * sin, normed * cos - other_normed * sin)
        tl.store(out_ptr + row_offsets, rotated, mask=mask)

else:
    _adazero_kernel = None
    _adazero_from_ada_kernel = None
    _adaln_affine_from_ada_kernel = None
    _fp8_quantize_e4m3_kernel = None
    _bias_add_bf16_kernel = None
    _gelu_tanh_fp8_quantize_e4m3_kernel = None
    _residual_gate_kernel = None
    _head_gate_mul_kernel = None
    _residual_gate_from_ada_kernel = None
    _residual_gate_bias_from_ada_kernel = None
    _mul_from_ada_kernel = None
    _fp8_per_head_amax_kernel = None
    _fp8_per_head_scale_kernel = None
    _fp8_per_head_quantize_kernel = None
    _video_rmsnorm_rope_kernel = None
    _video_qkv_rmsnorm_rope_pack_kernel = None
    _video_bias_rmsnorm_rope_kernel = None
