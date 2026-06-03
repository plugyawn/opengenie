import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only Triton kernel tests")


def test_fused_adazero_respects_noncontiguous_scale_shift_strides() -> None:
    from ltx_serve.triton_ltx_ops import fused_adazero

    torch.manual_seed(0)
    rows = 7
    cols = 2048
    x = torch.randn((1, rows, cols), device="cuda", dtype=torch.bfloat16)
    scale_shift = torch.randn((1, rows, 3, cols), device="cuda", dtype=torch.bfloat16)
    scale = scale_shift[:, :, 1, :]
    shift = scale_shift[:, :, 2, :]

    assert not scale.is_contiguous()
    assert scale.stride(1) == 3 * cols

    out = fused_adazero(x, 1e-6, scale, shift)
    ref = torch.nn.functional.rms_norm(x, (cols,), eps=1e-6) * (1 + scale) + shift
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), ref.float(), rtol=0, atol=0.0625)


def test_fused_residual_gate_matches_pytorch_two_kernel_rounding_for_strided_gate() -> None:
    from ltx_serve.triton_ltx_ops import fused_residual_gate_bf16

    torch.manual_seed(1)
    rows = 7
    cols = 2048
    x = torch.randn((1, rows, cols), device="cuda", dtype=torch.bfloat16)
    y = torch.randn((1, rows, cols), device="cuda", dtype=torch.bfloat16)
    gates = torch.randn((1, rows, 3, cols), device="cuda", dtype=torch.bfloat16)
    gate = gates[:, :, 1, :]

    assert not gate.is_contiguous()
    assert gate.stride(1) == 3 * cols

    out = fused_residual_gate_bf16(x, y, gate)
    ref = x + y * gate
    torch.cuda.synchronize()

    assert torch.equal(out, ref)


def test_fused_head_gate_mul_matches_pytorch_reference() -> None:
    from ltx_serve.triton_ltx_ops import fused_head_gate_mul_bf16

    torch.manual_seed(11)
    batch = 1
    tokens = 13
    heads = 32
    dim_head = 128
    out = torch.randn((batch, tokens, heads * dim_head), device="cuda", dtype=torch.bfloat16)
    gate_logits = torch.randn((batch, tokens, heads), device="cuda", dtype=torch.bfloat16)
    gates = 2.0 * torch.sigmoid(gate_logits)

    fused = fused_head_gate_mul_bf16(out, gates, heads, dim_head)
    ref = (out.view(batch, tokens, heads, dim_head) * gates.unsqueeze(-1)).view(
        batch, tokens, heads * dim_head
    )
    torch.cuda.synchronize()

    assert torch.equal(fused, ref)


def test_fused_adazero_from_ada_matches_materialized_reference() -> None:
    from ltx_serve.triton_ltx_ops import fused_adazero_from_ada

    torch.manual_seed(2)
    batch = 1
    rows = 9
    cols = 2048
    params = 6
    x = torch.randn((batch, rows, cols), device="cuda", dtype=torch.bfloat16)
    table = torch.randn((params, cols), device="cuda", dtype=torch.float32)
    timestep = torch.randn((batch, rows, params * cols), device="cuda", dtype=torch.bfloat16)

    out = fused_adazero_from_ada(x, 1e-6, table, timestep, shift_index=3, scale_index=4)

    ts_view = timestep.reshape(batch, rows, params, cols)
    table_cast = table.to(dtype=timestep.dtype)
    shift = table_cast[3].view(1, 1, cols) + ts_view[:, :, 3, :]
    scale = table_cast[4].view(1, 1, cols) + ts_view[:, :, 4, :]
    ref = torch.nn.functional.rms_norm(x, (cols,), eps=1e-6) * (1 + scale) + shift
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), ref.float(), rtol=0, atol=0.0625)


def test_fused_residual_gate_from_ada_matches_materialized_reference() -> None:
    from ltx_serve.triton_ltx_ops import fused_residual_gate_from_ada

    torch.manual_seed(3)
    batch = 1
    rows = 9
    cols = 2048
    params = 6
    x = torch.randn((batch, rows, cols), device="cuda", dtype=torch.bfloat16)
    y = torch.randn((batch, rows, cols), device="cuda", dtype=torch.bfloat16)
    table = torch.randn((params, cols), device="cuda", dtype=torch.float32)
    timestep = torch.randn((batch, rows, params * cols), device="cuda", dtype=torch.bfloat16)

    out = fused_residual_gate_from_ada(x, y, table, timestep, gate_index=5)

    ts_view = timestep.reshape(batch, rows, params, cols)
    table_cast = table.to(dtype=timestep.dtype)
    gate = table_cast[5].view(1, 1, cols) + ts_view[:, :, 5, :]
    ref = x + y * gate
    torch.cuda.synchronize()

    assert torch.equal(out, ref)


def test_fused_adaln_affine_from_ada_matches_materialized_reference() -> None:
    from ltx_serve.triton_ltx_ops import fused_adaln_affine_from_ada

    torch.manual_seed(4)
    batch = 1
    rows = 9
    cols = 2048
    params = 9
    x = torch.randn((batch, rows, cols), device="cuda", dtype=torch.bfloat16)
    table = torch.randn((params, cols), device="cuda", dtype=torch.float32)
    timestep = torch.randn((batch, rows, params * cols), device="cuda", dtype=torch.bfloat16)

    out = fused_adaln_affine_from_ada(x, table, timestep, shift_index=6, scale_index=7)

    ts_view = timestep.reshape(batch, rows, params, cols)
    table_cast = table.to(dtype=timestep.dtype)
    shift = table_cast[6].view(1, 1, cols) + ts_view[:, :, 6, :]
    scale = table_cast[7].view(1, 1, cols) + ts_view[:, :, 7, :]
    ref = x * (1 + scale) + shift
    torch.cuda.synchronize()

    assert torch.equal(out, ref)


def test_fused_mul_from_ada_matches_materialized_reference() -> None:
    from ltx_serve.triton_ltx_ops import fused_mul_from_ada

    torch.manual_seed(5)
    batch = 1
    rows = 9
    cols = 2048
    params = 9
    x = torch.randn((batch, rows, cols), device="cuda", dtype=torch.bfloat16)
    table = torch.randn((params, cols), device="cuda", dtype=torch.float32)
    timestep = torch.randn((batch, rows, params * cols), device="cuda", dtype=torch.bfloat16)

    out = fused_mul_from_ada(x, table, timestep, gate_index=8)

    ts_view = timestep.reshape(batch, rows, params, cols)
    table_cast = table.to(dtype=timestep.dtype)
    gate = table_cast[8].view(1, 1, cols) + ts_view[:, :, 8, :]
    ref = x * gate
    torch.cuda.synchronize()

    assert torch.equal(out, ref)


def test_fused_bias_add_bf16_matches_materialized_reference() -> None:
    from ltx_serve.triton_ltx_ops import fused_bias_add_bf16

    torch.manual_seed(6)
    rows = 11
    cols = 2048
    x = torch.randn((rows, cols), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((cols,), device="cuda", dtype=torch.float32)

    out = fused_bias_add_bf16(x, bias)
    ref = x + bias.to(x.dtype)
    torch.cuda.synchronize()

    assert torch.equal(out, ref)


def test_fused_residual_gate_bias_from_ada_matches_materialized_reference() -> None:
    from ltx_serve.triton_ltx_ops import fused_residual_gate_bias_from_ada

    torch.manual_seed(7)
    batch = 1
    rows = 9
    cols = 2048
    params = 9
    x = torch.randn((batch, rows, cols), device="cuda", dtype=torch.bfloat16)
    y = torch.randn((batch, rows, cols), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((cols,), device="cuda", dtype=torch.float32)
    table = torch.randn((params, cols), device="cuda", dtype=torch.float32)
    timestep = torch.randn((batch, rows, params * cols), device="cuda", dtype=torch.bfloat16)

    out = fused_residual_gate_bias_from_ada(x, y, bias, table, timestep, gate_index=8)

    ts_view = timestep.reshape(batch, rows, params, cols)
    table_cast = table.to(dtype=timestep.dtype)
    gate = table_cast[8].view(1, 1, cols) + ts_view[:, :, 8, :]
    ref = x + (y + bias.to(y.dtype)) * gate
    torch.cuda.synchronize()

    assert torch.equal(out, ref)


def test_fused_video_qk_rmsnorm_rope_matches_reference() -> None:
    from ltx_serve.triton_ltx_ops import fused_video_qk_rmsnorm_rope

    torch.manual_seed(8)
    tokens = 17
    heads = 32
    head_dim = 128
    cols = heads * head_dim
    q = torch.randn((1, tokens, cols), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    q_weight = torch.randn((cols,), device="cuda", dtype=torch.bfloat16)
    k_weight = torch.randn((cols,), device="cuda", dtype=torch.bfloat16)
    freqs = torch.randn((1, heads, tokens, head_dim // 2), device="cuda", dtype=torch.float32)
    cos = freqs.cos().to(torch.bfloat16).contiguous()
    sin = freqs.sin().to(torch.bfloat16).contiguous()

    out_q, out_k = fused_video_qk_rmsnorm_rope(q, k, q_weight, k_weight, 1e-6, cos, sin)
    ref_q = _torch_video_rmsnorm_split_rope(q, q_weight, 1e-6, cos, sin)
    ref_k = _torch_video_rmsnorm_split_rope(k, k_weight, 1e-6, cos, sin)
    torch.cuda.synchronize()

    torch.testing.assert_close(out_q.float(), ref_q.float(), rtol=0, atol=0.125)
    torch.testing.assert_close(out_k.float(), ref_k.float(), rtol=0, atol=0.125)


def _torch_video_rmsnorm_split_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x = torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)
    batch, tokens, dim = x.shape
    heads = cos.shape[1]
    head_dim = dim // heads
    half_dim = head_dim // 2
    xh = x.reshape(batch, tokens, heads, head_dim).transpose(1, 2)
    first = xh[..., :half_dim]
    second = xh[..., half_dim:]
    out_first = first * cos - second * sin
    out_second = second * cos + first * sin
    return torch.cat([out_first, out_second], dim=-1).transpose(1, 2).reshape(batch, tokens, dim)
