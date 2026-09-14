# MX format helpers on top of microsoft/microxcaling
import sys, os
import numpy as np
import torch

_VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "microxcaling")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from mx.mx_ops import _quantize_mx, _shared_exponents, _reshape_to_blocks, _undo_reshape_to_blocks
from mx.formats import ElemFormat, _get_format_params

BLOCK = 32
SCALE_BITS = 8

FORMATS = {"mxfp4": "fp4_e2m1", "mxfp8": "fp8_e4m3", "mxfp8_e5m2": "fp8_e5m2"}
ELEM_BITS = {"fp4_e2m1": 4, "fp8_e4m3": 8, "fp8_e5m2": 8}


def _level_table(elem_format):
    """All representable values of an element format, indexed by its bit code."""
    ebits, mbits, emax, max_norm, _ = _get_format_params(elem_format)
    man_bits = mbits - 2  # mbits counts sign and implicit one
    vals = np.zeros(2 ** (1 + ebits + man_bits), dtype=np.float64)
    bias = 2 ** (ebits - 1) - 1
    for s in (0, 1):
        for e in range(2 ** ebits):
            for m in range(2 ** man_bits):
                code = (s << (ebits + man_bits)) | (e << man_bits) | m
                if e == 0:
                    v = 2.0 ** (1 - bias) * (m / 2 ** man_bits)
                else:
                    v = 2.0 ** (e - bias) * (1 + m / 2 ** man_bits)
                if v > max_norm:
                    v = np.nan
                vals[code] = -v if s else v
    return vals


_TABLES = {}


def level_table(elem_format):
    if elem_format not in _TABLES:
        _TABLES[elem_format] = _level_table(elem_format)
    return _TABLES[elem_format]


def sorted_levels(elem_format):
    """Finite levels sorted ascending, with the code of each."""
    t = level_table(elem_format)
    codes = np.flatnonzero(np.isfinite(t))
    order = np.argsort(t[codes], kind="stable")
    return t[codes][order], codes[order].astype(np.uint8)


def quantize(w, elem_format, axis=-1, block=BLOCK):
    """Fake-quantize a tensor to an MX format. Returns the dequantized tensor."""
    return _quantize_mx(w, SCALE_BITS, elem_format, axes=[axis], block_size=block, round="nearest")


def quantize_full(w, elem_format, axis=-1, block=BLOCK):
    """Quantize and also return per-block E8M0 exponents and per-element integer codes."""
    w = w.float()
    deq = quantize(w, elem_format, axis, block)
    _, _, emax, _, _ = _get_format_params(elem_format)

    blocked, axes, orig_shape, padded_shape = _reshape_to_blocks(w, [axis], block)
    sax = axes[0] + 1
    shared_exp = _shared_exponents(blocked, method="max", axes=[sax], ebits=0) - emax
    lim = 2 ** (SCALE_BITS - 1) - 1
    shared_exp = shared_exp.clamp(-lim, lim)

    dq_blocked, _, _, _ = _reshape_to_blocks(deq, [axis], block)
    elems = dq_blocked / (2.0 ** shared_exp)

    levels, codes_of_level = sorted_levels(elem_format)
    lv = torch.tensor(levels, dtype=torch.float32, device=w.device)
    idx = torch.searchsorted(lv, elems.contiguous().flatten().clamp(lv[0], lv[-1]))
    idx = idx.clamp(1, len(lv) - 1)
    lo, hi = lv[idx - 1], lv[idx]
    pick = torch.where((elems.flatten() - lo).abs() <= (hi - elems.flatten()).abs(), idx - 1, idx)
    codes = torch.from_numpy(codes_of_level.astype(np.int16)).to(w.device)[pick].reshape(dq_blocked.shape)

    return deq, shared_exp.squeeze(sax).to(torch.int16), codes.to(torch.uint8), (axes, orig_shape, padded_shape)


def dequantize_codes(codes, shared_exp, elem_format, meta):
    """Inverse of quantize_full, used to check round trips and to rebuild edited tensors."""
    axes, orig_shape, padded_shape = meta
    t = torch.tensor(level_table(elem_format), dtype=torch.float32, device=codes.device)
    vals = t[codes.long()] * (2.0 ** shared_exp.float().to(codes.device).unsqueeze(axes[0] + 1))
    return _undo_reshape_to_blocks(vals, padded_shape, orig_shape, axes)


def bits_per_value(elem_format, block=BLOCK):
    """Storage cost of one quantized value including its share of the block scale."""
    return ELEM_BITS[elem_format] + SCALE_BITS / block


def block_max_ratio(w, elem_format, axis=-1, block=BLOCK):
    """Per block max|x|/scale, values above max_norm mean the block max is saturated."""
    _, _, emax, max_norm, _ = _get_format_params(elem_format)
    blocked, axes, _, _ = _reshape_to_blocks(w.float(), [axis], block)
    sax = axes[0] + 1
    shared_exp = (_shared_exponents(blocked, method="max", axes=[sax], ebits=0) - emax).clamp(-127, 127)
    return blocked.abs() / (2.0 ** shared_exp), max_norm


def block_shape_stats(w, axis=-1, block=BLOCK):
    """Kurtosis measured inside each block, averaged over blocks, and the block max to rms ratio."""
    blocked, _, _, _ = _reshape_to_blocks(w.float(), [axis], block)
    m2 = blocked.pow(2).mean(-1)
    m4 = blocked.pow(4).mean(-1)
    ok = m2 > 0
    kurt = torch.where(ok, m4 / m2.clamp_min(1e-30) ** 2, torch.full_like(m2, float("nan")))
    crest = torch.where(ok, blocked.abs().amax(-1) / m2.clamp_min(1e-30).sqrt(), torch.full_like(m2, float("nan")))
    return kurt, crest


def quantize_search(w, elem_format, axis=-1, block=BLOCK, offsets=(0, 1), act_norm=None, ref=None):
    """Pick the block exponent that minimises weighted block error instead of using the floor rule."""
    from mx.elemwise_ops import _quantize_elemwise_core
    ebits, mbits, emax, max_norm, _ = _get_format_params(elem_format)
    blocked, axes, orig_shape, padded_shape = _reshape_to_blocks(w.float(), [axis], block)
    sax = axes[0] + 1
    base = (_shared_exponents(blocked, method="max", axes=[sax], ebits=0) - emax).clamp(-127, 127)

    if act_norm is not None:
        a = act_norm.float().to(w.device)
        pad = padded_shape[axes[0]] - a.numel()
        if pad > 0:
            a = torch.cat([a, torch.zeros(pad, device=a.device)])
        wgt = a.reshape(1, -1, block).unsqueeze(0)[0] if blocked.dim() == 3 else a.reshape(*blocked.shape[1:])
        wgt = wgt.expand_as(blocked) ** 2
    else:
        wgt = None

    best_exp, best_err, best_q = None, None, None
    for off in offsets:
        e = (base + off).clamp(-127, 127)
        q = _quantize_elemwise_core(blocked / (2.0 ** e), mbits, ebits, max_norm,
                                    round="nearest", allow_denorm=True, saturate_normals=True)
        q = q * (2.0 ** e)
        d = (blocked - q) ** 2
        if wgt is not None:
            d = d * wgt
        err = d.sum(-1, keepdim=True)
        if best_err is None:
            best_exp, best_err, best_q = e, err, q
        else:
            take = err < best_err
            best_exp = torch.where(take, e, best_exp)
            best_q = torch.where(take, q, best_q)
            best_err = torch.where(take, err, best_err)

    deq = _undo_reshape_to_blocks(best_q, padded_shape, orig_shape, axes)
    return deq, best_exp.squeeze(sax).to(torch.int16), (axes, orig_shape, padded_shape)


def block_exponent(x, elem_format):
    """Shared exponent for one block laid out as rows by block size."""
    _, _, emax, _, _ = _get_format_params(elem_format)
    mx = x.abs().amax(dim=-1)
    e = torch.floor(torch.log2(mx + (mx == 0).float() * 1e-30)) - emax
    return e.clamp(-127, 127)


def quantize_with_exponent(x, elem_format, exp):
    """Quantize onto the format grid using an explicit per row exponent."""
    from mx.elemwise_ops import _quantize_elemwise_core
    ebits, mbits, _, max_norm, _ = _get_format_params(elem_format)
    s = 2.0 ** exp
    q = _quantize_elemwise_core(x / s, mbits, ebits, max_norm, round="nearest",
                                allow_denorm=True, saturate_normals=True)
    return q * s
