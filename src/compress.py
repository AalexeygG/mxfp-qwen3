# rate distortion coarsening of MX codes plus entropy coding of codes and scales
import io
import os

import numpy as np
import torch
import zstandard as zstd

from mxfmt import sorted_levels, level_table, ELEM_BITS, SCALE_BITS, BLOCK

ZLEVEL = int(os.environ.get("MX_ZSTD_LEVEL", 12))


def pack_codes(codes, bits):
    """Pack integer codes into bytes, two per byte for 4 bit formats."""
    a = codes.astype(np.uint8).ravel()
    if bits == 4:
        if a.size % 2:
            a = np.append(a, 0)
        return ((a[0::2] << 4) | a[1::2]).tobytes()
    return a.tobytes()


def zcompress(b, level=ZLEVEL):
    return zstd.ZstdCompressor(level=level).compress(b)


def zdecompress(b, size):
    return zstd.ZstdDecompressor().decompress(b, max_output_size=size)


def unpack_codes(raw, n, bits):
    """Inverse of pack_codes."""
    a = np.frombuffer(raw, dtype=np.uint8)
    if bits == 4:
        out = np.empty(a.size * 2, dtype=np.uint8)
        out[0::2] = a >> 4
        out[1::2] = a & 0x0F
        return out[:n]
    return a[:n].copy()


def entropy(counts):
    p = counts[counts > 0].astype(np.float64)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def encode_scales(shared_exp):
    """Delta code E8M0 exponents along the block axis, then zstd."""
    e = shared_exp.numpy().astype(np.int16)
    d = np.diff(e, axis=-1, prepend=np.zeros_like(e[..., :1]))
    z = ((d << 1) ^ (d >> 15)).astype(np.uint16)  # zigzag keeps small deltas small
    narrow = z.max() < 256
    raw = z.astype(np.uint8).tobytes() if narrow else z.tobytes()
    return zcompress(raw), e.size, (narrow, e.shape)


def decode_scales(blob, meta):
    """Inverse of encode_scales."""
    narrow, shape = meta
    n = int(np.prod(shape))
    raw = zdecompress(blob, n * (1 if narrow else 2))
    z = np.frombuffer(raw, dtype=np.uint8 if narrow else np.uint16).astype(np.uint16)
    d = ((z >> 1).astype(np.int16) ^ -(z & 1).astype(np.int16)).reshape(shape)
    return torch.from_numpy(np.cumsum(d, axis=-1).astype(np.int16))


def rank_of_code(codes, elem_format):
    """Map format bit codes to indices in the value sorted level list."""
    _, codes_of_level = sorted_levels(elem_format)
    lut = np.zeros(256, dtype=np.uint8)
    lut[codes_of_level] = np.arange(len(codes_of_level), dtype=np.uint8)
    return lut[codes]


def code_of_rank(ranks, elem_format):
    _, codes_of_level = sorted_levels(elem_format)
    return codes_of_level[ranks]


def rd_coarsen(ranks, levels, w, _unused, lam, protect, rate):
    """Move each code at most one level if the rate gain outweighs the weighted distortion."""
    n_lv = len(levels)
    best = ranks.copy()
    cur_cost = lam * rate[ranks]
    for off in (-1, 1):
        cand = ranks.astype(np.int16) + off
        ok = (cand >= 0) & (cand < n_lv) & (~protect)
        cand_c = np.clip(cand, 0, n_lv - 1)
        d = (levels[cand_c] - levels[ranks]) * w
        cost = d * d + lam * rate[cand_c]
        take = ok & (cost < cur_cost)
        best = np.where(take, cand_c, best).astype(np.uint8)
        cur_cost = np.where(take, cost, cur_cost)
    return best


def _importance(shape, act_norm, block):
    """Broadcast per input channel activation norms onto the blocked tensor shape."""
    if act_norm is None:
        return None
    a = act_norm.numpy().astype(np.float32)
    pad = shape[-2] * block - len(a)
    if pad > 0:
        a = np.concatenate([a, np.zeros(pad, dtype=np.float32)])
    return a.reshape(shape[-2], block)


def compress_tensor(codes, shared_exp, elem_format, act_norm=None, alpha=0.0, ref_mse=None,
                    protect_top=0.01, block=BLOCK, chunk_elems=8_000_000):
    """Entropy code one quantized tensor, optionally coarsening the unimportant part first."""
    bits = ELEM_BITS[elem_format]
    levels, codes_of_level = sorted_levels(elem_format)
    levels = levels.astype(np.float32)
    ranks = rank_of_code(codes.numpy(), elem_format)
    shape = codes.shape
    n = int(np.prod(shape))
    exps = shared_exp.numpy().astype(np.float32)
    imp2d = _importance(shape, act_norm, block)

    lam = 0.0 if alpha <= 0 else alpha * float(ref_mse)
    hist = np.bincount(ranks.ravel(), minlength=len(levels)).astype(np.float64)
    rate = -np.log2(np.maximum(hist / hist.sum(), 1e-12))

    thr = np.inf
    if lam > 0 and protect_top > 0:
        sel = np.random.default_rng(0).integers(0, shape[0], size=min(shape[0], 4096))
        sv = np.abs(levels[ranks[sel]]) * (2.0 ** exps[sel])[..., None]
        if imp2d is not None:
            sv = sv * imp2d
        thr = float(np.quantile(sv, 1 - protect_top))
        del sv

    rows_per_chunk = max(1, chunk_elems // max(1, n // shape[0]))
    buf = io.BytesIO()
    writer = zstd.ZstdCompressor(level=ZLEVEL).stream_writer(buf)
    changed, sq_dw, new_hist = 0, 0.0, np.zeros(len(levels), dtype=np.int64)
    out_chunks = []

    for i in range(0, shape[0], rows_per_chunk):
        r = ranks[i:i + rows_per_chunk]
        sc = (2.0 ** exps[i:i + rows_per_chunk])[..., None]
        w = sc if imp2d is None else sc * imp2d
        if lam > 0:
            val = np.abs(levels[r]) * w
            nr = rd_coarsen(r, levels, w, np.ones(1, dtype=np.float32), lam, val >= thr, rate)
            del val
        else:
            nr = r
        d = (levels[nr] - levels[r]) * sc
        sq_dw += float((d.astype(np.float64) ** 2).sum())
        changed += int((nr != r).sum())
        new_hist += np.bincount(nr.ravel(), minlength=len(levels))
        oc = codes_of_level[nr]
        out_chunks.append(oc)
        writer.write(pack_codes(oc, bits))
        del r, sc, w, d, nr

    writer.flush(zstd.FLUSH_FRAME)
    cbytes = buf.tell()
    sbytes, n_blocks, smeta = encode_scales(shared_exp)

    return dict(
        codes=torch.from_numpy(np.concatenate(out_chunks, axis=0)),
        blob_codes=buf.getvalue(), blob_scales=sbytes, scale_meta=smeta, elem_format=elem_format,
        shape=tuple(shape),
        bytes_codes=cbytes, bytes_scales=len(sbytes), n=n, n_blocks=n_blocks,
        bits_per_value=(cbytes + len(sbytes)) * 8 / n,
        bits_codes=cbytes * 8 / n, bits_scales=len(sbytes) * 8 / n,
        code_entropy=entropy(new_hist), changed_frac=changed / n, alpha=alpha, lam=lam,
        dw_rms=float(np.sqrt(sq_dw / n)),
    )


def decompress_tensor(r):
    """Rebuild codes and exponents from the compressed blobs of compress_tensor."""
    bits = ELEM_BITS[r["elem_format"]]
    shape = r["shape"]
    n = int(np.prod(shape))
    raw = zdecompress(r["blob_codes"], (n + 1) // 2 if bits == 4 else n)
    codes = unpack_codes(raw, n, bits).reshape(shape)
    return torch.from_numpy(codes), decode_scales(r["blob_scales"], r["scale_meta"])
