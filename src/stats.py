# per tensor quantization statistics, accumulated in row chunks
import numpy as np
import torch

from mxfmt import quantize_full, level_table, bits_per_value, block_max_ratio, block_shape_stats, BLOCK, SCALE_BITS, ELEM_BITS


def _entropy(counts):
    p = counts[counts > 0].astype(np.float64)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def tensor_stats(w, elem_format, chunk_rows=4096):
    """Quantize a 2D weight in row chunks and return error, distribution and size stats."""
    n = w.numel()
    tab = level_table(elem_format)
    n_codes = len(tab)
    max_norm = np.nanmax(tab)

    acc = dict(sw2=0.0, se2=0.0, sw4=0.0, maxerr=0.0, n=0, sat=0, satblk=0, nblk=0)
    code_hist = np.zeros(n_codes, dtype=np.int64)
    exp_hist = {}
    n_blocks = 0

    for i in range(0, w.shape[0], chunk_rows):
        c = w[i:i + chunk_rows].float()
        deq, se, codes, _ = quantize_full(c, elem_format)
        d = (c - deq)
        acc["sw2"] += float(c.pow(2).sum())
        acc["se2"] += float(d.pow(2).sum())
        acc["sw4"] += float(c.pow(4).sum())
        acc["maxerr"] = max(acc["maxerr"], float(d.abs().max()))
        acc["n"] += c.numel()
        code_hist += np.bincount(codes.flatten().numpy(), minlength=n_codes)
        for v, cnt in zip(*np.unique(se.numpy(), return_counts=True)):
            exp_hist[int(v)] = exp_hist.get(int(v), 0) + int(cnt)
        n_blocks += se.numel()
        r, mn = block_max_ratio(c, elem_format)
        acc["sat"] += int((r > mn).sum())
        acc["satblk"] += int((r.amax(-1) > mn).sum())
        acc["nblk"] += int(r[..., 0].numel())
        bk, cr = block_shape_stats(c)
        acc["bkurt"] = acc.get("bkurt", 0.0) + float(torch.nansum(bk))
        acc["crest"] = acc.get("crest", 0.0) + float(torch.nansum(cr))
        del c, deq, d, codes, se, r, bk, cr

    vals = tab[np.arange(n_codes)]
    finite = np.isfinite(vals)
    zero_n = int(code_hist[finite & (vals == 0)].sum())
    clip_n = int(code_hist[finite & (np.abs(vals) == max_norm)].sum())

    exp_counts = np.array(list(exp_hist.values()), dtype=np.int64)
    exp_vals = np.array(list(exp_hist.keys()), dtype=np.int64)

    mse = acc["se2"] / acc["n"]
    msw = acc["sw2"] / acc["n"]
    kurt = (acc["sw4"] / acc["n"]) / (msw ** 2) if msw > 0 else float("nan")

    bpv = bits_per_value(elem_format)
    return dict(
        n_params=n,
        rows=w.shape[0], cols=w.shape[1],
        sqnr_db=10 * np.log10(msw / mse) if mse > 0 else float("inf"),
        nmse=mse / msw if msw > 0 else float("nan"),
        max_abs_err=acc["maxerr"],
        rms_weight=float(np.sqrt(msw)),
        kurtosis=kurt,
        block_kurtosis=acc.get("bkurt", 0.0) / max(acc["nblk"], 1),
        block_crest=acc.get("crest", 0.0) / max(acc["nblk"], 1),
        code_entropy=_entropy(code_hist),
        code_entropy_ratio=_entropy(code_hist) / ELEM_BITS[elem_format],
        scale_entropy=_entropy(exp_counts),
        scale_span=int(exp_vals.max() - exp_vals.min()) if len(exp_vals) else 0,
        zero_rate=zero_n / acc["n"],
        top_level_rate=clip_n / acc["n"],
        sat_rate=acc["sat"] / acc["n"],
        sat_block_rate=acc["satblk"] / max(acc["nblk"], 1),
        n_blocks=n_blocks,
        bits_per_value=bpv,
        bytes_bf16=n * 2,
        bytes_mx=int(np.ceil(n * bpv / 8)),
    )
