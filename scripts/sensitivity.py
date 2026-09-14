#!/usr/bin/env python
# per tensor cost and activation weighted error for each format and coarsening strength
import argparse, os, sys, time
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from modelio import local_path, iter_tensors, classify, layer_of
from mxfmt import quantize_full, dequantize_codes, FORMATS
from compress import compress_tensor


def weighted_err(w, rec, a):
    """Sum over the tensor of (dw * activation norm)^2, a proxy for output perturbation."""
    d = (w - rec).float()
    if a is not None:
        d = d * a.view(1, -1)
    return float(d.pow(2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--alphas", default="0,0.5,2,5")
    ap.add_argument("--calib", default="results/calib")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    tag = args.model.split("/")[-1]
    path = local_path(args.model)
    if os.path.exists(f"{args.out}/sensitivity_{tag}.csv"):
        os.remove(f"{args.out}/sensitivity_{tag}.csv")
    norms = torch.load(f"{args.calib}/{tag}.pt")
    key = {k + ".weight": v for k, v in norms.items()}
    alphas = [float(x) for x in args.alphas.split(",")]

    out_csv = f"{args.out}/sensitivity_{tag}.csv"
    os.makedirs(args.out, exist_ok=True)
    rows, t0, wrote_header = [], time.time(), False
    for name, w in iter_tensors(path):
        if classify(name) in ("embed_tokens", "lm_head"):
            continue  # not quantized in the eval path, handled in the memory report
        a = key.get(name)
        w = w.float()
        for fmt, ef in [("mxfp4", FORMATS["mxfp4"]), ("mxfp8", FORMATS["mxfp8"])]:
            deq, se, codes, meta = quantize_full(w, ef)
            ref = (w - deq).pow(2).mean().item()
            base_werr = weighted_err(w, deq, a)
            for al in alphas:
                r = compress_tensor(codes, se, ef, act_norm=a, alpha=al, ref_mse=ref)
                rec = dequantize_codes(r["codes"], se, ef, meta) if al > 0 else deq
                rows.append(dict(model=tag, tensor=name, role=classify(name), layer=layer_of(name),
                                 fmt=fmt, alpha=al, n_params=w.numel(),
                                 bits_per_value=r["bits_per_value"], bits_codes=r["bits_codes"],
                                 bits_scales=r["bits_scales"], bytes=r["bytes_codes"] + r["bytes_scales"],
                                 changed_frac=r["changed_frac"], has_calib=a is not None,
                                 werr=weighted_err(w, rec, a), werr_mxonly=base_werr,
                                 mse=(w - rec).pow(2).mean().item(), mse_mxonly=ref))
            del deq, codes, se
        pd.DataFrame(rows[-2 * len(alphas):]).to_csv(out_csv, mode="a", header=not wrote_header, index=False)
        wrote_header = True
        print(f"{time.time()-t0:7.1f}s {name}", flush=True)

    print(f"wrote {out_csv}  {len(rows)} rows")


if __name__ == "__main__":
    main()
