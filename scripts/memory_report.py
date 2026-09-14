#!/usr/bin/env python
# memory footprint of bf16 vs MX formats, from measured per tensor sizes
import argparse, glob, os, sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from mxfmt import bits_per_value, FORMATS, BLOCK, SCALE_BITS, ELEM_BITS

MB = 1024 ** 2


def report(stats_csv, skipped_csv):
    d = pd.read_csv(stats_csv)
    model = d.model.iloc[0]
    kept = pd.read_csv(skipped_csv) if os.path.exists(skipped_csv) else pd.DataFrame(columns=["n_params"])
    n_1d = int(kept.n_params.sum()) if len(kept) else 0

    rows = []
    base = d[d.fmt == d.fmt.iloc[0]]
    n2d = int(base.n_params.sum())
    bf16_bytes = n2d * 2 + n_1d * 2
    rows.append(dict(model=model, fmt="bf16", bits_per_value=16.0, quantized_params=0,
                     bf16_params=n2d + n_1d, bytes=bf16_bytes, mib=bf16_bytes / MB, ratio=1.0))

    for fmt, g in d.groupby("fmt"):
        b = int(g.bytes_mx.sum()) + n_1d * 2
        rows.append(dict(model=model, fmt=fmt, bits_per_value=bits_per_value(FORMATS[fmt]),
                         quantized_params=n2d, bf16_params=n_1d, bytes=b, mib=b / MB,
                         ratio=bf16_bytes / b))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    args = ap.parse_args()

    out = []
    for f in sorted([f for f in glob.glob(f"{args.dir}/stats_*.csv") if "summary" not in f]):
        tag = os.path.basename(f)[len("stats_"):-4]
        out.append(report(f, f"{args.dir}/skipped_{tag}.csv"))
    df = pd.concat(out, ignore_index=True)
    df.to_csv(f"{args.dir}/memory.csv", index=False)

    print(f"block size {BLOCK}, scale {SCALE_BITS} bits shared per block")
    for f, ef in FORMATS.items():
        if f in df.fmt.values:
            print(f"  {f:7s} {ELEM_BITS[ef]} + {SCALE_BITS}/{BLOCK} = {bits_per_value(ef)} bits per value, "
                  f"{16/bits_per_value(ef):.3f}x vs bf16 on quantized tensors")
    print()
    show = df.copy()
    show["MiB"] = show.mib.round(1)
    show["GiB"] = (show.mib / 1024).round(3)
    print(show[["model", "fmt", "bits_per_value", "bytes", "MiB", "GiB", "ratio"]].to_string(index=False))
    print(f"\nwrote {args.dir}/memory.csv")


if __name__ == "__main__":
    main()
