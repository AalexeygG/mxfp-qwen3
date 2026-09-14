#!/usr/bin/env python
# pick a format and coarsening strength per tensor under a global bit budget
import argparse, json, os, sys
import numpy as np
import pandas as pd


def frontier(df, thetas):
    """Lagrangian sweep, each theta gives the plan that minimises werr + theta * bytes."""
    out = []
    for th in thetas:
        d = df.assign(cost=df.werr + th * df.bytes)
        pick = d.loc[d.groupby("tensor").cost.idxmin()]
        out.append((th, pick))
    return out


def summarise(pick, total_params):
    bits = pick.bytes.sum() * 8 / total_params
    return dict(bits_per_value=bits, werr=float(pick.werr.sum()), bytes=int(pick.bytes.sum()),
                n_fp8=int((pick.fmt == "mxfp8").sum()), n_fp4=int((pick.fmt == "mxfp4").sum()),
                params_fp8=int(pick.loc[pick.fmt == "mxfp8", "n_params"].sum()))


def uniform(df, fmt, alpha, total_params):
    p = df[(df.fmt == fmt) & (df.alpha == alpha)]
    return summarise(p, total_params), p


def size_greedy(df, budget_bytes, total_params):
    """Naive mixed baseline, upgrade tensors to fp8 from smallest to largest until the budget runs out."""
    f4 = df[(df.fmt == "mxfp4") & (df.alpha == 0)].set_index("tensor")
    f8 = df[(df.fmt == "mxfp8") & (df.alpha == 0)].set_index("tensor")
    cur = f4.bytes.sum()
    chosen = {t: "mxfp4" for t in f4.index}
    for t in f4.sort_values("n_params").index:
        extra = f8.loc[t, "bytes"] - f4.loc[t, "bytes"]
        if cur + extra <= budget_bytes:
            chosen[t] = "mxfp8"
            cur += extra
    rows = [(f8 if v == "mxfp8" else f4).loc[t] for t, v in chosen.items()]
    p = pd.DataFrame(rows).reset_index()
    return summarise(p, total_params), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--budgets", default="4.5,5.0,6.0")
    ap.add_argument("--dir", default="results")
    args = ap.parse_args()

    tag = args.model.split("/")[-1]
    df = pd.read_csv(f"{args.dir}/sensitivity_{tag}.csv")
    df = df[~df.role.isin(["embed_tokens", "lm_head"])]
    total_params = int(df[(df.fmt == "mxfp4") & (df.alpha == 0)].n_params.sum())

    ref = {}
    for fmt in ["mxfp4", "mxfp8"]:
        s, _ = uniform(df, fmt, 0.0, total_params)
        ref[f"uniform_{fmt}"] = s
        print(f"uniform {fmt}: {s['bits_per_value']:.3f} b/v  werr {s['werr']:.4g}")

    thetas = np.logspace(-14, -2, 240)
    fr = frontier(df, thetas)
    curve = [summarise(p, total_params) | {"theta": th} for th, p in fr]
    pd.DataFrame(curve).to_csv(f"{args.dir}/frontier_{tag}.csv", index=False)

    plans = {}
    for b in [float(x) for x in args.budgets.split(",")]:
        ok = [(s, p) for s, p in zip(curve, [p for _, p in fr]) if s["bits_per_value"] <= b]
        if not ok:
            print(f"budget {b}: unreachable")
            continue
        s, p = min(ok, key=lambda x: x[0]["werr"])
        nb, _ = size_greedy(df, b * total_params / 8, total_params)
        gain = nb["werr"] / s["werr"]
        print(f"budget {b:.2f}: ours {s['bits_per_value']:.3f} b/v werr {s['werr']:.4g}  "
              f"| size-greedy {nb['bits_per_value']:.3f} b/v werr {nb['werr']:.4g}  | {gain:.2f}x less error")
        plans[str(b)] = dict(summary=s, baseline=nb,
                             plan={r.tensor: dict(fmt=r.fmt, alpha=float(r.alpha)) for r in p.itertuples()})

    with open(f"{args.dir}/plans_{tag}.json", "w") as f:
        json.dump(dict(model=args.model, total_params=total_params, reference=ref, budgets=plans), f, indent=2)
    print(f"wrote {args.dir}/plans_{tag}.json")


if __name__ == "__main__":
    main()
