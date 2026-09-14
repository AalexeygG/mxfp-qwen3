#!/usr/bin/env python
# build the summary tables and figures from everything under results/
import glob, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

RES = "results"
FIG = "results/figures"
ROLES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens"]
C4, C8 = "#1f77b4", "#d62728"


def load_stats():
    fs = sorted(glob.glob(f"{RES}/stats_*.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def model_order(d):
    """Models sorted by parameter count, not alphabetically."""
    n = d[d.fmt == d.fmt.iloc[0]].groupby("model").n_params.sum()
    return list(n.sort_values().index)


def fig_sqnr_uniform(d):
    models = model_order(d)
    cols = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    for ax, fmt in zip(axes, ["mxfp4", "mxfp8"]):
        s = d[d.fmt == fmt]
        names = [r for r in ROLES if (s.role == r).any()]
        for k, m in enumerate(models):
            g = s[s.model == m]
            xs, ys = [], []
            for i, r in enumerate(names):
                v = g[g.role == r].sqnr_db.values
                xs += list(np.full(len(v), i + (k - 1.5) * 0.17))
                ys += list(v)
            ax.scatter(xs, ys, s=13, alpha=.75, color=cols[k], label=m, edgecolors="none")
        lo, hi = s.sqnr_db.min(), s.sqnr_db.max()
        pad = max(0.05, (hi - lo) * 0.15)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylabel("SQNR, dB")
        ax.set_title(f"{fmt}, full range {hi - lo:.2f} dB across {len(s)} tensors", fontsize=11)
        ax.grid(alpha=.3, axis="y")
    axes[0].legend(fontsize=8, ncol=4, loc="upper center")
    fig.suptitle("Quantization error is set by the format, not by the layer or the model size")
    fig.tight_layout()
    fig.savefig(f"{FIG}/sqnr_by_role.png", dpi=140)
    plt.close(fig)


def fig_kurtosis(d):
    s = d[d.fmt == "mxfp4"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, lab in [(axes[0], "kurtosis", "kurtosis of the whole tensor"),
                         (axes[1], "block_kurtosis", "kurtosis inside a 32 value block")]:
        for m, mk in zip(model_order(s), ["o", "s", "^", "v"]):
            g = s[s.model == m]
            ax.scatter(g[col], g.sqnr_db, s=14, alpha=.6, marker=mk, label=m)
        r = np.corrcoef(s[col], s.sqnr_db)[0, 1]
        ax.set_xlabel(lab)
        ax.set_ylabel("SQNR, dB")
        ax.set_title(f"r = {r:+.2f}")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    axes[0].set_xscale("log")
    fig.suptitle("The usual outlier statistic does not predict MX error, the within block one does")
    fig.tight_layout()
    fig.savefig(f"{FIG}/kurtosis_vs_sqnr.png", dpi=140)
    plt.close(fig)


def fig_memory():
    f = f"{RES}/memory.csv"
    if not os.path.exists(f):
        return
    d = pd.read_csv(f)
    models = list(d[d.fmt == "bf16"].sort_values("bytes").model)
    fig, ax = plt.subplots(figsize=(1.9 * len(models) + 3, 4))
    w, fmts = 0.26, ["bf16", "mxfp8", "mxfp4"]
    for i, fmt in enumerate(fmts):
        g = d[d.fmt == fmt].set_index("model").reindex(models)
        x = np.arange(len(models)) + (i - 1) * w
        ax.bar(x, g.mib / 1024, w, label=fmt, color=["#888", C8, C4][i])
        for xi, v in zip(x, g.mib / 1024):
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylabel("weights, GiB")
    ax.legend()
    ax.grid(alpha=.3, axis="y")
    ax.set_title("Measured weight footprint")
    fig.tight_layout()
    fig.savefig(f"{FIG}/memory.png", dpi=140)
    plt.close(fig)


def fig_frontier():
    for f in sorted(glob.glob(f"{RES}/frontier_*.csv")):
        tag = os.path.basename(f)[len("frontier_"):-4]
        d = pd.read_csv(f).sort_values("bits_per_value")
        pj = f"{RES}/plans_{tag}.json"
        if not os.path.exists(pj):
            continue
        p = json.load(open(pj))
        fig, ax = plt.subplots(figsize=(6.5, 4.4))
        ax.plot(d.bits_per_value, d.werr, "-", color=C4, label="ours, sensitivity driven")
        for k, s in p["reference"].items():
            ax.scatter([s["bits_per_value"]], [s["werr"]], marker="*", s=170, zorder=5,
                       color=C8 if "fp8" in k else "#444", label=k.replace("_", " "))
        bl = [(v["baseline"]["bits_per_value"], v["baseline"]["werr"]) for v in p["budgets"].values()]
        if bl:
            bl = np.array(sorted(bl))
            ax.plot(bl[:, 0], bl[:, 1], "--o", color="#999", ms=4, label="naive mixed baseline")
        ax.set_yscale("log")
        ax.set_xlabel("bits per value, measured after entropy coding")
        ax.set_ylabel("activation weighted squared error")
        ax.set_title(f"Rate distortion, {tag}")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{FIG}/frontier_{tag}.png", dpi=140)
        plt.close(fig)


BITS = {"bf16": 16.0, "mxfp8": 8.25, "mxfp4": 4.25}


def ppl_table():
    rows = []
    for f in sorted(glob.glob(f"{RES}/ppl_*_kaggle.csv")):
        d = pd.read_csv(f)
        base = {"none": 16.0, "mxfp8": 8.25, "mxfp4": 4.25}
        for r in d.itertuples():
            rows.append(dict(run=f"{r.model}_{r.config}", model=r.model, weights=r.weights,
                             acts=r.acts, errcomp="mxfp4" if "compensation" in str(r.method) else None,
                             search="search" in str(r.method), budget=None,
                             bits=base.get(r.weights, 16.0), ppl=r.ppl, seconds=r.seconds))
    for f in sorted(glob.glob(f"{RES}/ppl/*.json")):
        r = json.load(open(f))
        name = os.path.basename(f)[:-5]
        w = r["weights"] if r["weights"] != "none" else ("bf16" if not r.get("errcomp") else r["errcomp"])
        bits = r.get("bits_per_value") or BITS.get(w, 16.0)
        rows.append(dict(run=name, model=r["model"].split("/")[-1], weights=r["weights"],
                         acts=r["acts"], errcomp=r.get("errcomp"), search=r.get("search", False),
                         budget=r.get("budget"), bits=bits, ppl=r["ppl"], seconds=round(r["seconds"])))
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows).sort_values(["model", "ppl"])
    d.to_csv(f"{RES}/ppl_summary.csv", index=False)
    return d


def fig_ppl(d):
    """Perplexity against real storage cost, one panel per model."""
    for model, g in d.groupby("model"):
        base = g[(g.weights == "none") & (g.acts == "none") & g.errcomp.isna()]
        if base.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.6))
        ax.axhline(base.ppl.iloc[0], color="#444", ls=":", lw=1)
        ax.text(4.3, base.ppl.iloc[0], " bf16", va="bottom", fontsize=8, color="#444")
        act_w = g.run.str.contains("search_act")
        groups = [
            (g[g.budget.notna()], "bit allocation, rejected", "x", "#999", 55, 3),
            (g[act_w], "exponent search, activation weighted, rejected", "x", "#999", 55, 3),
            (g[(g.acts != "none") & g.errcomp.isna()], "activations quantized", "^", "#9467bd", 70, 4),
            (g[(g.acts == "none") & g.errcomp.isna() & g.search & ~act_w], "exponent search", "s", "#2ca02c", 62, 5),
            (g[g.errcomp.notna()], "error compensation", "D", C8, 72, 6),
            (g[(g.acts == "none") & g.errcomp.isna() & (~g.search) & (g.weights != "none") & g.budget.isna()],
             "MX as normally applied", "o", C4, 110, 7),
        ]
        seen = set()
        for sub, lab, mk, c, sz, z in groups:
            if len(sub) and lab not in seen:
                face = "none" if mk == "o" else c
                ax.scatter(sub.bits, sub.ppl, marker=mk, s=sz, facecolors=face, edgecolors=c,
                           linewidths=1.8, label=lab, zorder=z)
                seen.add(lab)
        ax.set_xlabel("bits per value of the quantized weights")
        ax.set_ylabel("wikitext2 perplexity")
        ax.set_title(f"{model}, lower and left is better")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{FIG}/ppl_vs_bits_{model}.png", dpi=140)
        plt.close(fig)


def eval_table():
    rows = []
    for f in sorted(glob.glob(f"{RES}/eval_*_kaggle.csv")):
        d = pd.read_csv(f)
        for r in d.itertuples():
            rows.append({"run": f"{r.model}_{r.config}", "model": r.model, "weights": r.weights,
                         "acts": r.acts, "budget": None, "seconds": None,
                         "arc_easy/acc": r.arc_easy_acc, "arc_easy/acc_stderr": r.arc_easy_acc_stderr,
                         "piqa/acc": r.piqa_acc, "piqa/acc_stderr": r.piqa_acc_stderr,
                         "winogrande/acc": r.winogrande_acc,
                         "winogrande/acc_stderr": r.winogrande_acc_stderr,
                         "lambada_openai/perplexity": r.lambada_ppl,
                         "lambada_openai/acc": r.lambada_acc,
                         "lambada_openai/acc_stderr": r.lambada_acc_stderr})
    for f in sorted(glob.glob(f"{RES}/eval/*.json")):
        p = json.load(open(f))
        r = {"run": os.path.basename(f)[:-5], "model": p["model"].split("/")[-1],
             "weights": p["weights"], "acts": p["acts"], "budget": p.get("budget"),
             "seconds": round(p["seconds"])}
        for task, m in p["results"].items():
            for k, v in m.items():
                if k.endswith(",none") and not k.startswith("alias") and isinstance(v, float):
                    r[f"{task}/{k.split(',')[0]}"] = round(v, 4)
        rows.append(r)
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    d.to_csv(f"{RES}/eval_summary.csv", index=False)
    return d


def fig_benchmark(model="Qwen3-4B"):
    """How much of the MXFP4 damage the method gives back, per task."""
    f = f"{RES}/eval_{model}_kaggle.csv"
    if not os.path.exists(f):
        return
    d = pd.read_csv(f).set_index("config")
    need = ["bf16", "w-mxfp4", "errcomp-mxfp4_search"]
    if any(c not in d.index for c in need):
        return
    b, m, o = (d.loc[c] for c in need)
    tasks = [("arc_easy_acc", "arc_easy"), ("piqa_acc", "piqa"),
             ("winogrande_acc", "winogrande"), ("lambada_acc", "lambada acc")]

    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(tasks))
    w = 0.26
    for k, (row, lab, c) in enumerate([(b, "bf16", "#444"), (m, "MXFP4", C4),
                                       (o, "MXFP4 plus our method", C8)]):
        vals = [row[t] for t, _ in tasks]
        errs = [row[t + "_stderr"] for t, _ in tasks]
        ax.bar(x + (k - 1) * w, vals, w, yerr=errs, capsize=3, label=lab,
               color=c, alpha=.85 if k else .6)
    for i, (t, _) in enumerate(tasks):
        gap = b[t] - m[t]
        if abs(gap) > 1e-9:
            ax.text(i + w, o[t] + 0.02, f"{(o[t]-m[t])/gap*100:.0f}%", ha="center", fontsize=9,
                    color=C8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in tasks])
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, max(b[t] for t, _ in tasks) * 1.22)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=.3, axis="y")
    ax.set_title(f"{model}, share of the MXFP4 loss recovered at the same 4.25 bits")
    fig.tight_layout()
    fig.savefig(f"{FIG}/benchmark_{model}.png", dpi=140)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    d = load_stats()
    if len(d):
        fig_sqnr_uniform(d)
        fig_kurtosis(d)
        summary = d.groupby(["model", "fmt"]).agg(
            sqnr_mean=("sqnr_db", "mean"), sqnr_sd=("sqnr_db", "std"),
            sqnr_min=("sqnr_db", "min"), sqnr_max=("sqnr_db", "max"),
            code_entropy=("code_entropy", "mean"), scale_entropy=("scale_entropy", "mean"),
            sat_block=("sat_block_rate", "mean"), params=("n_params", "sum")).round(4)
        summary.to_csv(f"{RES}/stats_summary.csv")
        print(summary.to_string())
    fig_memory()
    fig_frontier()
    p = ppl_table()
    if len(p):
        fig_ppl(p)
        print("\n", p[["run", "bits", "ppl"]].to_string(index=False))
    fig_benchmark()
    e = eval_table()
    if len(e):
        print("\n", e.to_string(index=False))
    print(f"\nfigures in {FIG}")


if __name__ == "__main__":
    main()
