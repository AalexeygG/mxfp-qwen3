#!/usr/bin/env python
# quantize every 2D weight of a checkpoint and dump per tensor stats
import argparse, json, os, sys, time
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from modelio import local_path, iter_tensors, classify, layer_of, tensor_index
from stats import tensor_stats
from mxfmt import FORMATS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--formats", default="mxfp4,mxfp8")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    path = local_path(args.model)
    tag = args.model.split("/")[-1]
    rows = []
    t0 = time.time()

    tied = json.load(open(os.path.join(path, "config.json"))).get("tie_word_embeddings", False)

    for name, w in iter_tensors(path):
        if tied and name == "lm_head.weight":
            print(f"skip {name}, tied to embed_tokens")
            continue
        for fmt in args.formats.split(","):
            s = tensor_stats(w, FORMATS[fmt])
            s.update(model=tag, tensor=name, fmt=fmt, role=classify(name), layer=layer_of(name),
                     dtype=str(w.dtype).replace("torch.", ""))
            rows.append(s)
        print(f"{time.time()-t0:7.1f}s {name} {tuple(w.shape)}", flush=True)

    # 1D tensors stay in bf16, count them so the memory budget is honest
    idx = tensor_index(path)
    skipped = {k: v for k, v in idx.items() if len(v[0]) != 2}
    os.makedirs(args.out, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out}/stats_{tag}.csv", index=False)
    pd.DataFrame([{"tensor": k, "shape": str(v[0]), "dtype": v[1],
                   "n_params": int(v[0][0]) if v[0] else 0} for k, v in skipped.items()]).to_csv(
        f"{args.out}/skipped_{tag}.csv", index=False)
    print(f"wrote {args.out}/stats_{tag}.csv  {len(df)} rows  {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
