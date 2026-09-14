#!/usr/bin/env python
# per tensor signal power sum (w * activation norm)^2, used to normalise sensitivity
import argparse, os, sys
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from modelio import local_path, iter_tensors, classify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--calib", default="results/calib")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    tag = args.model.split("/")[-1]
    norms = torch.load(f"{args.calib}/{tag}.pt")
    key = {k + ".weight": v for k, v in norms.items()}
    rows = []
    for name, w in iter_tensors(local_path(args.model)):
        if classify(name) in ("embed_tokens", "lm_head"):
            continue
        a = key.get(name)
        w = w.float()
        p = (w * a.view(1, -1)).pow(2).sum().item() if a is not None else w.pow(2).sum().item()
        rows.append(dict(tensor=name, wnorm=p, w2=w.pow(2).sum().item()))
    pd.DataFrame(rows).to_csv(f"{args.out}/signal_{tag}.csv", index=False)
    print(f"wrote {args.out}/signal_{tag}.csv  {len(rows)} tensors")


if __name__ == "__main__":
    main()
