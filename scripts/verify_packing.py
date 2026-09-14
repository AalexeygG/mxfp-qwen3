#!/usr/bin/env python
# pack a quantized model to disk and compare the real file size against the formula
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from modelio import local_path, iter_tensors, classify
from mxfmt import quantize_full, dequantize_codes, FORMATS, ELEM_BITS, SCALE_BITS, BLOCK, bits_per_value
from compress import pack_codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--fmt", default="mxfp4")
    ap.add_argument("--out", default=None)
    ap.add_argument("--check-roundtrip", action="store_true")
    args = ap.parse_args()

    tag = args.model.split("/")[-1]
    ef = FORMATS[args.fmt]
    bits = ELEM_BITS[ef]
    out = args.out or f"/tmp/{tag}_{args.fmt}.bin"
    path = local_path(args.model)

    tied = json.load(open(os.path.join(path, "config.json"))).get("tie_word_embeddings", False)
    n_total, n_blocks, worst = 0, 0, 0.0
    t0 = time.time()
    with open(out, "wb") as f:
        for name, w in iter_tensors(path):
            if tied and classify(name) == "lm_head":
                continue
            for i in range(0, w.shape[0], 2048):
                c = w[i:i + 2048].float()
                deq, se, codes, meta = quantize_full(c, ef)
                f.write(pack_codes(codes.numpy(), bits))
                f.write(se.numpy().astype(np.int8).tobytes())
                n_total += c.numel()
                n_blocks += se.numel()
                if args.check_roundtrip:
                    rt = dequantize_codes(codes, se, ef, meta)
                    worst = max(worst, float((rt - deq).abs().max()))
                del c, deq, se, codes
            print(f"  {name}", flush=True)

    real = os.path.getsize(out)
    formula = int(np.ceil(n_total * bits / 8)) + int(np.ceil(n_blocks * SCALE_BITS / 8))
    print(f"\n{tag} {args.fmt}")
    print(f"  values          {n_total}")
    print(f"  blocks          {n_blocks}  ({n_total / n_blocks:.2f} values per block)")
    print(f"  file on disk    {real} bytes")
    print(f"  formula         {formula} bytes  ({bits} + {SCALE_BITS}/{BLOCK} = {bits_per_value(ef)} bits per value)")
    print(f"  difference      {real - formula} bytes")
    print(f"  measured        {real * 8 / n_total:.4f} bits per value")
    if args.check_roundtrip:
        print(f"  max round trip error {worst}")
    print(f"  {time.time()-t0:.0f}s, file kept at {out}")


if __name__ == "__main__":
    main()
