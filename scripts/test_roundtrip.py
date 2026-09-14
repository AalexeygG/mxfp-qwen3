#!/usr/bin/env python
# compress every tensor, decompress it, and check the model comes back bit for bit
import argparse, os, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from modelio import local_path, iter_tensors, classify
from mxfmt import quantize_full, dequantize_codes, FORMATS
from compress import compress_tensor, decompress_tensor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--formats", default="mxfp4,mxfp8")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many tensors")
    ap.add_argument("--chunk-rows", type=int, default=2048)
    args = ap.parse_args()

    path = local_path(args.model)
    t0, bad, n_t, n_val, raw, comp = time.time(), 0, 0, 0, 0, 0

    for name, w in iter_tensors(path):
        if classify(name) == "lm_head":
            continue
        for fmt in args.formats.split(","):
            ef = FORMATS[fmt]
            for i in range(0, w.shape[0], args.chunk_rows):
                c = w[i:i + args.chunk_rows].float()
                deq, se, codes, meta = quantize_full(c, ef)
                r = compress_tensor(codes, se, ef, alpha=0.0)
                codes2, se2 = decompress_tensor(r)

                if not torch.equal(codes, codes2):
                    print(f"  codes differ: {name} {fmt} rows {i}")
                    bad += 1
                elif not torch.equal(se, se2):
                    print(f"  exponents differ: {name} {fmt} rows {i}")
                    bad += 1
                else:
                    rebuilt = dequantize_codes(codes2, se2, ef, meta)
                    if not torch.equal(rebuilt, deq):
                        print(f"  tensor differs: {name} {fmt} rows {i}")
                        bad += 1
                n_val += c.numel()
                raw += c.numel() * 2
                comp += r["bytes_codes"] + r["bytes_scales"]
                del c, deq, se, codes, r, codes2, se2
            n_t += 1
        if args.limit and n_t >= args.limit * len(args.formats.split(",")):
            break

    print(f"\n{args.model}")
    print(f"  tensors checked   {n_t}")
    print(f"  values            {n_val:,}")
    print(f"  mismatches        {bad}")
    print(f"  bf16 bytes        {raw:,}")
    print(f"  compressed bytes  {comp:,}")
    print(f"  ratio             {raw / comp:.3f}x")
    print(f"  bits per value    {comp * 8 / n_val:.3f}")
    print(f"  {time.time()-t0:.0f}s")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
