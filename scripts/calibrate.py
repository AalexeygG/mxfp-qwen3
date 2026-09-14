#!/usr/bin/env python
# collect per input channel activation norms for every linear, used as importance weights
import argparse, os, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from fakequant import target_linears
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--nsamples", type=int, default=64)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--max-memory", default=None, help="per device cap, e.g. 0=13GiB,1=13GiB,cpu=20GiB")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", default="results/calib")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    dt = getattr(torch, args.dtype)
    kw = dict(device_map=args.device_map) if args.device_map else {}
    if args.max_memory:
        kw["max_memory"] = {(int(k) if k.isdigit() else k): v
                            for k, v in (p.split("=") for p in args.max_memory.split(","))}
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dt, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt, **kw)
    if not args.device_map:
        model = model.to(args.device)
    model = model.eval()
    dev = next(model.parameters()).device if args.device_map else args.device

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]

    body = model.model if hasattr(model, "model") else model  # skip lm_head, logits are not needed
    acc, counts = {}, {}
    hooks = []

    def mk(name):
        def hook(mod, args_):
            x = args_[0].detach().float().reshape(-1, args_[0].shape[-1])
            s = x.pow(2).sum(0).cpu()
            acc[name] = s if name not in acc else acc[name] + s
            counts[name] = counts.get(name, 0) + x.shape[0]
        return hook

    for name, m in target_linears(model):
        hooks.append(m.register_forward_pre_hook(mk(name)))

    t0 = time.time()
    step = ids.numel() // args.nsamples
    with torch.no_grad():
        for i in range(args.nsamples):
            chunk = ids[i * step: i * step + args.seqlen].unsqueeze(0).to(dev)
            if chunk.shape[1] < args.seqlen:
                break
            body(chunk)
            if (i + 1) % 8 == 0:
                print(f"{i+1}/{args.nsamples} {time.time()-t0:.0f}s", flush=True)

    for h in hooks:
        h.remove()
    norms = {k: (v / counts[k]).sqrt() for k, v in acc.items()}

    os.makedirs(args.out, exist_ok=True)
    tag = args.model.split("/")[-1]
    torch.save(norms, f"{args.out}/{tag}.pt")
    print(f"wrote {args.out}/{tag}.pt  {len(norms)} linears  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
