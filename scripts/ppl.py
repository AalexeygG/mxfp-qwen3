#!/usr/bin/env python
# direct perplexity on wikitext2, more sensitive than accuracy on small samples
import argparse, json, os, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from fakequant import build_model
from datasets import load_dataset


@torch.no_grad()
def perplexity(model, tok, device, seqlen=1024, nwin=32, chunk=128):
    """Sliced cross entropy, the full logits tensor does not fit next to a 150k vocabulary."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    nll, ntok = 0.0, 0
    for i in range(min(nwin, ids.numel() // seqlen)):
        x = ids[i * seqlen:(i + 1) * seqlen].unsqueeze(0).to(device)
        body = model.model if hasattr(model, "model") else model
        h = body(x)[0][0, :-1]
        tgt = x[0, 1:]
        for j in range(0, h.shape[0], chunk):
            lg = model.lm_head(h[j:j + chunk]).float()
            nll += torch.nn.functional.cross_entropy(lg, tgt[j:j + chunk], reduction="sum").item()
            del lg
        ntok += tgt.numel()
        del h, x
    return float(torch.exp(torch.tensor(nll / ntok)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--weights", default="none")
    ap.add_argument("--acts", default="none")
    ap.add_argument("--plan", default=None)
    ap.add_argument("--budget", default=None)
    ap.add_argument("--calib", default=None)
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--errcomp", default=None)
    ap.add_argument("--calib-source", default="wikitext", choices=["wikitext", "c4"])
    ap.add_argument("--errcomp-group", type=int, default=2)
    ap.add_argument("--errcomp-nsamples", type=int, default=16)
    ap.add_argument("--nwin", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--device-map", default=None, help="auto spreads the model over every visible GPU")
    ap.add_argument("--max-memory", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default="results/ppl")
    args = ap.parse_args()

    t0 = time.time()
    model, tok, act, info = build_model(args.model, args.weights, args.acts, args.device,
                                        getattr(torch, args.dtype), False, args.plan, args.budget,
                                        args.calib, args.search, args.device_map, args.max_memory)
    if args.errcomp:
        from errcomp import apply_compensated, calib_tokens
        ids = calib_tokens(tok, args.calib_source)
        dev0 = next(model.parameters()).device if args.device_map else args.device
        apply_compensated(model, ids, args.errcomp, dev0, group=args.errcomp_group,
                   nsamples=args.errcomp_nsamples, seqlen=args.seqlen, search=args.search)
    dev = next(model.parameters()).device if args.device_map else args.device
    ppl = perplexity(model, tok, dev, args.seqlen, args.nwin)
    tag = args.tag or f"{args.model.split('/')[-1]}_w-{args.weights}_a-{args.acts}"
    os.makedirs(args.out, exist_ok=True)
    rec = dict(model=args.model, weights=args.weights, acts=args.acts, plan=args.plan, search=args.search,
               errcomp=args.errcomp,
               budget=args.budget, nwin=args.nwin, seqlen=args.seqlen, ppl=ppl,
               dtype=args.dtype, device_map=args.device_map,
               bits_per_value=info.get("bits_per_value"), seconds=time.time() - t0)
    with open(f"{args.out}/{tag}.json", "w") as f:
        json.dump(rec, f, indent=2)
    print(f"{tag}: ppl {ppl:.4f}  bits {info.get('bits_per_value')}  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
