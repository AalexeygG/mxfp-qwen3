#!/usr/bin/env python
# lm-eval on a model with optional MX fake quantization of weights and activations
import argparse, json, os, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from fakequant import quantize_weights, ActQuant, apply_plan
import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer


def build(model_id, wfmt, afmt, device, dtype, lm_head, plan_file=None, budget=None, calib=None,
          search=False, device_map=None, max_memory=None):
    tok = AutoTokenizer.from_pretrained(model_id)
    kw = dict(device_map=device_map) if device_map else {}
    if max_memory:
        kw["max_memory"] = {(int(k) if k.isdigit() else k): v
                            for k, v in (q.split("=") for q in max_memory.split(","))}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, **kw)
    if not device_map:
        model = model.to(device)
    model = model.eval()
    act = None
    if plan_file:
        with open(plan_file) as f:
            plans = json.load(f)
        plan = plans["budgets"][budget]["plan"]
        norms = torch.load(calib) if calib else None
        t = time.time()
        info = apply_plan(model, plan, norms)
        print(f"applied plan at budget {budget}: {info['tensors']} tensors, "
              f"{info['bits_per_value']:.3f} bits per value, {time.time()-t:.0f}s", flush=True)
    elif wfmt != "none":
        t = time.time()
        n = quantize_weights(model, wfmt, lm_head, search=search)
        how = " with exponent search" if search else ""
        print(f"quantized {n/1e6:.0f}M weights to {wfmt}{how} in {time.time()-t:.0f}s", flush=True)
    if afmt != "none":
        act = ActQuant(model, afmt, lm_head)
        print(f"activations quantized to {afmt}", flush=True)
    return model, tok, act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--weights", default="none")
    ap.add_argument("--acts", default="none")
    ap.add_argument("--tasks", default="wikitext")
    ap.add_argument("--limit", type=float, default=None)
    ap.add_argument("--batch-size", default="4")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--device-map", default=None, help="auto spreads the model over every visible GPU")
    ap.add_argument("--max-memory", default=None, help="per device cap, e.g. 0=13GiB,1=13GiB,cpu=20GiB")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--quant-lm-head", action="store_true")
    ap.add_argument("--errcomp", default=None)
    ap.add_argument("--calib-source", default="wikitext", choices=["wikitext", "c4"])
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--plan", default=None)
    ap.add_argument("--budget", default=None)
    ap.add_argument("--calib", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default="results/eval")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    model, tok, act = build(args.model, args.weights, args.acts, args.device, dtype,
                            args.quant_lm_head, args.plan, args.budget, args.calib, args.search,
                            args.device_map, args.max_memory)
    if args.errcomp:
        from errcomp import apply_compensated, calib_tokens
        ids = calib_tokens(tok, args.calib_source)
        t = time.time()
        dev = next(model.parameters()).device if args.device_map else args.device
        apply_compensated(model, ids, args.errcomp, dev, search=args.search)
        print(f"error compensation in {args.errcomp} done in {time.time()-t:.0f}s", flush=True)

    hf_kw = dict(pretrained=model, tokenizer=tok, batch_size=args.batch_size,
                 max_length=args.max_length)
    if not args.device_map:
        hf_kw["device"] = args.device
    lm = HFLM(**hf_kw)

    t0 = time.time()
    res = lm_eval.simple_evaluate(model=lm, tasks=args.tasks.split(","), limit=args.limit, verbosity="WARNING")
    dt = time.time() - t0

    tag = args.tag or f"{args.model.split('/')[-1]}_w-{args.weights}_a-{args.acts}"
    os.makedirs(args.out, exist_ok=True)
    payload = dict(model=args.model, weights=args.weights, acts=args.acts, tasks=args.tasks,
                   plan=args.plan, budget=args.budget, errcomp=args.errcomp, search=args.search, calib_source=args.calib_source,
                   limit=args.limit, dtype=args.dtype, device=args.device, seconds=dt,
                   max_length=args.max_length, device_map=args.device_map,
                   results=res["results"], n_samples={k: v for k, v in res.get("n-samples", {}).items()})
    with open(f"{args.out}/{tag}.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    for task, r in res["results"].items():
        print(task, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)})
    print(f"done in {dt:.0f}s -> {args.out}/{tag}.json")


if __name__ == "__main__":
    main()
