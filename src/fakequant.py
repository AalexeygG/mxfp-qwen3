# in place MX fake quantization of a loaded HF model, weights and activations
import json

import torch
import torch.nn as nn

from mxfmt import quantize, quantize_search, FORMATS

SKIP = ("lm_head",)


def target_linears(model, include_lm_head=False):
    """All nn.Linear modules that carry transformer matmul weights."""
    out = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            if not include_lm_head and any(s in name for s in SKIP):
                continue
            out.append((name, m))
    return out


@torch.no_grad()
def quantize_weights(model, fmt, include_lm_head=False, chunk_rows=2048, search=False,
                     act_norms=None, offsets=(0, 1)):
    """Replace each linear weight by its MX round trip, in row chunks to cap memory."""
    ef = FORMATS[fmt]
    n = 0
    for name, m in target_linears(model, include_lm_head):
        w = m.weight.data
        a = None if act_norms is None else act_norms.get(name)
        for i in range(0, w.shape[0], chunk_rows):
            c = w[i:i + chunk_rows].float()
            if search:
                q, _, _ = quantize_search(c, ef, offsets=offsets, act_norm=a)
            else:
                q = quantize(c, ef)
            w[i:i + chunk_rows] = q.to(w.dtype)
        n += w.numel()
    return n


class ActQuant:
    """Forward pre hooks that quantize linear inputs along the reduction axis."""

    def __init__(self, model, fmt, include_lm_head=False):
        self.ef = FORMATS[fmt]
        self.handles = []
        for name, m in target_linears(model, include_lm_head):
            self.handles.append(m.register_forward_pre_hook(self._hook))

    def _hook(self, module, args):
        x = args[0]
        return (quantize(x.float(), self.ef).to(x.dtype),) + args[1:]

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def apply_plan(model, plan, act_norms=None, protect_top=0.01, chunk_rows=4096):
    """Rebuild each linear weight from its planned format and coarsening strength."""
    from mxfmt import quantize_full, dequantize_codes
    from compress import compress_tensor

    total_bytes, total_n, touched = 0, 0, 0
    for name, m in target_linears(model):
        spec = plan.get(name + ".weight")
        if spec is None:
            continue
        ef = FORMATS[spec["fmt"]]
        a = None if act_norms is None else act_norms.get(name)
        w = m.weight.data
        for i in range(0, w.shape[0], chunk_rows):
            c = w[i:i + chunk_rows].float().cpu()  # packing and zstd run on cpu anyway
            deq, se, codes, meta = quantize_full(c, ef)
            if spec["alpha"] > 0:
                ref = (c - deq).pow(2).mean().item()
                r = compress_tensor(codes, se, ef, act_norm=a, alpha=spec["alpha"],
                                    ref_mse=ref, protect_top=protect_top)
                rec = dequantize_codes(r["codes"], se, ef, meta)
                total_bytes += r["bytes_codes"] + r["bytes_scales"]
            else:
                r = compress_tensor(codes, se, ef, alpha=0.0)
                rec = deq
                total_bytes += r["bytes_codes"] + r["bytes_scales"]
            w[i:i + chunk_rows] = rec.to(device=w.device, dtype=w.dtype)
            total_n += c.numel()
        touched += 1
    return dict(tensors=touched, params=total_n, bytes=total_bytes,
                bits_per_value=total_bytes * 8 / max(total_n, 1))


def build_model(model_id, weights="none", acts="none", device="cpu", dtype=torch.bfloat16,
                lm_head=False, plan_file=None, budget=None, calib=None, search=False,
                device_map=None, max_memory=None):
    """Load a model and apply one quantization configuration to it."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
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

    info = {}
    if plan_file:
        with open(plan_file) as f:
            plans = json.load(f)
        norms = torch.load(calib) if calib else None
        info = apply_plan(model, plans["budgets"][budget]["plan"], norms)
    elif weights != "none":
        norms = torch.load(calib) if calib else None
        quantize_weights(model, weights, lm_head, search=search, act_norms=norms)
    act = ActQuant(model, acts, lm_head) if acts != "none" else None
    return model, tok, act, info
