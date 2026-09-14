# error compensated MX quantization, group size follows the MX block
import torch

from mxfmt import BLOCK, block_exponent, quantize_with_exponent


class Hessian:
    """Accumulates X^T X over calibration batches for one linear layer."""

    def __init__(self, n_in, device="cpu"):
        self.h = torch.zeros(n_in, n_in, dtype=torch.float32, device=device)
        self.n = 0

    def add(self, x):
        x = x.reshape(-1, x.shape[-1]).float()
        self.h += x.T @ x
        self.n += x.shape[0]

    def finish(self):
        return self.h / max(self.n, 1)


@torch.no_grad()
def quantize_compensated(w, elem_format, h, block=BLOCK, damp=0.01, search=False):
    """Quantize a weight block by block, pushing each block error onto the columns still to come."""
    w = w.float().clone()
    n_in = w.shape[1]
    d = torch.arange(n_in, device=w.device)
    diag = torch.diag(h)
    dead = diag == 0
    h = h.clone()
    h[dead, dead] = 1
    w[:, dead] = 0
    h[d, d] += damp * diag.mean()

    try:
        hinv = torch.linalg.cholesky(h)
        hinv = torch.cholesky_inverse(hinv)
        hinv = torch.linalg.cholesky(hinv, upper=True)
    except RuntimeError:
        return quantize_with_exponent(w, elem_format, block_exponent(
            w.reshape(w.shape[0], -1, block), elem_format).unsqueeze(-1).expand(-1, -1, block).reshape(w.shape))

    q = torch.zeros_like(w)
    for i1 in range(0, n_in, block):
        i2 = min(i1 + block, n_in)
        cnt = i2 - i1
        wb = w[:, i1:i2].clone()
        qb = torch.zeros_like(wb)
        eb = torch.zeros_like(wb)
        hb = hinv[i1:i2, i1:i2]

        exp = block_exponent(wb, elem_format).unsqueeze(-1)
        if search:
            best = None
            for off in (0, 1):
                cand = quantize_with_exponent(wb, elem_format, exp + off)
                err = (wb - cand).pow(2).sum(-1, keepdim=True)
                if best is None or True:
                    if best is None:
                        best, best_err, best_exp = cand, err, exp + off
                    else:
                        take = err < best_err
                        best = torch.where(take, cand, best)
                        best_exp = torch.where(take, exp + off, best_exp)
                        best_err = torch.where(take, err, best_err)
            exp = best_exp

        for j in range(cnt):
            col = wb[:, j]
            qc = quantize_with_exponent(col.unsqueeze(-1), elem_format, exp).squeeze(-1)
            qb[:, j] = qc
            e = (col - qc) / hb[j, j]
            if j + 1 < cnt:
                wb[:, j + 1:] -= e.unsqueeze(1) * hb[j, j + 1:].unsqueeze(0)
            eb[:, j] = e

        q[:, i1:i2] = qb
        if i2 < n_in:
            w[:, i2:] -= eb @ hinv[i1:i2, i2:]

    return q


def _run(w, h, ef, damp, search):
    """Solve on the weight's own GPU when it fits, fall back to cpu on an allocation failure."""
    if w.device.type == "cuda":
        try:
            return quantize_compensated(w.float(), ef, h.to(w.device), damp=damp, search=search)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    return quantize_compensated(w.float().cpu(), ef, h.cpu(), damp=damp, search=search)


def _layer_index(name):
    import re
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


@torch.no_grad()
def apply_compensated(model, calib_ids, fmt, device, group=2, nsamples=16, seqlen=1024,
               damp=0.01, search=False, verbose=True):
    """Quantize all linears layer group by layer group so later groups see quantized inputs.

    Group size trades passes for memory: one Hessian is d_in by d_in floats, and down_proj on a
    4B model is already 378 MB of that.
    """
    import time
    from fakequant import target_linears
    from mxfmt import FORMATS

    ef = FORMATS[fmt]
    mods = target_linears(model)
    layers = sorted({_layer_index(n) for n, _ in mods if _layer_index(n) >= 0})
    t0 = time.time()

    for start in range(0, len(layers), group):
        sel = set(layers[start:start + group])
        todo = [(n, m) for n, m in mods if _layer_index(n) in sel]
        if not todo:
            continue
        hs, handles = {}, []

        def mk(name, n_in):
            hs[name] = Hessian(n_in)

            def hook(mod, args):
                hs[name].add(args[0].detach().cpu())
            return hook

        for n, m in todo:
            handles.append(m.register_forward_pre_hook(mk(n, m.in_features)))

        step = calib_ids.numel() // nsamples
        for i in range(nsamples):
            x = calib_ids[i * step:i * step + seqlen].unsqueeze(0).to(device)
            if x.shape[1] < seqlen:
                break
            model(x)
        for h in handles:
            h.remove()

        for n, m in todo:
            w = m.weight.data
            h = hs[n].finish()
            q = _run(w, h, ef, damp, search)
            m.weight.data = q.to(device=w.device, dtype=w.dtype)
            del hs[n], h
        if verbose:
            print(f"  layers {sorted(sel)} done, {time.time()-t0:.0f}s", flush=True)
    return len(mods)


def calib_tokens(tok, source="wikitext", n_chars=2_000_000):
    """Calibration text, either wikitext or a slice of C4 web text."""
    from datasets import load_dataset
    if source == "c4":
        ds = load_dataset("allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
                          split="train", streaming=True)
        parts, total = [], 0
        for r in ds:
            parts.append(r["text"])
            total += len(r["text"])
            if total >= n_chars:
                break
        text = "\n\n".join(parts)
    else:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(ds["text"][:20000])
    return tok(text, return_tensors="pt").input_ids[0]
