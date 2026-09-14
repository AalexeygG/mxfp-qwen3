# streaming access to safetensors checkpoints, never loads a whole model
import glob, json, os, re
from safetensors import safe_open
from huggingface_hub import snapshot_download

PROJ_RE = re.compile(r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$")
LAYER_RE = re.compile(r"layers\.(\d+)\.")


def local_path(model_id):
    return snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.txt"])


def shard_files(path):
    return sorted(glob.glob(os.path.join(path, "*.safetensors")))


def iter_tensors(path, only_2d=True):
    """Yield (name, tensor) one at a time so peak memory stays at one tensor."""
    for f in shard_files(path):
        with safe_open(f, framework="pt") as sf:
            for name in sf.keys():
                t = sf.get_tensor(name)
                if only_2d and t.ndim != 2:
                    continue
                yield name, t
                del t


def tensor_index(path):
    """Map of tensor name to shape and dtype without reading the data."""
    out = {}
    for f in shard_files(path):
        with safe_open(f, framework="pt") as sf:
            for name in sf.keys():
                sl = sf.get_slice(name)
                out[name] = (tuple(sl.get_shape()), sl.get_dtype())
    return out


def classify(name):
    """Coarse role of a tensor, used to group the analysis."""
    m = PROJ_RE.search(name)
    if m:
        return m.group(1)
    if "embed_tokens" in name:
        return "embed_tokens"
    if "lm_head" in name:
        return "lm_head"
    if "norm" in name:
        return "norm"
    return "other"


def layer_of(name):
    m = LAYER_RE.search(name)
    return int(m.group(1)) if m else -1
