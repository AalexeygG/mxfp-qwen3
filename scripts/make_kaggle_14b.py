#!/usr/bin/env python
# Qwen3-14B only, for a session of its own
import glob, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def md(t):
    ls = t.strip().split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in ls[:-1]] + [ls[-1]]}


def code(t):
    ls = t.strip("\n").split("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" for l in ls[:-1]] + [ls[-1]]}


def file_cell(p):
    return code(f"%%writefile {os.path.relpath(p, ROOT)}\n{open(p).read()}")


def build():
    cells = [
        md("# Qwen3-14B\n\n"
           "Settings: Accelerator GPU T4 x2, Internet on. The model is 27.5 GB, split across\n"
           "both cards with a small tail in host memory. T4 has no usable bf16, so float16.\n\n"
           "Two stages, each with its own archive, so the first is safe before the second starts.\n"
           "Stage one is the baseline and both formats on the weights, which is what the task\n"
           "statement asks for. Stage two is the method from this project, whose Hessian solve at\n"
           "this size is the most expensive step in the whole project.\n\n"
           "Two benchmark tasks, full sets, no subsampling. One benchmark pass here costs about\n"
           "sixteen times what it costs on Qwen3-4B, because part of the model crosses the bus on\n"
           "every forward."),
        code("import os\n"
             "os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'  # less fragmentation\n"
             "!nvidia-smi --query-gpu=name,memory.total --format=csv\n"
             "import torch; print(torch.__version__, torch.cuda.device_count(), 'GPUs')"),
        code("!pip -q install -U 'transformers>=4.51' safetensors accelerate datasets zstandard 'lm-eval>=0.4.5' 2>&1 | tail -3"),
        code("import os\nos.chdir('/kaggle/working')\n"
             "!git clone -q --depth 1 https://github.com/microsoft/microxcaling.git vendor/microxcaling\n"
             "!mkdir -p src scripts results"),
        md("## Project sources"),
    ]
    for p in sorted(glob.glob(f"{ROOT}/src/*.py")):
        cells.append(file_cell(p))
    for p in sorted(glob.glob(f"{ROOT}/scripts/*.py")) + sorted(glob.glob(f"{ROOT}/scripts/*.sh")):
        if os.path.basename(p).startswith("make_"):
            continue
        cells.append(file_cell(p))

    env = "MIN=1 NOERRCOMP=1 CORE=1 DEV=cuda DTYPE=float16 DEVMAP=auto MAXMEM=0=13GiB,1=13GiB,cpu=24GiB PYTORCH_ALLOC_CONF=expandable_segments:True"
    save = code("!cd /kaggle/working && zip -qr results_14b.zip results && ls -la results_14b.zip")
    cells += [
        md("## Calibration"),
        code("!PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/calibrate.py Qwen/Qwen3-14B "
             "--nsamples 64 --seqlen 512 --device cuda --dtype float16 --device-map auto "
             "--max-memory 0=13GiB,1=13GiB,cpu=24GiB"),
        md("## Perplexity"),
        code(f"!{env} NWIN=32 SEQLEN=512 bash scripts/ppl_matrix.sh Qwen/Qwen3-14B"),
        code("import os\n"
             "for d in ['results/ppl', 'results/eval']:\n"
             "    fs = sorted(os.listdir(d)) if os.path.isdir(d) else []\n"
             "    print(d, len(fs), 'files'); [print('  ', f) for f in fs]"),
        save,
        md("## lm-eval\nFirst, because the benchmark is what the task statement asks for."),
        code(f"!TASKS=arc_easy,lambada_openai {env} BS=8 MAXLEN=512 \\\n"
             f"  bash scripts/run_matrix.sh Qwen/Qwen3-14B"),
        save,
    ]

    cells += [
        md("# Qwen3-14B, the method from this project\n\n"
           "Everything above is saved before this starts. The Hessian solve at this size is the\n"
           "most expensive step in the project, so it runs last and has its own archive."),
        md("## lm-eval with error compensation"),
        code("!TASKS=arc_easy,lambada_openai ONLYERRCOMP=1 DEV=cuda DTYPE=float16 DEVMAP=auto "
             "MAXMEM=0=13GiB,1=13GiB,cpu=24GiB PYTORCH_ALLOC_CONF=expandable_segments:True "
             "BS=8 MAXLEN=512 \\\n  bash scripts/run_matrix.sh Qwen/Qwen3-14B"),
        md("## Perplexity with error compensation"),
        code("!ONLYERRCOMP=1 DEV=cuda DTYPE=float16 DEVMAP=auto MAXMEM=0=13GiB,1=13GiB,cpu=24GiB "
             "PYTORCH_ALLOC_CONF=expandable_segments:True NWIN=32 SEQLEN=512 \\\n"
             "  bash scripts/ppl_matrix.sh Qwen/Qwen3-14B"),
        code("import os\n"
             "for d in ['results/ppl', 'results/eval']:\n"
             "    fs = sorted(os.listdir(d)) if os.path.isdir(d) else []\n"
             "    print(d, len(fs), 'files'); [print('  ', f) for f in fs]"),
        code("!cd /kaggle/working && zip -qr results_14b.zip results && ls -la results_14b.zip"),
    ]

    nb = {"cells": cells, "metadata": {"accelerator": "GPU",
          "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    out = f"{ROOT}/kaggle/run_14b.ipynb"
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out, len(cells), "cells")


if __name__ == "__main__":
    build()
