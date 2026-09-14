#!/usr/bin/env python
# Qwen3-4B only, sized to finish inside a short session
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
        md("# Qwen3-4B\n\n"
           "Settings: Accelerator GPU T4 x2, Internet on.\n\n"
           "Six configurations, the ones the task statement asks for: the 16 bit baseline, both\n"
           "formats on the weights, both activation variants, and the method from this project.\n"
           "lm-eval runs before perplexity because the benchmark is the requirement and the\n"
           "perplexity is the extra measurement."),
        code("!nvidia-smi --query-gpu=name,memory.total --format=csv\n"
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

    env = "CORE=1 DEV=cuda DTYPE=float16"
    cells += [
        md("## Calibration"),
        code("!python scripts/calibrate.py Qwen/Qwen3-4B --nsamples 64 --device cuda --dtype float16"),
        md("## lm-eval\nFour tasks, full sets, no subsampling."),
        code(f"!TASKS=arc_easy,piqa,winogrande,lambada_openai {env} BS=16 MAXLEN=1024 \\\n"
             f"  bash scripts/run_matrix.sh Qwen/Qwen3-4B"),
        md("## Perplexity"),
        code(f"!{env} NWIN=32 bash scripts/ppl_matrix.sh Qwen/Qwen3-4B"),
        md("## What was produced"),
        code("import os\n"
             "for d in ['results/ppl', 'results/eval']:\n"
             "    fs = sorted(os.listdir(d)) if os.path.isdir(d) else []\n"
             "    print(d, len(fs), 'files'); [print('  ', f) for f in fs]"),
        code("!cd /kaggle/working && zip -qr results_4b.zip results && ls -la results_4b.zip"),
    ]

    nb = {"cells": cells, "metadata": {"accelerator": "GPU",
          "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    out = f"{ROOT}/kaggle/run_4b.ipynb"
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out, len(cells), "cells")


if __name__ == "__main__":
    build()
