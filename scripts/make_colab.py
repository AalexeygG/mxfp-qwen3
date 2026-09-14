#!/usr/bin/env python
# pack the project sources into a self contained Colab notebook
import glob, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def md(text):
    lines = text.strip().split("\n")
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def code(text):
    lines = text.strip("\n").split("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def file_cell(path):
    rel = os.path.relpath(path, ROOT)
    body = open(path).read()
    return code(f"%%writefile {rel}\n{body}")


def build():
    cells = [
        md("# MXFP4 / MXFP8 quantization of Qwen3\nRun on a GPU runtime. Runtime -> Change runtime type -> T4 or better."),
        code("!nvidia-smi -L\nimport torch; print(torch.__version__, torch.cuda.is_available())"),
        code("!pip -q install 'transformers>=4.51' safetensors accelerate datasets zstandard 'lm-eval>=0.4.5' 2>&1 | tail -2"),
        code("!git clone -q --depth 1 https://github.com/microsoft/microxcaling.git vendor/microxcaling\n"
             "!mkdir -p src scripts results\n!ls vendor/microxcaling/mx | head -3"),
        md("## Project sources"),
    ]
    for p in sorted(glob.glob(f"{ROOT}/src/*.py")):
        cells.append(file_cell(p))
    for p in sorted(glob.glob(f"{ROOT}/scripts/*.py")) + sorted(glob.glob(f"{ROOT}/scripts/*.sh")):
        if os.path.basename(p) == "make_colab.py":
            continue
        cells.append(file_cell(p))

    cells += [
        md("## Quantization and per tensor stats\nStreams shard by shard, so model size is not limited by GPU memory."),
        code("!python scripts/quantize_model.py Qwen/Qwen3-0.6B\n"
             "!python scripts/quantize_model.py Qwen/Qwen3-1.7B\n"
             "!python scripts/quantize_model.py Qwen/Qwen3-4B"),
        code("!python scripts/memory_report.py"),
        md("## Calibration and bit allocation"),
        code("M = 'Qwen/Qwen3-1.7B'  # 4B needs L4 or A100\nimport os; os.environ['M'] = M\n"
             "TAG = M.split('/')[-1]; os.environ['TAG'] = TAG"),
        code("!python scripts/calibrate.py $M --nsamples 64 --device cuda\n"
             "!python scripts/sensitivity.py $M --alphas 0,1,5\n"
             "!python scripts/allocate.py $M --budgets 4.2,4.5,5.0,6.0\n"
             "!python scripts/signal_power.py $M"),
        md("## lm-eval matrix\nBaseline, then weight only, then our allocation, then activations."),
        code("import os\n"
             "os.environ['TASKS'] = 'wikitext,arc_easy,arc_challenge,piqa,winogrande,lambada_openai,hellaswag'\n"
             "os.environ['DEV'] = 'cuda'\nos.environ['BS'] = '16'\nos.environ['NWIN'] = '32'"),
        code("!bash scripts/ppl_matrix.sh $M"),
        code("!bash scripts/run_matrix.sh $M"),
        md("## Collect results"),
        code("!python scripts/analyze.py\n!zip -qr results.zip results\n"
             "from google.colab import files; files.download('results.zip')"),
    ]

    nb = {"cells": cells, "metadata": {"accelerator": "GPU",
          "colab": {"provenance": []},
          "kernelspec": {"display_name": "Python 3", "name": "python3"},
          "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}

    os.makedirs(f"{ROOT}/colab", exist_ok=True)
    out = f"{ROOT}/colab/run.ipynb"
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out, len(cells), "cells")


if __name__ == "__main__":
    build()
