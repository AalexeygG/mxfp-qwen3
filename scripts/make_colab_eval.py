#!/usr/bin/env python
# minimal notebook that only measures quality, the layer statistics are already done
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
    return code(f"%%writefile {os.path.relpath(path, ROOT)}\n{open(path).read()}")


def build():
    cells = [
        md("# Quality measurements\n"
           "Run every cell in order. Qwen3-1.7B first, then Qwen3-4B if the session survives.\n"
           "Qwen3-14B is not here: 27.5 GB of bf16 weights do not fit a 16 GB T4."),
        code("!nvidia-smi -L\nimport torch; print(torch.__version__, torch.cuda.is_available())"),
        code("!pip -q install 'transformers>=4.51' safetensors accelerate datasets zstandard 'lm-eval>=0.4.5' 2>&1 | tail -2"),
        code("!git clone -q --depth 1 https://github.com/microsoft/microxcaling.git vendor/microxcaling\n"
             "!mkdir -p src scripts results"),
        md("## Project sources"),
    ]
    for p in sorted(glob.glob(f"{ROOT}/src/*.py")):
        cells.append(file_cell(p))
    for p in sorted(glob.glob(f"{ROOT}/scripts/*.py")) + sorted(glob.glob(f"{ROOT}/scripts/*.sh")):
        if os.path.basename(p).startswith("make_colab"):
            continue
        cells.append(file_cell(p))

    evals = []
    for tag, note in [("Qwen3-1.7B", "3.4 GB of weights, comfortable on a T4"),
                      ("Qwen3-4B", "8 GB of weights, fits a 16 GB T4 but leaves little room")]:
        m = f"Qwen/{tag}"
        evals += [
            md(f"# {tag}\n{note}"),
            md("## Calibration\nNeeded by the error compensated run, about a minute on a GPU."),
            code(f"!python scripts/calibrate.py {m} --nsamples 64 --device cuda"),
            md("## Perplexity\nEach line prints one number. If one config fails the rest still run."),
            code(f"!DEV=cuda NWIN=32 bash scripts/ppl_matrix.sh {m}"),
            md("## lm-eval\nThe long one. Expect a table of accuracies per configuration."),
            code(f"!TASKS=arc_easy,piqa,winogrande,lambada_openai DEV=cuda BS=8 \\\n"
                 f"  bash scripts/run_matrix.sh {m}"),
            code("import os\n"
                 "for d in ['results/ppl', 'results/eval']:\n"
                 "    fs = sorted(os.listdir(d)) if os.path.isdir(d) else []\n"
                 "    print(d, len(fs), 'files'); [print('  ', f) for f in fs]"),
            md("## Save what is done so far\nRun this after each model, so a lost session does not cost everything."),
            code("!zip -qr results_eval.zip results\n"
                 "from google.colab import files; files.download('results_eval.zip')"),
        ]
    cells += evals

    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
          "kernelspec": {"display_name": "Python 3", "name": "python3"},
          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
    out = f"{ROOT}/colab/eval_only.ipynb"
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out, len(cells), "cells")


if __name__ == "__main__":
    build()
