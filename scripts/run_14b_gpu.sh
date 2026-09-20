#!/bin/bash
# run the error-compensated method on Qwen3-14B
set -eu
cd /root 2>/dev/null || cd ~
[ -d mxfp-qwen3 ] || git clone -q https://github.com/AalexeygG/mxfp-qwen3.git
cd mxfp-qwen3
pip install -q -r requirements.txt
[ -d vendor/microxcaling ] || git clone -q --depth 1 https://github.com/microsoft/microxcaling.git vendor/microxcaling

python scripts/calibrate.py Qwen/Qwen3-14B --nsamples 64 --device cuda
ONLYERRCOMP=1 DEV=cuda TASKS=arc_easy,lambada_openai BS=8 bash scripts/run_matrix.sh Qwen/Qwen3-14B
ONLYERRCOMP=1 DEV=cuda bash scripts/ppl_matrix.sh Qwen/Qwen3-14B

echo "done, results in results/eval and results/ppl:"
ls results/eval/*errcomp* results/ppl/*errcomp* 2>/dev/null
