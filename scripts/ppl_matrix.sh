#!/bin/bash
# perplexity across all quantization configurations for one model
set -u
M=${1:-Qwen/Qwen3-0.6B}
TAG=$(basename $M)
DEV=${DEV:-mps}
DEVMAP=${DEVMAP:-}
DTYPE=${DTYPE:-bfloat16}
EXTRA=""
MAXMEM=${MAXMEM:-}
[ -n "$DEVMAP" ] && EXTRA="--device-map $DEVMAP"
[ -n "$MAXMEM" ] && EXTRA="$EXTRA --max-memory $MAXMEM"
NWIN=${NWIN:-32}
SEQLEN=${SEQLEN:-1024}
# use the local venv when there is one, plain python on Colab or Kaggle
if [ -z "${PY:-}" ]; then
  if [ -x ./venv/bin/python ]; then PY=./venv/bin/python; else PY=python; fi
fi
# print the result line on success, the tail of the output on failure
run() {
  out=$($PY -u scripts/ppl.py $M --device $DEV --nwin $NWIN --seqlen $SEQLEN --dtype $DTYPE $EXTRA "$@" 2>&1)
  if echo "$out" | grep -aqE "^[A-Za-z0-9_.-]+: ppl "; then
    echo "$out" | grep -aE "^[A-Za-z0-9_.-]+: ppl "
  else
    echo "FAILED: $*"
    echo "$out" | tr '\r' '\n' | grep -av "it/s\]$" | tail -6
  fi
}

# ONLYERRCOMP skips straight to the slow compensated configuration
if [ -n "${ONLYERRCOMP:-}" ]; then
  run --errcomp mxfp4 --search --tag ${TAG}_errcomp-mxfp4_search
  exit 0
fi

# ordered by priority, the task statement asks for weights and activations first
run --tag ${TAG}_bf16
run --weights mxfp8 --tag ${TAG}_w-mxfp8
run --weights mxfp4 --tag ${TAG}_w-mxfp4
[ -n "${MIN:-}" ] && { echo "### minimal set done $(date +%T)"; exit 0; }
run --weights mxfp8 --acts mxfp8 --tag ${TAG}_w-mxfp8_a-mxfp8
run --weights mxfp4 --acts mxfp8 --tag ${TAG}_w-mxfp4_a-mxfp8
[ -z "${NOERRCOMP:-}" ] && run --errcomp mxfp4 --search --tag ${TAG}_errcomp-mxfp4_search

# CORE=1 stops here, everything below is extra
[ -n "${CORE:-}" ] && exit 0

run --acts mxfp8 --tag ${TAG}_a-mxfp8
run --acts mxfp4 --tag ${TAG}_a-mxfp4
run --weights mxfp4 --search --tag ${TAG}_w-mxfp4_search
run --weights mxfp8 --search --tag ${TAG}_w-mxfp8_search
run --errcomp mxfp4 --tag ${TAG}_errcomp-mxfp4
if [ -f results/plans_${TAG}.json ]; then
  for b in 4.2 5.0; do
    run --plan results/plans_${TAG}.json --budget $b --calib results/calib/${TAG}.pt --tag ${TAG}_ours-$b
  done
fi
