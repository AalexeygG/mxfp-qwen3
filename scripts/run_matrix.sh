#!/bin/bash
# full evaluation matrix for one model
set -u
M=${1:-Qwen/Qwen3-0.6B}
TAG=$(basename $M)
TASKS=${TASKS:-wikitext,arc_easy,piqa,winogrande,lambada_openai}
DEV=${DEV:-mps}
DEVMAP=${DEVMAP:-}
DTYPE=${DTYPE:-bfloat16}
EXTRA=""
MAXMEM=${MAXMEM:-}
[ -n "$DEVMAP" ] && EXTRA="--device-map $DEVMAP"
[ -n "$MAXMEM" ] && EXTRA="$EXTRA --max-memory $MAXMEM"
BS=${BS:-4}
MAXLEN=${MAXLEN:-2048}
# use the local venv when there is one, plain python on Colab or Kaggle
if [ -z "${PY:-}" ]; then
  if [ -x ./venv/bin/python ]; then PY=./venv/bin/python; else PY=python; fi
fi
LIMIT=${LIMIT:-}
LIMARG=""; [ -n "$LIMIT" ] && LIMARG="--limit $LIMIT"
run() {
  echo "### $* $(date +%T)"
  out=$($PY -u scripts/run_eval.py $M --tasks $TASKS --device $DEV --batch-size $BS \
        --max-length $MAXLEN --dtype $DTYPE $EXTRA $LIMARG "$@" 2>&1)
  if echo "$out" | grep -aqE "^[a-z_0-9]+ \{"; then
    echo "$out" | grep -aE "^[a-z_0-9]+ \{"
  else
    echo "FAILED: $*"
    echo "$out" | tr '\r' '\n' | grep -av "it/s\]$" | tail -8
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
if [ -z "${CORE:-}" ]; then
  [ -f results/plans_${TAG}.json ] && run --plan results/plans_${TAG}.json --budget 4.2 --calib results/calib/${TAG}.pt --tag ${TAG}_ours-4.2
fi
echo "### matrix done $(date +%T)"
