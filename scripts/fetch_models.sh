#!/bin/bash
# download checkpoints in order of priority
export HF_HUB_ENABLE_HF_TRANSFER=1
for m in Qwen/Qwen3-1.7B Qwen/Qwen3-0.6B Qwen/Qwen3-4B Qwen/Qwen3-14B; do
  echo "=== $m $(date +%T)"
  ./venv/bin/hf download "$m" || echo "FAILED $m"
done
echo "=== all done $(date +%T)"
