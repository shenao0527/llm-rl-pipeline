#!/bin/bash
# ============================================================
# data/download.sh — 数据集下载脚本
#
# 从 HuggingFace 下载 Jackrong 蒸馏数据集 (Apache 2.0)
# 使用 huggingface_hub 的 snapshot_download
#
# 使用方法:
#   bash data/download.sh
# ============================================================

set -euo pipefail

echo "========================================"
echo "  Downloading Jackrong Distill Datasets"
echo "========================================"
echo ""

# 数据集列表
DATASETS=(
    "R6410418/Chinese-Qwen3-235B-Thinking-2507-Distill-100k"
    "R6410418/gpt-oss-120b-distilled-reasoning"
)

CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"

for ds in "${DATASETS[@]}"; do
    echo "  Downloading: $ds ..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$ds', repo_type='dataset')
" || echo "  ⚠️ Failed to download $ds (check network or token)"
    echo ""
done

echo "========================================"
echo "  Download complete!"
echo "  Cached at: $CACHE_DIR"
echo ""
echo "  Next: Run Notebook 1 to format data for training."
echo "========================================"