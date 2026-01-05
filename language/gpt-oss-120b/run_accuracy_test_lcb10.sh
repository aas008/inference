#!/bin/bash
# LiveCodeBench accuracy test with vLLM backend (10 samples only)

set -e

cd /mnt/mlperf_gpt_oss/language/gpt-oss-120b
source .venv/bin/activate

# Disable vLLM v1 engine (use stable legacy engine)
export VLLM_USE_V1=0

# Use full v4 dataset but limit to first 10 samples (all LiveCodeBench)
LCB_DATASET="/mnt/mlperf_gpt_oss/dataset/v4/acc/acc_eval_ref.parquet"

echo "================================================================================"
echo "LiveCodeBench Accuracy Test (10 samples)"
echo "================================================================================"
echo "Dataset: First 10 samples from v4"
echo "Backend: vLLM with max_concurrency=1"
echo "Max tokens: 32768"
echo ""
echo "================================================================================"
echo "Step 1: Running MLPerf Accuracy Mode"
echo "================================================================================"

python3 run_mlperf.py \
    --backend vllm \
    --scenario offline \
    --accuracy \
    --max-concurrency 1 \
    --input-file "$LCB_DATASET" \
    --output-dir mlperf_lcb_test

echo ""
echo "================================================================================"
echo "Step 2: Evaluating Accuracy Results"
echo "================================================================================"

python3 eval_mlperf_accuracy.py \
    --mlperf-log mlperf_lcb_test/offline/accuracy/mlperf_log_accuracy.json \
    --reference-data "$LCB_DATASET" \
    --tokenizer /mnt/models/gpt-oss-120b \
    --output-file mlperf_lcb_test/accuracy_results.json \
    --verbose

echo ""
echo "================================================================================"
echo "Test Complete!"
echo "Results saved to: mlperf_lcb_test/accuracy_results.json"
echo "================================================================================"
