#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/build/Release"

BM="./tests/ut/knowhere_benchmark"
OUTPUT="$SCRIPT_DIR/output.txt"
RESULT="$SCRIPT_DIR/result.txt"

# Clear output files
echo "=== Benchmark Raw Output ===" > "$OUTPUT"
echo "Date: $(date)" >> "$OUTPUT"
echo "" >> "$OUTPUT"

echo "=== Benchmark Results ===" > "$RESULT"
echo "Date: $(date)" >> "$RESULT"
echo "" >> "$RESULT"
printf "%-10s %-20s %-12s %-12s %-10s %-10s\n" "Dataset" "Index" "Build(s)" "QPS" "Recall" "AvgNbrs" >> "$RESULT"
printf "%-10s %-20s %-12s %-12s %-10s %-10s\n" "-------" "-----" "--------" "---" "------" "-------" >> "$RESULT"

# Function to run benchmark
run_benchmark() {
    local dataset=$1
    local index=$2
    local label=$3
    local extra_args=$4

    echo ""
    echo "========================================"
    echo "Running: $dataset $label $extra_args"
    echo "========================================"

    # Run benchmark with tee to show in terminal and save to output.txt
    $BM --dataset "$dataset" --index "$index" --topk 100 --times 10 $extra_args 2>&1 | tee -a "$OUTPUT"
}

# Dataset configurations: "name:dim:metric"
# DATASETS="siftsmall:128:L2 cohere:768:COSINE gist:768:L2"
DATASETS="cohere:768:COSINE gist:768:L2"

# NN_DESCENT_NITER values to test
NN_DESCENT_NITER_VALUES="20 40 60 80 100"

# VAMANA_ITERS values to test
VAMANA_ITERS_VALUES="1 2"

# Run benchmarks for each dataset
for ds_config in $DATASETS; do
    IFS=':' read -r ds dim metric <<< "$ds_config"

    # Generate ground truth if not exists
    truth_file="/home/ubuntu/data/$ds/${ds}_query_*_100.truth"
    if ls $truth_file 1>/dev/null 2>&1; then
        echo "Ground truth for $ds already exists, skipping..."
    else
        echo "Generating ground truth for $ds..."
        $BM --mode truth --dataset $ds --topk 100 2>&1 | tee -a "$OUTPUT"
    fi

    echo "" >> "$RESULT"
    echo "# $ds ($dim dim, $metric)" >> "$RESULT"

    run_benchmark "$ds" "HNSWLIB_DEPRECATED" "HNSWLIB_M32" ""

    for niter in $VAMANA_ITERS_VALUES; do
        run_benchmark "$ds" "GPU_VAMANA" "VAMANA_D64" "--adapt_for_cpu --vamana_iters $niter"
    done

    for niter in $NN_DESCENT_NITER_VALUES; do
        run_benchmark "$ds" "GPU_CAGRA" "CAGRA_D64_iter${niter}" "--adapt_for_cpu --nn_descent_niter $niter"
    done
done

echo ""
echo "=== All benchmarks completed ==="
echo ""

# Extract results from output.txt to result.txt
echo "Extracting results from output.txt..."

# Parse output.txt and extract metrics for each run
while IFS= read -r line; do
    if [[ "$line" =~ dataset=([a-z]+).*index=([A-Z_0-9]+) ]]; then
        dataset="${BASH_REMATCH[1]}"
        label="${BASH_REMATCH[2]}"
    elif [[ "$line" =~ build_time=([0-9.]+)s.*qps=([0-9.]+) ]]; then
        build_time="${BASH_REMATCH[1]}"
        qps="${BASH_REMATCH[2]}"
    elif [[ "$line" =~ avg_recall=([0-9.]+) ]]; then
        recall="${BASH_REMATCH[1]}"
    elif [[ "$line" =~ \[INDEX_STATS\]\ avg_neighbors=([0-9.]+) ]]; then
        avg_neighbors="${BASH_REMATCH[1]}"
    elif [[ "$line" =~ max_disterr= ]] && [[ -n "$dataset" ]]; then
        # End of a benchmark run, write to result
        printf "%-10s %-20s %-12s %-12s %-10s %-10s\n" "$dataset" "$label" "${build_time:--}" "${qps:--}" "${recall:--}" "${avg_neighbors:--}" >> "$RESULT"
        # Reset for next run
        dataset=""
        label=""
        build_time=""
        qps=""
        recall=""
        avg_neighbors=""
    fi
done < "$OUTPUT"

echo "" >> "$RESULT"
echo "=== Done ===" >> "$RESULT"

echo ""
echo "Raw output saved to: $OUTPUT"
echo "Summary results saved to: $RESULT"
echo ""
cat "$RESULT"
