#!/bin/bash
# 定义温度列表
temperatures="1.1 1.2 1.3 1.4 1.5 1.6 1.8"

# 遍历温度列表
for temp in $temperatures
do
    echo "Running with temperature: $temp"
    
    CUDA_VISIBLE_DEVICES=2 python -m evaluation.inference_kangaroo_MTBench \
        --model-path "vicuna-7b-v1.3" \
        --adapter-path "kangaroo-vicuna-7b-v1.3" \
        --exitlayer 2 \
        --model-id "kangaroo-vicuna-7b-v1.3" \
        --threshold 0.6 \
        --temperature "$temp" \
        --steps 6 \
        --bench-name "mt_bench" \
        --dtype "float16" \
        --max-new-token 256 \
        --max-length 512 \
        --do_sample "top_p" \
        --hyper_p 0.5

    echo "Finished running with temperature: $temp"
    echo "----------------------------------------"
done

echo "All temperature variations completed."
# python queue_script.py --bench_type TrustLLM --model_type Kangaroo --GPU_number 4
# 
# python queue_script.py --bench_type TrustLLM --model_type Baseline --GPU_number 8
# 
# python queue_script.py --bench_type MTBench --model_type Baseline --GPU_number 4
# python queue_script.py --bench_type TrustLLM --model_type Medusa --GPU_number 8
# python queue_script.py --bench_type MTBench --model_type Kangaroo --GPU_number 4
