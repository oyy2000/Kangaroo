## Kangaroo 

## Kangaroo
CUDA_VISIBLE_DEVICES=0 python -m evaluation.inference_kangaroo --adapter-path "cache/CKPT/kangaroo-vicuna-7b-v1.3" --exitlayer 2 --model-path "cache/CKPT/vicuna-7b-v1.3" --threshold 0.6 --steps 6 --model-id "vicuna-7b-v1.3-kangaroo-thres-0.6-steps-6-float16" --bench-name "Kangaroo" --dtype "float16"

## Kangaroo   safety_evaluation

CUDA_VISIBLE_DEVICES=1 python -m evaluation.inference_kangaroo_safety --adapter-path "cache/CKPT/kangaroo-vicuna-7b-v1.3" --exitlayer 2 --model-path "cache/CKPT/vicuna-7b-v1.3" --threshold 0.6 --steps 6 --model-id "vicuna-7b-v1.3-kangaroo-thres-0.6-steps-6-float16" --bench-name "Kangaroo_safety" --dtype "float16"

# Vanilla vicuna safety

CUDA_VISIBLE_DEVICES=2 python -m evaluation.inference_baseline_safety --model-path "cache/CKPT/vicuna-7b-v1.3" --model-id "vicuna-7b-v1.3-vanilla-float16-temp-0.0" --bench-name "Kangaroo_safety" --temperature 0.0 --dtype "float16" 




# Kangaroo case-study

CUDA_VISIBLE_DEVICES=0 python -m evaluation.kangaroo_safety_casestudy --adapter-path "cache/CKPT/kangaroo-vicuna-7b-v1.3" --exitlayer 2 --model-path "cache/CKPT/vicuna-7b-v1.3" --threshold 0.6 --steps 6 --model-id "kangaroo_casestudy" --bench-name "Kangaroo_safety" --dtype "float16"
