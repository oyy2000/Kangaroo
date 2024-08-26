## inference CLI

CUDA_VISIBLE_DEVICES=1 python -m medusa.inference.cli --model checkpoints/medusa-vicuna-7b-v1.3

## inference

CUDA_VISIBLE_DEVICES=1 python -m medusa_inference


#  medusa safety

CUDA_VISIBLE_DEVICES=3 python -m medusa_safety_inference --temperature 0.7 --run_no 1
