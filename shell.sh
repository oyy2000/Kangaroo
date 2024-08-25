!python -m evaluation.inference_kangaroo_typical_sampling --task "safety" --subtask "jailbreak" --adapter-path "kangaroo-vicuna-7b-v1.3" --exitlayer 2 --model-path "lmsys/vicuna-7b-v1.3" --threshold 0.6 --temperature 1.0 --steps 6 --model-id "vicuna-7b-v1.3-kangaroo-top_p_0.1_temp_1.0" --bench-name "Kangaroo" --dtype "float16" --do_sample "top_p" --max-new-tokens 1024 --hyper_p 0.1