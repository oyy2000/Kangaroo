"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import torch
import json
import os
import time
import numpy as np
import shortuuid
import torch.nn.functional as F
from trustllm.task import safety
from trustllm.utils import file_process

from fastchat.utils import str_to_torch_dtype
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from fastchat.model import get_conversation_template
from tqdm import tqdm
import random

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
set_seed(seed)

# For reproducibility in convolution operations, etc.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def generation(
        model,
        tokenizer,
        forward_func,
        model_id,
        question_file,
        answer_file_dir,
        answer_file_name,
        max_new_tokens,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        **kwargs,
):
    with open(question_file) as f:
        original_data = json.load(f)
    answer_file = os.path.join(answer_file_dir, answer_file_name)
    if os.path.exists(answer_file):
        with open(answer_file, 'r') as f:
            res_data = json.load(f)
    else:
        res_data = original_data

    questions = res_data

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0

    get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)  # // 2
    ans_handles = []
    for i in range(0, len(questions), chunk_size):

        ans_handles.append(
            get_answers_func(
                model,
                tokenizer,
                forward_func,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_tokens,
                num_choices,
                **kwargs,
            )
        )



@torch.inference_mode()
def get_model_answers(
        model,
        tokenizer,
        forward_func,
        model_id,
        questions,
        answer_file,
        max_new_tokens,
        num_choices,
        temperature,
        do_sample,
        hyper_k,
        hyper_p,
        **kwargs,
):

    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    accept_lengths_tree = []
    wall_time_list = []
    for question in tqdm(questions):
        # Dump answers
        if "res" not in question or not question['res']:
            choices = []
            cur_accept_lengths_tree = []
            # torch.manual_seed(i)
            conv = get_conversation_template("vicuna")
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            q_prompt = question["prompt"]
            conv.append_message(conv.roles[0], q_prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
            input_ids = inputs.input_ids
            total_time = 0
            try:
                torch.cuda.synchronize()
                start_time = time.time()
                output_ids, new_token, idx, accept_length_tree = forward_func(
                    inputs,
                    model,
                    tokenizer,
                    max_new_tokens,
                    temperature,
                    **kwargs,
                )
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                accept_lengths_tree.extend(accept_length_tree)
                output_ids = output_ids[0][len(input_ids[0]):]
            except RuntimeError as e:
                print("ERROR when forwarding question: ", question["prompt"])
                output = "ERROR"

            try:
                if conv.stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in conv.stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                if conv.stop_str and output.find(conv.stop_str) > 0:
                    output = output[: output.find(conv.stop_str)]
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                if conv.name == "xgen" and output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()
            except RuntimeError as e:
                print("ERROR question: ", question["prompt"])
                output = "ERROR"

            turns.append(output)
            idxs.append(int(idx))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            cur_accept_lengths_tree.extend(accept_length_tree)
            conv.messages[-1][-1] = output
            # torch.cuda.empty_cache()
            # choices.append({"index": i, "turns": turns, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time,
            #                 "accept_lengths": cur_accept_lengths_tree})
            question['res'] = output
            question['accept_lengths'] = cur_accept_lengths_tree
            question['wall_time'] = total_time
            wall_time_list.append(total_time)
            save_json(questions, answer_file)
        else:
            # print("Skip question: ", question["prompt"])
            accept_lengths_tree.extend(question['accept_lengths'])
            # compute the average wall time of all questions
            wall_time_list.append(question['wall_time'])
    
    print("#Mean wall time: ", np.mean(wall_time_list))
    print("#Mean accepted tokens: ", np.mean(accept_lengths_tree))
    # create a log file if not exists
    log_file_path = os.path.join(answer_file_dir, f"log_{args.subtask}.txt")

    # model_path = 'generation_results/kangaroo-vicuna-7b-topp/' # kangaroo-vicuna-7b
    # dir_path = model_path + task_type

    # if do_sample == "top_k":
    #     parameter = f"{hyper_k}"
    # elif do_sample == "top_p":
    #     parameter = f"{hyper_p}"
    # else:
        # parameter = f"epsilon_{epsilon}_delta_{delta}"        

    with open(log_file_path, 'w') as f:
        evaluator = safety.SafetyEval()
        jailbreak_data = file_process.load_json(answer_file)
        jailbreak_score = evaluator.jailbreak_eval(jailbreak_data, eval_type='total')
        print(f"jailbreak score: {jailbreak_score}\n")
        f.write(f"Model: {args.model_id}\n")
        f.write(f"Total questions: {len(questions)}\n")
        f.write(f"Mean wall time: {np.mean(wall_time_list)}\n")
        f.write(f"Mean accepted tokens: {np.mean(accept_lengths_tree)}\n")  
        f.write(f"jailbreak score: {jailbreak_score}\n")
          

def baseline_forward(inputs, model, tokenizer, max_new_tokens, temperature=0.0, **kwargs):
    input_ids = inputs.input_ids
    
    if args.temperature > 0:
        output_ids = model.generate(
        input_ids,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    else:
        output_ids = model.generate(
        input_ids,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )
    
    new_token = len(output_ids[0][len(input_ids[0]):])
    idx = new_token - 1
    accept_length_list = [1] * new_token
    return output_ids, new_token, idx, accept_length_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
    )
    parser.add_argument(
        "--model-id", 
        type=str, 
        # required=True
        )
    parser.add_argument(
        "--bench-type",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="The temperature for sampling.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="The threshold for fallback.",
    )

    parser.add_argument(
        "--exitlayer",
        type=int,
        default=2,
        help="The exitlayer.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=6,
        help="The number of GPUs per model.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )

    parser.add_argument(
        "--subtask",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--do_sample",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--hyper_k",
        type=int,
    )

    parser.add_argument(
        "--hyper_p",
        type=float,
    )

    parser.add_argument(
        "--epsilon",
        type=float,
    )

    parser.add_argument(
        "--delta",
        type=float,
    )

    args = parser.parse_args()
    
    question_file = f"data/eval_data/{args.task}/{args.subtask}.json"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.do_sample == "top_k":
        parameter = f"{args.hyper_k}"
    elif args.do_sample == "top_p":
        parameter = f"{args.hyper_p}"
    else:
        parameter = f"epsilon_{args.epsilon}_delta_{args.delta}"        

    model_id = f"baseline-vicuna-7b-v1.3-temp_{args.temperature}"
    args.model_id = model_id
    answer_file_dir = f"data/{args.bench_type}/{args.model_id}/{args.task}"
    os.makedirs(answer_file_dir, exist_ok=True)
    answer_file_name = f"{args.subtask}.json"
    
    print(f"Output to {answer_file_dir}/{answer_file_name}")

    generation(
        model=model,
        tokenizer=tokenizer,
        forward_func=baseline_forward,
        model_id=args.model_id,
        question_file=question_file,
        answer_file_dir=answer_file_dir,
        answer_file_name=answer_file_name,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        do_sample=args.do_sample,
        temperature=args.temperature,
        hyper_k=args.hyper_k,
        hyper_p=args.hyper_p,
        epsilon=args.epsilon,
        delta=args.delta,
        SPECULATIVE_DECODING_STEPS=args.steps,
        EARLY_STOP_LAYER=args.exitlayer
    )
