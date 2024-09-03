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
import torch.nn.functional as F
from trustllm.task import safety
from trustllm.utils import file_process

from transformers import AutoModelForCausalLM, AutoTokenizer
from kangaroo.adapter import AdapterModel
from transformers.models.llama import LlamaConfig
from fastchat.model import get_conversation_template
from tqdm import tqdm
import random
from transformers import set_seed

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
set_seed(seed)

# For reproducibility in convolution operations, etc.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_quantized_model_and_tokenizer(model_name, adapter_path, device, dtype):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load quantized model
    if dtype == "int8":
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
    elif dtype == "int4":
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_4bit=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=getattr(torch, dtype))
    
    # Load and quantize AdapterModel
    config = LlamaConfig.from_pretrained(os.path.join(adapter_path, "config.json"))
    adapter_model = AdapterModel(config)
    adapter_state_dict = torch.load(os.path.join(adapter_path, "pytorch_model.bin"), map_location="cpu")
    
    # Quantize adapter weights
    for key, value in adapter_state_dict.items():
        if value.dtype == torch.float32:
            adapter_state_dict[key] = value.to(getattr(torch, dtype))
    
    adapter_model.load_state_dict(adapter_state_dict, strict=False)
    adapter_model = adapter_model.eval().to(device)
    
    return model, tokenizer, adapter_model

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def generation(
        model,
        tokenizer,
        model_id,
        question_file,
        answer_file_dir,
        answer_file_name,
        max_new_tokens,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        device,
        adapter_model,
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

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)  # // 2
    for i in range(0, len(questions), chunk_size):
        get_model_answers(
            model,
            tokenizer,
            model_id,
            questions[i: i + chunk_size],
            answer_file,
            max_new_tokens,
            num_choices,
            device=device,
            adapter_model=adapter_model,
            **kwargs,
        )



@torch.inference_mode()
def get_model_answers(
        model,
        tokenizer,
        model_id,
        questions,
        answer_file,
        max_new_tokens,
        num_choices,
        device,
        adapter_model,
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
            cur_accept_lengths_tree = []
            # torch.manual_seed(i)
            conv = get_conversation_template("vicuna")
            turns = []
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
                output_ids = generate_sequence(
                    inputs,
                    model,
                    adapter_model,
                    tokenizer,
                    max_new_tokens,
                    device=device,
                    **kwargs,
                )
                torch.cuda.synchronize()
                total_time = time.time() - start_time
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
            wall_time.append(total_time)
            conv.messages[-1][-1] = output
            # torch.cuda.empty_cache()
            # choices.append({"index": i, "turns": turns, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time,
            #                 "accept_lengths": cur_accept_lengths_tree})
            question['res'] = output
            question['accept_lengths'] = cur_accept_lengths_tree
            question['wall_time'] = total_time
            wall_time_list.append(total_time)
            save_json(questions, answer_file)
        # else:
            # print("Skip question: ", question["prompt"])
            # accept_lengths_tree.extend(question['accept_lengths'])
            # compute the average wall time of all questions
            # wall_time_list.append(question['wall_time'])
    
    print("#Mean wall time: ", np.mean(wall_time_list))
    print("#Mean accepted tokens: ", np.mean(accept_lengths_tree))
    # create a log file if not exists
    log_file_path = os.path.join(answer_file_dir, f"log_{args.subtask}.txt")
    
        
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


def calculate_entropy(logits):
    # Softmax to probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    # Calculate entropy
    entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
    return entropy

def fuse_layers(layer_output, final_logits, model, temperature=1.0, alpha=0.1):
    # Apply the language model head to the intermediate layer output
    layer_logits = model.lm_head(layer_output)

    # Calculate alpha using sigmoid function
    # alpha = 0.01 # torch.sigmoid(-entropy) + 0.5
    
    # Apply temperature scaling
    layer_logits /= (temperature + 1e-5)
    final_logits /= (temperature + 1e-5)
    
    # Compute softmax probabilities
    layer_probs = F.softmax(layer_logits, dim=-1)
    final_probs = F.softmax(final_logits, dim=-1)
    
    # Fuse probabilities
    fused_probs = alpha * layer_probs + (1 - alpha) * final_probs
    
    return fused_probs

# Modify the generate_with_fusion function to use the AdapterModel
def generate_with_fusion(model, adapter_model, tokenizer, input_ids, max_new_tokens, temperature=1.0, batch_size=1, fusion_layer=2, alpha=0.1, **kwargs):
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    batch_size = min(batch_size, input_ids.shape[0])

    past_key_values = None
    generated_tokens = []

    for _ in range(0, max_new_tokens, batch_size):
        with torch.no_grad():
            outputs = model(input_ids, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)

        past_key_values = outputs.past_key_values

        layer_output = outputs.hidden_states[fusion_layer]
        final_logits = outputs.logits  # Use the logits directly from the model output

        # Use AdapterModel to refine layer 5 output
        refined_fusion_layer_output = adapter_model(inputs_embeds=layer_output)

        fused_probs = fuse_layers(refined_fusion_layer_output, final_logits, model, temperature=temperature, alpha=alpha)
      
        next_token_probs = fused_probs[:, -1, :]
        next_tokens = torch.multinomial(next_token_probs, num_samples=1)

        generated_tokens.append(next_tokens)
        input_ids = next_tokens

        if (next_tokens == tokenizer.eos_token_id).all():
            break

    return torch.cat(generated_tokens, dim=1)

def generate_sequence(inputs, model, adapter_model, tokenizer, max_new_tokens, device, temperature=0.7, batch_size=10, **kwargs):

    input_ids = inputs.input_ids.to(device)

    generated_sequence = generate_with_fusion(model, adapter_model, tokenizer, input_ids, max_new_tokens, temperature, batch_size, **kwargs)

    full_sequence = torch.cat([input_ids, generated_sequence], dim=1)

    # Find the first occurrence of the EOS token for each sequence
    eos_positions = (full_sequence == tokenizer.eos_token_id).float().argmax(dim=1)

    # Truncate sequences at EOS token
    for i in range(full_sequence.shape[0]):
        if eos_positions[i] > 0:
            full_sequence[i, eos_positions[i]+1:] = tokenizer.pad_token_id

    return full_sequence


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
        required=True,
    )
    parser.add_argument("--model-id", type=str)
    parser.add_argument(
        "--bench-name",
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
    
    parser.add_argument(
        "--fusion-layer",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--alpha",
        type=float,
    )

    parser.add_argument(
        "--quantization",
        type=str,
        default="float16",
        choices=["float32", "float16", "bfloat16", "int8", "int4"],
        help="Quantization type for the model.",
    )
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "vicuna-7b-v1.3"
    model, tokenizer, adapter_model = load_quantized_model_and_tokenizer(
        args.model_path, 
        args.adapter_path, 
        device, 
        args.quantization
    )
    
    question_file = f"data/eval_data/{args.task}/{args.subtask}.json"

    # # Initialize and load AdapterModel
    # adapter_model_path = args.adapter_path
    # config = LlamaConfig.from_pretrained(os.path.join(adapter_model_path, "config.json"))
    # adapter_model = AdapterModel(config)
    # adapter_model.load_state_dict(torch.load(os.path.join(adapter_model_path, "pytorch_model.bin"), map_location="cpu"), strict=False)
    # adapter_model = adapter_model.eval().to(device)
    
   
    if args.do_sample == "top_k":
        parameter = f"{args.hyper_k}"
    elif args.do_sample == "top_p":
        parameter = f"{args.hyper_p}"
    else:
        parameter = f"epsilon_{args.epsilon}_delta_{args.delta}"        

    model_id = f"vicuna-7b-v1.3-df2-temp-{args.temperature}-layer-{args.fusion_layer}-alpha-{args.alpha}"
    args.model_id = model_id
    answer_file_dir = f"data/{args.bench_name}/dynamic_fusion/{args.model_id}/{args.task}"
    os.makedirs(answer_file_dir, exist_ok=True)
    answer_file_name = f"{args.subtask}.json"
    
    print(f"Output to {answer_file_dir}/{answer_file_name}")

    generation(
        model=model,
        tokenizer=tokenizer,
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
        EARLY_STOP_LAYER=args.exitlayer,
        fusion_layer=args.fusion_layer,
        device=device,  # Pass the device to the generation function
        adapter_model=adapter_model,
        alpha=args.alpha
    )
