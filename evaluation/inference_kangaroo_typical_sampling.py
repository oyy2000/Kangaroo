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

from transformers import AutoModelForCausalLM, AutoTokenizer
from kangaroo.kangaroo_model import KangarooModel
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
    for i in range(0, len(questions), chunk_size):
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


def kangaroo_forward(inputs, model, tokenizer, max_new_tokens, do_sample="typical", EARLY_STOP_LAYER = 2, SPECULATIVE_DECODING_STEPS = 6, temperature = 0.7, threshold = 0.6, hyper_k = 2, hyper_p = 0.8, epsilon = 0.3, delta = 0.09):
    print("temperature", temperature)
    context_tokens = inputs.input_ids # 把prompt转换成token
    device = context_tokens.device 
    token_eos = tokenizer.eos_token_id # 1
    batch_size, context_length = context_tokens.shape # batchsize, input_length
    start_index = context_length # 从context_length开始，也就是 prompt的下一个token
    
    global_tokens = torch.ones((batch_size, start_index + max_new_tokens), dtype=torch.long, device=device) * token_eos # 用于存储本轮输出，生成一个全是1的tensor，大小为batchsize, max_length
    global_position_ids = torch.LongTensor([[i for i in range(start_index + max_new_tokens)]]).to(device) # 生成一个从0到max_length的tensor，作为position_ids 
    accept_length_list = [1] # 每次接收小模型token长度的list, 生成一个长度为1的list，值为1

    
    global_tokens[:, :start_index] = context_tokens # 把prompt的token放到global_tokens的前面， : 是一个左闭右开的区间，所以不包含start_index

    # Init KV-chache and sample the first token
    with torch.no_grad():
        position_ids = global_position_ids[:, :start_index] # 给 context_tokens 生成一个position_ids

        # 把context_tokens和position_ids输入到base_model中
        output = model.base_model(context_tokens, position_ids=position_ids, output_hidden_states=True) # output是一个tuple，包含了last_hidden_state, past_key_values, hidden_states
        model.base_model.past_key_values = list(output.past_key_values) # KV-cache, 用于存储每一层的key和value
        hidden_state = output.hidden_states[-1] # hidden_state是最后一层的hidden_state
        logits = output.logits # batchsize, input_length, vocab_size
        global_tokens[:, start_index] = torch.argmax(logits[:, -1, :], dim=-1).item() # 把预测的token放到global_tokens的start_index位置，也就是context_tokens的下一个位置
        hidden_state_early = output.hidden_states[EARLY_STOP_LAYER] # early stopping layer, 用于提前停止的层， 第二层

        # KV-cache for the adapter 
        # 已经有了base_model的KV-cache，现在需要为adapter_model生成KV-cache
        hidden_state, adapter_past_key_values = model.adapter_model.forward_early_stop(inputs_embeds=hidden_state_early[:,:,:], position_ids=global_position_ids[:, :context_length], use_cache=True) 

    total_inference_steps = 0

    with torch.no_grad():
        max_infer_steps = start_index + max_new_tokens
        stop = False
        # 使用draft model进行快速推理
        while start_index < max_infer_steps - 1 - SPECULATIVE_DECODING_STEPS:

            start_index_copy = start_index # 保存start_index的值
            end_index = start_index + 1
            
            # STEP 1: Small model decoding
            for step in range(1 + SPECULATIVE_DECODING_STEPS):
                assert adapter_past_key_values[0][0].shape[2] <= end_index-1, "{} - {}".format(adapter_past_key_values[0][0].shape, end_index-1) 
                in_tokens_small = global_tokens[:, end_index-1:end_index]
                # 如果KV-cache的长度小于end_index-1, 也就是所有的draft token都被接受了
                if adapter_past_key_values[0][0].shape[2] < end_index-1: 
                    # As illustrated in the framework of Kangaroo, once all drafted tokens are accepted, the KV-cache of the last draft token for the adapter is missing.
                    position_ids = global_position_ids[:, start_index-1:end_index]
                    hidden_state_early_last = exited_hidden_states[:,-1:,:]
                else:
                    position_ids = global_position_ids[:, end_index-1:end_index]
                    hidden_state_early_last = None
                
                hidden_state_early = model.base_model.forward_draft_or_large_model(in_tokens_small=in_tokens_small[:,-1:], position_ids=position_ids[:,-1:])
                
                if step==0:
                    exited_hidden_states = None

                exited_hidden_states = hidden_state_early if exited_hidden_states is None else torch.cat([exited_hidden_states, hidden_state_early], dim = 1)
                
                if hidden_state_early_last is not None:
                    hidden_state_early = torch.cat([hidden_state_early_last, hidden_state_early], dim = 1)

                # early exiting 
                if step == SPECULATIVE_DECODING_STEPS or (step > 0 and predict_score < threshold):
                    break

                hidden_state, adapter_past_key_values = model.adapter_model.forward_early_stop(inputs_embeds=hidden_state_early, position_ids=position_ids, past_key_values=adapter_past_key_values, use_cache=True)

                predict_logits = model.head_model(hidden_state[:,-1:,:]).float() 
                global_tokens[:, end_index] = torch.argmax(predict_logits[:, -1, :], dim=-1)
                
                end_index += 1
                predict_score = predict_logits.softmax(dim=-1).max().item()

            # STEP2: Big model inference
            position_ids = global_position_ids[:, start_index:end_index]
            assert model.base_model.past_key_values[EARLY_STOP_LAYER][0].shape[2] == start_index, "{} - {}".format(model.base_model.past_key_values[EARLY_STOP_LAYER][0].shape, start_index)
            assert exited_hidden_states.shape[1] == position_ids.shape[1]
            hidden_state_, hidden_state = model.base_model.forward_draft_or_large_model(in_features_large=exited_hidden_states, position_ids=position_ids)
            
            logits = model.head_model(hidden_state).float() # batchsize, input_length, vocab_size
            output_tokens = torch.argmax(logits[:, :, :], dim=-1) # batchsize, input_length
            output_length = end_index - start_index

            posterior_threshold = epsilon # 0.3
            posterior_alpha = delta # 0.09
            
            if temperature == 0.0:
                print("greedy decoding")
                for i in range(output_length):
                    if i == output_length - 1 or output_tokens[0, i] == token_eos or output_tokens[0, i] != global_tokens[0, start_index + 1 + i]:
                        global_tokens[0, start_index + 1 + i] = output_tokens[0, i]
                        start_index = start_index + 1 + i
                        if output_tokens[0, i] == token_eos:
                            stop = True
                        break
                
                    
            else:
                if do_sample == "typical":
                    for i in range(output_length):
                        if i == output_length - 1 or output_tokens[0, i] == token_eos or output_tokens[0, i] != global_tokens[0, start_index + 1 + i]:
                            # apply typical acceptance here
                            posterior_prob = torch.softmax(logits[:, i, :] / temperature, dim=-1)
                            posterior_entropy = -torch.sum(
                                posterior_prob * torch.log(posterior_prob + 1e-5), dim=-1
                            )  # torch.sum(torch.log(*)) is faster than torch.prod
                            threshold = torch.minimum(
                                torch.ones_like(posterior_entropy) * posterior_threshold,
                                torch.exp(-posterior_entropy) * posterior_alpha,
                            )
                            is_accepted = posterior_prob.max(dim=-1).values > threshold
                            if is_accepted:
                                start_index += 1
                            else:
                                start_index += 1
                                stop = True
                            if output_tokens[0, i] == token_eos:
                                stop = True
                            if stop or start_index >= max_infer_steps:
                                break

                elif do_sample == "typical_sampling":
                    for i in range(output_length):
                        # Calculate posterior probabilities
                        posterior_prob = torch.softmax(logits[:, i, :] / temperature, dim=-1)
                        
                        # Calculate the entropy of the distribution
                        posterior_entropy = -torch.sum(posterior_prob * torch.log(posterior_prob + 1e-5), dim=-1)
                        
                        # Sort probabilities and calculate cumulative probabilities
                        sorted_prob, sorted_indices = torch.sort(posterior_prob, descending=True)
                        cumulative_probs = torch.cumsum(sorted_prob, dim=-1)
                        
                        # Find the smallest k such that the cumulative probability exceeds the threshold
                        threshold = torch.minimum(
                            torch.ones_like(posterior_entropy) * posterior_threshold,
                            torch.exp(-posterior_entropy) * posterior_alpha,
                        )
                        cutoff_index = torch.sum(cumulative_probs < threshold.unsqueeze(-1), dim=-1)
                        
                        # Mask the logits for tokens beyond the cutoff
                        mask = torch.ones_like(posterior_prob)
                        for batch_idx in range(mask.size(0)):
                            mask[batch_idx, cutoff_index[batch_idx]:] = 0
                        
                        # Re-normalize the probabilities
                        filtered_prob = posterior_prob * mask
                        filtered_prob = filtered_prob / filtered_prob.sum(dim=-1, keepdim=True)
                        
                        # Sample from the filtered distribution
                        sampled_index = torch.multinomial(filtered_prob, 1).squeeze(-1)
                        
                        # Set the selected token in global_tokens
                        global_tokens[0, start_index + 1 + i] = sorted_indices[0, sampled_index]
                        start_index = start_index + 1 + i
                        
                        # Check for end-of-sequence token
                        if global_tokens[0, start_index] == token_eos:
                            stop = True
                            break

                elif do_sample == "top_p":
                    # set the temperature
                    logits = logits / temperature
                    probs = F.softmax(logits , dim=-1)  # 将logits转换为概率分布
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)  # 将概率从大到小排序
                    cumulative_probs = sorted_probs.cumsum(dim=-1)  # 计算累积概率
                    top_p_mask = cumulative_probs <= hyper_p  # 找出累积概率小于等于hyper_p的token
                    top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # 确保至少有一个token被选择
                    top_p_mask[..., 0] = True  # 确保第一个token被选择
                    output_top_p_tokens = sorted_indices.masked_select(top_p_mask)  # 选择这些token
    
                    output_lenght = end_index - start_index
                    for i in range(output_lenght):
                        if i == output_lenght-1 or output_tokens[0, i] == token_eos or global_tokens[0, start_index+1+i] not in output_top_p_tokens:
                            global_tokens[0, start_index+1+i] = output_tokens[0, i]
                            start_index = start_index+1+i
                            if output_tokens[0, i] == token_eos:
                                stop = True
                            break


                elif do_sample == "top_k":
                    # Verification for top-k sampling
                    output_topk_tokens = torch.topk(logits,k=hyper_k,dim=-1).indices

                    output_lenght = end_index - start_index
                    for i in range(output_lenght):

                        if i == output_lenght-1 or output_tokens[0, i] == token_eos or global_tokens[0, start_index+1+i] not in output_topk_tokens[0, i, :]:
                            global_tokens[0, start_index+1+i] = output_tokens[0, i]
                            start_index = start_index+1+i
                            if output_tokens[0, i] == token_eos:
                                stop = True
                            break
                # greedy decoding
                else:
                    for i in range(output_length):
                        if i == output_length - 1 or output_tokens[0, i] == token_eos or output_tokens[0, i] != global_tokens[0, start_index + 1 + i]:
                            global_tokens[0, start_index + 1 + i] = output_tokens[0, i]
                            start_index = start_index + 1 + i
                            if output_tokens[0, i] == token_eos:
                                stop = True
                            break
                    

            accept_length_list.append(start_index - start_index_copy)
            hidden_state = hidden_state[:, :output_length-(end_index-start_index), :]

            # 将 basemodel以及adapter model 的 KV-cache 进行更新
            # STEP 4: Post process KV-cache
            if model.base_model.past_key_values[0][0].shape[2] > start_index:
                past_key_values_large_ = []
                for k,v in model.base_model.past_key_values:
                    past_key_values_large_.append((k[:,:,:start_index,:], v[:,:,:start_index,:]))
                model.base_model.past_key_values = past_key_values_large_

            if adapter_past_key_values[0][0].shape[2] > start_index:
                adapter_past_key_values_ = []
                for k,v in adapter_past_key_values:
                    adapter_past_key_values_.append((k[:,:,:start_index,:], v[:,:,:start_index,:]))
                adapter_past_key_values = tuple(adapter_past_key_values_)
                del adapter_past_key_values_
            
            total_inference_steps += 1

            if stop:
                break

    output_ids = global_tokens[0, :start_index+1].tolist()
    new_token = start_index - context_length + 1
    idx = len(accept_length_list) - 1
    return [output_ids], new_token, idx, accept_length_list


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

    args = parser.parse_args()
    
    question_file = f"data/eval_data/{args.task}/{args.subtask}.json"

    model = KangarooModel(args.model_path, args.adapter_path, args, EARLY_STOP_LAYER = args.exitlayer)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.do_sample == "top_k":
        parameter = f"{args.hyper_k}"
    elif args.do_sample == "top_p":
        parameter = f"{args.hyper_p}"
    else:
        parameter = f"epsilon_{args.epsilon}_delta_{args.delta}"        

    model_id = f"vicuna-7b-v1.3-kangaroo-{args.do_sample}_{parameter}_temp_{args.temperature}"
    args.model_id = model_id + "-special"
    answer_file_dir = f"data/{args.bench_name}/kangaroo/{args.model_id}/{args.task}"
    os.makedirs(answer_file_dir, exist_ok=True)
    answer_file_name = f"{args.subtask}.json"
    
    print(f"Output to {answer_file_dir}/{answer_file_name}")

    generation(
        model=model,
        tokenizer=tokenizer,
        forward_func=kangaroo_forward,
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
