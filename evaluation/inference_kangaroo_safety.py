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

from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from kangaroo.kangaroo_model import KangarooModel

def run_eval(
        model,
        tokenizer,
        forward_func,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_tokens,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        **kwargs,
):
    questions = load_questions(question_file, question_begin, question_end)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        import ray
        ray.init()
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
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

    if use_ray:
        ray.get(ans_handles)


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
    for question in tqdm(questions):
        choices = []
        for i in range(num_choices):
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
                print("ERROR when forwarding question ID: ", question["question_id"])
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
                print("ERROR question ID: ", question["question_id"])
                output = "ERROR"

            turns.append(output)
            idxs.append(int(idx))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            cur_accept_lengths_tree.extend(accept_length_tree)
            conv.messages[-1][-1] = output
            # torch.cuda.empty_cache()
            choices.append({"index": i, "turns": turns, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time,
                            "accept_lengths": cur_accept_lengths_tree})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "label": question["label"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
                "res": output,
                "prompt": q_prompt,
                "source": question["source"]
            }
            fout.write(json.dumps(ans_json) + "\n")
    print("#Mean accepted tokens: ", np.mean(accept_lengths_tree))


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


def kangaroo_forward(inputs, model, tokenizer, max_new_tokens, do_sample=False, max_length = 2048, EARLY_STOP_LAYER = 2, SPECULATIVE_DECODING_STEPS = 6, threshold = 0.6):
    context_tokens = inputs.input_ids
    device = context_tokens.device
    token_eos = tokenizer.eos_token_id
    batch_size, context_length = context_tokens.shape
    global_tokens = torch.ones((batch_size, max_length), dtype=torch.long, device=device) * token_eos
    global_position_ids = torch.LongTensor([[i for i in range(max_length)]]).to(device)
    accept_length_list = [1]

    start_index = context_length
    global_tokens[:, :start_index] = context_tokens

    # Init KV-chache and sample the first token
    with torch.no_grad():
        position_ids = global_position_ids[:, :start_index]
        output = model.base_model(context_tokens, position_ids=position_ids, output_hidden_states=True)
        model.base_model.past_key_values = list(output.past_key_values)
        hidden_state = output.hidden_states[-1]
        logits = output.logits # batchsize, input_length, vocab_size
        global_tokens[:, start_index] = torch.argmax(logits[:, -1, :], dim=-1).item()
        hidden_state_early = output.hidden_states[EARLY_STOP_LAYER]

        # KV-cache for the adapter
        hidden_state, adapter_past_key_values = model.adapter_model.forward_early_stop(inputs_embeds=hidden_state_early[:,:,:], position_ids=global_position_ids[:, :context_length], use_cache=True) 

    total_inference_steps = 0

    with torch.no_grad():
        max_infer_steps = min(max_length, start_index + max_new_tokens)
        stop = False

        while start_index < max_infer_steps - 1 - SPECULATIVE_DECODING_STEPS:

            start_index_copy = start_index
            end_index = start_index + 1
            
            # STEP 1: Small model decoding
            for step in range(1 + SPECULATIVE_DECODING_STEPS):
                assert adapter_past_key_values[0][0].shape[2] <= end_index-1, "{} - {}".format(adapter_past_key_values[0][0].shape, end_index-1)
                in_tokens_small = global_tokens[:, end_index-1:end_index]
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
            output_tokens = torch.argmax(logits[:, :, :], dim=-1)

            # Verification for greedy decoding
            output_lenght = end_index - start_index
            for i in range(output_lenght):
                if i == output_lenght-1 or output_tokens[0, i] == token_eos or output_tokens[0, i] != global_tokens[0, start_index+1+i]:
                    global_tokens[0, start_index+1+i] = output_tokens[0, i]
                    start_index = start_index+1+i
                    if output_tokens[0, i] == token_eos:
                        stop = True
                    break

            accept_length_list.append(start_index - start_index_copy)
            hidden_state = hidden_state[:, :output_lenght-(end_index-start_index), :]

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
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end",
        type=int,
        help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
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
        "--threshold",
        type=float,
        default=0.4,
        help="The temperature for medusa sampling.",
    )
    parser.add_argument(
        "--exitlayer",
        type=int,
        default=2,
        help="The temperature for medusa sampling.",
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

    args = parser.parse_args()

    question_file = f"jailbreak.jsonl"

    model = KangarooModel(args.model_path, args.adapter_path, args, EARLY_STOP_LAYER = args.exitlayer)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    do_sample = True

    assert not args.answer_file
    os.makedirs(f"data/{args.bench_name}/{args.model_id}", exist_ok=True)

    for run in range(1):
        answer_file = f"data/{args.bench_name}/{args.model_id}/{run}_safety.jsonl"
        print(f"Output to {answer_file}")

        run_eval(
            model=model,
            tokenizer=tokenizer,
            forward_func=kangaroo_forward,
            model_id=args.model_id,
            question_file=question_file,
            question_begin=args.question_begin,
            question_end=args.question_end,
            answer_file=answer_file,
            max_new_tokens=args.max_new_tokens,
            num_choices=args.num_choices,
            num_gpus_per_model=args.num_gpus_per_model,
            num_gpus_total=args.num_gpus_total,
            do_sample=do_sample,
            threshold=args.threshold,
            SPECULATIVE_DECODING_STEPS=args.steps,
            EARLY_STOP_LAYER=args.exitlayer
        )

        reorg_answer_file(answer_file)