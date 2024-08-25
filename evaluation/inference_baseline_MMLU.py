import os
import argparse
from fastchat.utils import str_to_torch_dtype
from evaluation.categories import subcategories, categories
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import pandas as pd

# 设置随机种子
torch.manual_seed(0)
np.random.seed(0)

choices = ["A", "B", "C", "D"]

def baseline_forward(inputs, model, tokenizer, max_new_tokens, temperature=0.0, do_sample=False):
    input_ids = inputs
    output_ids = model.generate(
        input_ids,
        do_sample=do_sample,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    return output_ids


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s.strip()


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def run_eval(args, subject, model, tokenizer, dev_df, test_df, do_sample):
    cors = []
    all_probs = []
    preds = []
    answers = choices[: test_df.shape[1] - 2]

    for i in range(test_df.shape[0]):
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        while input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        label = test_df.iloc[i, test_df.shape[1] - 1]
        MAX_NEW_TOKENS = 5
        output_ids = baseline_forward(
            input_ids, model, tokenizer, MAX_NEW_TOKENS, args.temperature, do_sample
        )
        new_tokens_logits = output_ids[0, input_ids.shape[-1] :]
        new_tokens = tokenizer.decode(new_tokens_logits, skip_special_tokens=True)
        
        probs = []
        pred = new_tokens[0]
        preds.append(pred)
        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)
    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs, preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
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
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for sampling.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing the data."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Directory to save the results."
    )
    parser.add_argument(
        "--ntrain",
        type=int,
        default=5,
        help="Number of training examples to use."
    )

    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.temperature > 0:
        do_sample = True
    else:
        do_sample = False

    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )
    model_id = "Vicuna-7B-baseline_temp_{}".format(args.temperature)
    args.model_id = model_id
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    results_dir = os.path.join(args.save_dir, "results_{}".format(args.model_id))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    
    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )
        
        cors, acc, probs, preds = run_eval(args, subject, model, tokenizer, dev_df, test_df, do_sample)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(args.model_id)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(args.model_id, choice)] = probs[:, j]
        test_df["{}_preds".format(args.model_id)] = preds
        test_df.to_csv(
            os.path.join(
                results_dir, "{}.csv".format(subject)
            ),
            index=None,
        )

    accuracy_file = os.path.join(results_dir, "accuracy_results.txt")
    with open(accuracy_file, "w") as f:
        f.write("Subject-wise, Subcategory-wise, and Category-wise Accuracy Results\n")
        f.write("="*50 + "\n\n")
        
        for subcat in subcat_cors:
            subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
            # subcat_acc = np.random.rand()
            print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))
            f.write("Average accuracy {:.3f} - {}\n".format(subcat_acc, subcat))
        
        for cat in cat_cors:
            cat_acc = np.mean(np.concatenate(cat_cors[cat]))
            # cat_acc = np.random.rand()
            print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
            f.write("Average accuracy {:.3f} - {}\n".format(cat_acc, cat))
            
        weighted_acc = np.mean(np.concatenate(all_cors))
        print("Average accuracy: {:.3f}".format(weighted_acc))
        f.write("Average accuracy: {:.3f}\n".format(weighted_acc))