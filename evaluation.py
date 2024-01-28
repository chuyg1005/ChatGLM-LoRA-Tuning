import os
import torch
import json
from pprint import pprint
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import argparse
from tqdm import tqdm


def compute_f1_score(preds, golds, relations):
    ground_true = 0
    predict_cnt = 0
    gold_cnt = 0
    for pred, gold in zip(preds, golds):
        pred_triples = pred.split("\n")
        gold_triples = gold.split("\n")
        pred_triples = [triple.split("\t") for triple in pred_triples if len(triple.split("\t")) == 3]
        gold_triples = [triple.split("\t") for triple in gold_triples]

        print(pred_triples)
        print(gold_triples)

        gt = 0
        for pred_triple in pred_triples:
            if pred_triple in gold_triples:
                gt += 1

        ground_true += gt
        predict_cnt += len(pred_triples)
        gold_cnt += len(gold_triples)

    precision = ground_true * 1.0 / predict_cnt
    recall = ground_true * 1.0 / gold_cnt
    micro_f1 = 2 * precision * recall / (precision + recall)
    return micro_f1


def main(cmd_args):
    train_args_path = "./checkpoint/{}/train_deepspeed/{}/{}/train_args.json".format(cmd_args.data_name,
                                                                                     cmd_args.data_dir,
                                                                                     cmd_args.model_name)
    test_path = "./data/{}/instruct_data/{}/test.txt".format(cmd_args.data_name, cmd_args.data_dir)
    test_data = open(test_path, "r").readlines()
    with open(train_args_path, "r") as fp:
        args = json.load(fp)
        print(args)

    config = AutoConfig.from_pretrained(args["model_dir"], trust_remote_code=True)
    pprint(config)
    tokenizer = AutoTokenizer.from_pretrained(args["model_dir"], trust_remote_code=True)

    model = AutoModel.from_pretrained(args["model_dir"], trust_remote_code=True).half().cuda()
    model = model.eval()
    # 非zero-shot，加载lora参数
    if not cmd_args.zero_shot:
        model = PeftModel.from_pretrained(model, os.path.join(args["save_dir"]), torch_dtype=torch.float32,
                                          trust_remote_code=True)
    model.half().cuda()
    model.eval()
    relations = ["instance of", "symbol of", "part of", "similar to", "working on", "capital of", "located in",
                 "close to", "leader of", "industry peer", "relative of", "agree with", "disagree with",
                 "belongs to", "belong to", "president of", "founder of", "member of"]

    os.makedirs("./cache", exist_ok=True)
    if not cmd_args.zero_shot:
        cache_file = "./cache/{}_{}_{}.jsonl".format(cmd_args.data_name, cmd_args.data_dir, cmd_args.model_name)
    else:
        cache_file = "./cache/{}_{}_{}_zero_shot.jsonl".format(cmd_args.data_name, cmd_args.data_dir,
                                                               cmd_args.model_name)
    preds, golds = [], []

    if os.path.exists(cache_file):
        print("load from cache file: ", cache_file)
        for line in open(cache_file, "r"):
            item = json.loads(line)
            preds.append(item["pred"])
            golds.append(item["gold"])

    with open(cache_file, "a") as fp:
        for i in range(len(preds), len(test_data)):
            line = test_data[i]
            item = json.loads(line)
            instruct, query, answer = item["instruct"], item["query"], item["answer"]
            inp = instruct + "\n" + query
            inp = inp[:cmd_args.max_length]  # 适度截断
            print("start chat...")
            response, history = model.chat(tokenizer, inp, history=[])
            print("ChatRelationClassifier >>> ", response)
            print("answer >>> ", answer)
            print(f"[{i}/{len(test_data)}]" + "=" * 100)
            preds.append(response)
            golds.append(answer)
            fp.write(json.dumps({"pred": response, "gold": answer}) + "\n")
            fp.flush()

    f1 = compute_f1_score(preds, golds, relations)
    print("f1 score: ", f1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default='mixre')
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="chatglm2-6b-32k")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--zero_shot", action="store_true")
    cmd_args = parser.parse_args()
    main(cmd_args)
