import torch
import json
from transformers import AutoTokenizer


def load_data(path):
    """加载data"""
    with open(path, "r") as fp:
        data = fp.read().strip().split("\n")
    return data


def print_dataset_example(input_input_ids, label_input_ids, tokenizer):
    print("input_ids", input_input_ids)
    print("input_tokens", tokenizer.convert_ids_to_tokens(input_input_ids))
    print("inputs", tokenizer.decode(input_input_ids))
    print("label_ids", label_input_ids)
    print("label_tokens", tokenizer.convert_ids_to_tokens(label_input_ids))
    print("labels", tokenizer.decode(label_input_ids))


class NerCollate:
    def __init__(self, args, tokenizer):
        self.max_source_length = args.max_source_length
        self.max_target_length = args.max_target_length
        self.instruct_column = args.instruct_column
        self.query_column = args.query_column # "query"
        self.response_column = args.response_column # "answer"
        self.ignore_pad_token_for_loss = args.ignore_pad_token_for_loss
        self.history_column = None # "history"
        self.tokenizer = tokenizer
        self.max_seq_length = self.max_source_length + self.max_target_length

    def collate_fn(self, batch):

        model_inputs = {
            "input_ids": [],
            "labels": [],
        }

        for example in batch:
            if isinstance(example, str):
                example = json.loads(example)
            if example[self.query_column] and example[self.response_column]:
                instruct = example[self.instruct_column]
                query, answer = example[self.query_column], example[self.response_column]

                if self.history_column is None:
                    prompt = instruct + "\n" + query  # 不存在历史对话
                else:  # 存在历史对话
                    prompt = ""
                    history = example[self.history_column]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

                a_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False)
                b_ids = self.tokenizer.encode(text=answer, add_special_tokens=False)

                # print_dataset_example(a_ids, b_ids, self.tokenizer)

                # 截断超过长度的输入
                if len(a_ids) > self.max_source_length - 1:
                    a_ids = a_ids[: self.max_source_length - 1]

                if len(b_ids) > self.max_target_length - 2:
                    b_ids = b_ids[: self.max_target_length - 2]

                input_ids = self.tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

                # print(input_ids)
                # print(self.tokenizer.convert_ids_to_tokens(input_ids))
                # print(self.tokenizer.decode(input_ids))

                # context_length = input_ids.index(self.tokenizer.bos_token_id)  # sop
                context_length = len(a_ids) + 1  # common for chatglm and chatglm2
                mask_position = context_length - 1
                labels = [-100] * context_length + input_ids[mask_position + 1:]

                pad_len = self.max_seq_length - len(input_ids)
                input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
                labels = labels + [self.tokenizer.pad_token_id] * pad_len
                if self.ignore_pad_token_for_loss:
                    labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]

                # print(labels)
                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)

        model_inputs["input_ids"] = torch.tensor(model_inputs["input_ids"])
        model_inputs["labels"] = torch.tensor(model_inputs["labels"])

        return model_inputs


if __name__ == "__main__":
    class Args:
        max_source_length = 128
        max_target_length = 128
        instruct_column = "instruct"
        query_column = "query"
        response_column = "answer"
        ignore_pad_token_for_loss = True
        train_path = "data/msra/instruct_data/train.txt"
        model_name = 'THUDM/chatglm-6b-int4'


    args = Args()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    batch_size = 8
    data = load_data(args.train_path)
    print(data[0])

    batch_data = data[:batch_size]
    ner_collate = NerCollate(args, tokenizer)

    batch_data = ner_collate.collate_fn(batch_data)
    print(batch_data["input_ids"][0])
