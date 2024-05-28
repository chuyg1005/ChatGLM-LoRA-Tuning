import os
import json
import random
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm


def load_data(path):
    return json.load(open(path, "r"))


def main(args):
    data_dir = args.data_dir
    output_path = args.output_path

    en_data = load_data(os.path.join(data_dir, "en", "train_annotated.json"))
    zh_data = load_data(os.path.join(data_dir, "zh", "train_annotated.json"))
    de_data = load_data(os.path.join(data_dir, "de", "train_annotated.json"))
    jp_data = load_data(os.path.join(data_dir, "jp", "train_annotated.json"))

    new_data = []

    for i in tqdm(range(len(en_data))):
        en_mentions = [random.choice(entity)['name'] for entity in en_data[i]['vertexSet']]
        zh_mentions = [random.choice(entity)['name'] for entity in zh_data[i]['vertexSet']]
        de_mentions = [random.choice(entity)['name'] for entity in de_data[i]['vertexSet']]
        jp_mentions = [random.choice(entity)['name'] for entity in jp_data[i]['vertexSet']]

        if not (len(en_mentions) == len(zh_mentions)
                and len(de_mentions) == len(jp_mentions)
                and len(en_mentions) == len(de_mentions)):
            print(
                f"mention length not equal: {len(en_mentions)}, {len(zh_mentions)}, {len(de_mentions)}, {len(jp_mentions)}")
            continue

        mentions = [en_mentions, zh_mentions, de_mentions, jp_mentions]

        for _ in range(100):
            triples = []
            entity_num = len(en_mentions) # 实体数量
            indices = np.random.choice(4, entity_num, replace=True)
            entity_mention = [mentions[j][i] for i, j in enumerate(indices)]
            for labels in en_data[i]['labels']:
                relation = labels['r']
                head = labels['h']
                tail = labels['t']

                # 可放回采样
                # indices = np.random.choice(4, 2, replace=True)
                triples.append((entity_mention[head], relation, entity_mention[tail]))

            new_data.append(triples)

    with open(output_path, "w") as f:
        for triples in new_data:
            contents = []
            for triple in triples:
                rela = ' '.join(triple[1].split('_')[:-2])
                contents.append(f'{triple[0]}##{rela}##{triple[2]}.')

            if len(contents) > 0:
                f.write(' '.join(contents) + '\n')
        # json.dump(new_data, f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_path", type=str, default="data/train_for_bert_lora.txt")

    args = parser.parse_args()
    main(args)
