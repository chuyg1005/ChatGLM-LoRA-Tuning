import json
import argparse
import os
from tqdm import tqdm
from hashlib import md5


def load_data(path):
    """加载data"""
    with open(path, "r") as fp:
        data = json.load(fp)
    return data


def compute_sign(paragraph):
    return md5(paragraph.encode()).hexdigest()


def save_data(data, save_path):
    """保存data"""
    with open(save_path, "w") as fp:
        for item in data:
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")


def extract_triples_with_sent_ids(item, sent_ids):
    triples = []
    mentions = [entity[0]['name'] for entity in item['vertexSet']]
    mention_sent_ids = [entity[0]['sent_id'] for entity in item['vertexSet']]
    for label in item['labels']:
        head_sent_id, tail_sent_id = mention_sent_ids[label['h']], mention_sent_ids[label['t']]
        if head_sent_id in sent_ids and tail_sent_id in sent_ids:
            relation = ' '.join(label['r'].split('_')[:-2])
            head, tail = mentions[label['h']], mentions[label['t']]
            triples.append((head, relation, tail))
    return triples


def process_item(item, lang, split='train'):
    """处理单个item"""
    if lang == 'zh':
        sents = [''.join(sent) for sent in item['sents']]
    else:
        sents = [' '.join(sent) for sent in item['sents']]
    instruct = "Relation extraction means identifying the relations and the entity pairs that have these relations in a sentencce. Given a sentence, you must output the (subject entity, relation, object entity) triples in it.\n"
    instruct += "Now possible relations are as follows: [instance of, symbol of, part_of, similar to, working on, "\
                "capital of, located in, close to, leader of, industry peer, relative of, agree with, disagree with, belong to, president of, founder of, member of, belongs to].\n"
    mentions = [entity[0]['name'] for entity in item['vertexSet']]
    mention_sent_ids = [entity[0]['sent_id'] for entity in item['vertexSet']]
    if split == 'train':
        paragraph_signs = set()
        for label in item['labels']:
            # 保留两个实体之间的句子
            head_sent_id, tail_sent_id = mention_sent_ids[label['h']], mention_sent_ids[label['t']]
            begin, end = min(head_sent_id, tail_sent_id), max(head_sent_id, tail_sent_id)
            if begin != end:
                paragraph = sents[begin] + ' ' + sents[end]
                sent_ids = [begin, end]
            else:
                paragraph = sents[begin]
                sent_ids = [begin]
            sign = compute_sign(paragraph)
            if sign in paragraph_signs: continue
            paragraph_signs.add(sign)
            query = f"Now the given sentence is: {paragraph}\n"
            triples = extract_triples_with_sent_ids(item, sent_ids)
            entities = ','.join(['\t'.join([triple[0], triple[2]]) for triple in triples])
            query += "Now the given entities are: {}\n".format(entities)
            query += "You should direct output relation triples like 'head_entity\trelation\ttail_entity', and multiple triples are separated by '\n'"

            answer = '\n'.join([f"{triple[0]}\t{triple[1]}\t{triple[2]}" for triple in triples])

            yield {'instruct': instruct, 'query': query, 'answer': answer}
    else:
        paragraph = ' '.join(sents)
        sent_ids = list(range(len(sents)))
        query = f"Now the given sentence is: {paragraph}\n"
        triples = extract_triples_with_sent_ids(item, sent_ids)
        entities = ','.join(['\t'.join([triple[0], triple[2]]) for triple in triples])
        query += "Now the given entities are: {}\n".format(entities)
        query += "You should output like 'head_entity\trelationt\ttail_entity', and multiple triples are separated by '\n'"
        answer = '\n'.join([f"{triple[0]}\t{triple[1]}\t{triple[2]}" for triple in triples])
        yield {'instruct': instruct, 'query': query, 'answer': answer}


def process_data(data, lang, split='train'):
    new_data = []
    for item in tqdm(data):
        for new_item in process_item(item, lang, split):
            new_data.append(new_item)
    print("split: {}, data size: {}".format(split, len(new_data)))
    return new_data


def main(args):
    data_dir = args.data_dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    lang = args.lang

    train_data = load_data(os.path.join(data_dir, "train_annotated.json"))
    dev_data = load_data(os.path.join(data_dir, "dev.json"))
    test_data = load_data(os.path.join(data_dir, "test.json"))

    train_data = process_data(train_data, lang, split='train')
    dev_data = process_data(dev_data, lang, split='dev')
    test_data = process_data(test_data, lang, split='test')

    save_data(train_data, os.path.join(save_dir, "train.txt"))
    save_data(dev_data, os.path.join(save_dir, "dev.txt"))
    save_data(test_data, os.path.join(save_dir, "test.txt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)

    args = parser.parse_args()
    main(args)
