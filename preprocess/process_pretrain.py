import json
import argparse
import os


def load_data(path):
    """加载data"""
    with open(path, "r") as fp:
        data = json.load(fp)
    return data


def save_data(data, save_path):
    """保存data"""
    with open(save_path, "w") as fp:
        for item in data:
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")



def process_triples(triples):
    instruct = "Relation extraction means identifying the relations and the entity pairs that have these relations in a sentencce. Given a sentence, you must output the (subject entity, relation, object entity) triples in it.\n"
    instruct += "Now possible relations are as follows: [instance of, symbol of, part_of, similar to, working on, "\
                "capital of, located in, close to, leader of, industry peer, relative of, agree with, disagree with, belong to, president of, founder of, member of, belongs to].\n"

    entities = []
    for triple in triples:
        entities.extend([triple[0], triple[2]])
    entities = list(set(entities))
    query = f"Now the given sentence is: {' '.join(entities)}\n"
    entities = ','.join(['\t'.join([triple[0], triple[2]]) for triple in triples])
    query += "Now the given entity pairs are: {}\n".format(entities)
    query += "You should direct output relation triples like 'head_entity\trelation\ttail_entity', and multiple triples are separated by '\n'"

    answer = '\n'.join([f"{triple[0]}\t{triple[1]}\t{triple[2]}" for triple in triples])

    return {'instruct': instruct, 'query': query, 'answer': answer}




def main(args):
    lora_data = load_data(args.lora_data_path)
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    new_data = []
    for item in lora_data:
        new_data.append(process_triples(item))

    save_data(new_data, os.path.join(save_dir, "train.txt"))
    save_data(new_data[:100], os.path.join(save_dir, "dev.txt"))
    save_data(new_data[:100], os.path.join(save_dir, "test.txt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--lora_data_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    args = parser.parse_args()
    main(args)
