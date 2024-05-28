"""sample 1/3 data from the original data for quick testing."""
import json
import os
import shutil
import argparse
import numpy as np


def main(args):
    data_dir = args.data_dir
    src_dir = os.path.join("data/mixre/data", data_dir)
    dst_dir = os.path.join("data/mixre/data", f"{data_dir}-3k")

    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)

    shutil.copytree(src_dir, dst_dir)
    for filename in ['train_annotated.json', 'dev.json', 'test.json']:
        filepath = os.path.join(dst_dir, filename)
        data = json.load(open(filepath))
        length = len(data) // 3
        offsets = np.random.randint(0, 3, size=length)

        new_data = []
        for i in range(length):
            new_data.append(data[3 * i + offsets[i]])

        # sample 1/3 data from it
        with open(filepath, 'w') as f:
            json.dump(new_data, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="mix-30")
    args = parser.parse_args()
    np.random.seed(42)

    main(args)
