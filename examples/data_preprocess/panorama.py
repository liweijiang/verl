import argparse
import os

import pandas as pd
from datasets import load_dataset

from tqdm.auto import tqdm

from verl.utils.fs import copy, makedirs


def generate_rl_dataset(local_dir='~/data/full_hh_rlhf/rl'):
    local_dir = os.path.expanduser(local_dir)
    dataset = load_dataset('liweijiang/panorama_train_v1_deduped')
    train_dataset = dataset['train']

    dataset_eval = load_dataset('liweijiang/panorama_eval')
    test_dataset = dataset_eval['test']
    val_dataset = dataset_eval['validation']

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            prompt = example.pop('prompt')
            data_source = example.pop('data_source')

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": prompt
                }],
                "reward_model": {
                    "style": "model",
                    "ground_truth": None  # should not be used
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    local_path = os.path.join(local_dir, 'train.parquet')
    train_dataset.to_parquet(local_path)

    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    local_path = os.path.join(local_dir, 'test.parquet')
    test_dataset.to_parquet(local_path)

    val_dataset = val_dataset.map(function=make_map_fn('val'), with_indices=True)
    local_path = os.path.join(local_dir, 'val.parquet')
    val_dataset.to_parquet(local_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, choices=['sft', 'rm', 'rl'], default='rl')
    parser.add_argument('--local_dir', type=str, default='data/panorama')

    args = parser.parse_args()

    if args.data_type == 'rl':
        generate_rl_dataset(os.path.join(args.local_dir, args.data_type))
    else:
        raise NotImplementedError
