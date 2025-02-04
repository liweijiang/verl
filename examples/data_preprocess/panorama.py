import argparse
import os

import pandas as pd
from datasets import load_dataset, Dataset

from tqdm.auto import tqdm

from verl.utils.fs import copy, makedirs

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


def generate_rl_dataset(local_dir):
    local_dir = os.path.expanduser(local_dir)
    dataset = load_dataset('liweijiang/panorama_train_v1_deduped')
    train_dataset = dataset['train']

    dataset_eval = load_dataset('liweijiang/panorama_eval')
    test_dataset = dataset_eval['test']
    val_dataset = dataset_eval['validation']

    train_dataset = train_dataset.map(
        function=make_map_fn('train'), with_indices=True)
    local_path = os.path.join(local_dir, 'train.parquet')
    train_dataset.to_parquet(local_path)

    test_dataset = test_dataset.map(
        function=make_map_fn('test'), with_indices=True)
    local_path = os.path.join(local_dir, 'test.parquet')
    test_dataset.to_parquet(local_path)

    val_dataset = val_dataset.map(
        function=make_map_fn('val'), with_indices=True)
    local_path = os.path.join(local_dir, 'val.parquet')
    val_dataset.to_parquet(local_path)


def small_train_sample_dataset(local_dir):
    data = [
        "Turn the phrase \"silent as the night\" into a creative metaphor for a cat.",
        "How would you explain quantum mechanics to a 7-year-old in 3 sentences?", 
        "Rewrite this boring text into something exciting: \"The meeting has been rescheduled.\"",
        "Create a fictional holiday, its traditions, and a name for it.",
        "Imagine a world where dogs can talk. Write one funny law from that world.",
        "Write a review of a fictional ice cream flavor called \"Starlight Serenade.\"",
        "What's a quirky name for a bookstore that only sells mysteries?",
        "Create a riddle about time.",
        "Translate this phrase into emojis: \"Time flies when you're having fun!\"",
        "Write a short, whimsical description of the smell of fresh rain.",
        "Write a two-sentence horror story involving a vending machine.",
        "What's a fun way to describe the color yellow without using the word \"bright\"?",
        "Write a humorous fortune-cookie message for someone with a sweet tooth.",
        "Can you make a limerick about penguins?",
        "Create a five-word slogan for a tech startup specializing in wearable devices.",
        "Write a motivational message for someone training for their first marathon.",
        "Rephrase: \"That was incredibly difficult.\"",
        "Imagine a futuristic invention to help introverts at social gatherings. Describe it briefly.",
        "Give me a \"dad joke\" about Wi-Fi.",
        "Write an intriguing first line for a science fiction short story.",
        "What’s one interesting fact about jellyfish that most people don’t know?",
        "Describe a day in the life of a cat in less than 100 words.",
        "Can you invent a silly, fake word and its definition?",
        "If clouds could talk, what’s one thing they might say?",
        "What's a unique way to make mornings less chaotic?",
        "Rewrite this phrase as a compliment: \"Your persistence is impressive.\"",
        "Write a pun about tacos and the moon.",
        "What’s one quirky skill that can make someone seem fascinating at a party?",
        "Turn these emojis into a dramatic movie tagline: “🔥🌊🕶️💥💃🐉.”",
        "Write a two-sentence spooky story that takes place in a library.",
        "Imagine a world where cats could give TED Talks—what would their first topic be?",
        "Write a poetic description of a thunderstorm, in exactly three sentences.",
    ]

    # Create dataset with prompts
    dataset_dict = []
    for idx, prompt in enumerate(data):
        dataset_dict.append({
            "data_source": "panorama",
            "prompt": [{
                "role": "user", 
                "content": prompt
            }],
            "reward_model": {
                "style": "model",
                "ground_truth": None  # should not be used
            },
            "extra_info": {
                'split': 'train',
                'index': idx
            }
        })
    
    dataset = Dataset.from_list(dataset_dict)
    
    # Save as parquet file
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, 'small_train_sample.parquet')
    dataset.to_parquet(local_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str,
                        choices=['sft', 'rm', 'rl'], default='rl')
    parser.add_argument('--local_dir', type=str, default='data/panorama')

    args = parser.parse_args()

    if args.data_type == 'rl':
        generate_rl_dataset(os.path.join(args.local_dir, args.data_type))
        small_train_sample_dataset(os.path.join(args.local_dir, args.data_type))
    else:
        raise NotImplementedError
