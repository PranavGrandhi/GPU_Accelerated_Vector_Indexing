import json
import glob
import os

import numpy as np
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    return parser

def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

if __name__ == "__main__":
    parser = args_parser()
    parser.add_argument("-b", "--batch-size", type=int, default=1024)
    args = parser.parse_args()

    model = get_model()
    embeddings = []
    file_lengths = []
    files = sorted(glob.glob(f"{args.data_dir}/*.json"))
    for filename in tqdm(files):
        with open(filename, "r", encoding="utf-8") as f:
            articles = json.load(f)
        file_lengths.append((os.path.relpath(filename, args.data_dir), len(articles)))
        texts = [article["text"] for article in articles]
        new_embeddings = model.encode(texts, device="cuda", batch_size=args.batch_size, show_progress_bar=True)
        embeddings.append(new_embeddings)
    embeddings = np.concatenate(embeddings)
    print(embeddings.shape)
    np.save(f"{args.data_dir}/embeddings.npy", embeddings)
    with open(f"{args.data_dir}/file_lengths.json", "w") as f:
        json.dump(file_lengths, f)