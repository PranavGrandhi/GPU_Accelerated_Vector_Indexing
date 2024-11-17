import json

import numpy as np
from sklearn.cluster import KMeans

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    return parser

if __name__ == "__main__":
    parser = args_parser()
    parser.add_argument("-n", "--n-clusters", type=int, default=128)
    args = parser.parse_args()

    embeddings = np.load(f"{args.data_dir}/embeddings.npy", mmap_mode='r')[:1000000]

    kmeans = KMeans(n_clusters=args.n_clusters, init="k-means++", n_init=1, random_state=42, verbose=1, max_iter=5).fit(embeddings)

    cluster_centroids = kmeans.cluster_centers_
    print("cluster_centroids", cluster_centroids.shape)
    np.save(f"{args.data_dir}/cluster_centroids.npy", cluster_centroids)

    cluster2idx = [list() for _ in range(args.n_clusters)]
    for i, label in enumerate(kmeans.labels_):
        cluster2idx[label].append(i)
    with open(f"{args.data_dir}/cluster_mappings.json", "w") as f:
        json.dump(cluster2idx, f)

    for cluster, idxs in enumerate(cluster2idx):
        cluster_embeddings = embeddings[idxs]
        print(f"cluster_embeddings_{cluster}", cluster_embeddings.shape)
        np.save(f"{args.data_dir}/cluster_embeddings_{cluster}.npy", cluster_embeddings)