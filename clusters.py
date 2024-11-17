import json
import time
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
    print("About to Load Embeddings")
    embeddings = np.load(f"{args.data_dir}/embeddings.npy", mmap_mode='r')[:500000]
    print("Loaded Embeddings", embeddings.shape)
    time1 = time.time()
    kmeans = KMeans(n_clusters=args.n_clusters, init="k-means++", n_init=1, random_state=42, verbose=1, max_iter=8).fit(embeddings)
    print("Time taken for KMeans", time.time()-time1)
    cluster_centroids = kmeans.cluster_centers_
    print("cluster_centroids", cluster_centroids.shape)
    np.save(f"{args.data_dir}/cluster_centroids_Small_Data.npy", cluster_centroids)

    cluster2idx = [list() for _ in range(args.n_clusters)]
    for i, label in enumerate(kmeans.labels_):
        cluster2idx[label].append(i)
    with open(f"{args.data_dir}/cluster_mappings_Small_Data.json", "w") as f:
        json.dump(cluster2idx, f)

    for cluster, idxs in enumerate(cluster2idx):
        cluster_embeddings = embeddings[idxs]
        print(f"cluster_embeddings_{cluster}_Small_Data", cluster_embeddings.shape)
        np.save(f"{args.data_dir}/cluster_embeddings_{cluster}_Small_Data.npy", cluster_embeddings)