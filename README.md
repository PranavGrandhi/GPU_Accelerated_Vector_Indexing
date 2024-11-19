
# GPU Accelerated Vector Indexing

This project demonstrates efficient vector indexing using GPU acceleration, applied to a large-scale dataset of Wikipedia articles. The process involves embedding generation, clustering, and approximate nearest neighbor search (GPU accelerated) to retrieve relevant articles based on input queries.

## Dataset

We utilize the Wikipedia dataset containing plain text from November 2020. You can download the dataset from Kaggle:

[Plain Text Wikipedia 202011 Dataset](https://www.kaggle.com/datasets/ltcmdrdata/plain-text-wikipedia-202011/data)

## Workflow

1. **Embedding Generation**: Use `embedding.py` to generate vector embeddings for the Wikipedia articles.
2. **Clustering**: Apply K-means clustering with `cluster.py` to group the embeddings into 128 clusters. Save the clustered data in a designated folder.
3. **Data Conversion**: Convert the `.npy` files to `.bin` format using `convert_npy_bin.py` to ensure compatibility with C++.
4. **Query Processing**: Place the embedding of the query "What is learning rate in gradient descent?" in the `queries_data` folder.
5. **Approximate Nearest Neighbor Search**: Compile and execute `IVF.cpp` to find the closest matching article to the query.

## Configurable Parameters

- **Top K Centroids**: The user can modify the number of top centroids by changing the value passed to the `pretrained()` function in the `main()` function of `IVF.cpp`.
- **Top K Closest Matches**: Adjust the number of closest elements matched by modifying the relevant parameter in `IVF.cpp`.

## Compilation and Execution

To compile and run the program on a CUDA-enabled machine:
The following arguments need to be passed to the executable:
1. n_probe: Value from 1 to 128 which denotes how many top clusters can be chosen in the coarse search to do the fine search in
2. Which kernel mode: This defines which cuda kernel will run. It can either be "Atomic", or "NonAtomic". These are the 2 different types of kernels we use to compute the coarse and fine search
3. Sequential Search: This can be true or false. "true" stands for sequential search and "false" for non sequential search. This defines if each cluster is handled by a seperate kernel or all the clusters are combined into one and a single kernel handles them all.
4. Use CUDA coarse: This can be true or false. This stands for using the CPU or GPU for the coarse search part (find the top n_probe cluster centroids).
5. Use CUDA fine: This can be true or false. This stands for using the CPU or GPU for the fine search part (find the top k closest elements in the top n_probe clusters).

```bash
nvcc IVF.cpp cosine_similarity.cu -o IVF
./IVF $n_probe $kernel_mode $Sequential_search $cuda_coarse $cuda_fine
Example: ./IVF 40 Atomic true false true
```

Upon execution, the program will output the article most relevant to the input query.

## Output

The program will display the title and content of the Wikipedia article that best matches the query "What is learning rate in gradient descent?".
![Screenshot 2024-11-18 201002](https://github.com/user-attachments/assets/4be1e7a9-3e6e-4e65-9e68-0c65c89a770b)


For more details, refer to the source code and comments within each script.
