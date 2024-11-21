
# GPU Accelerated Vector Database Querying

This project demonstrates efficient vector database querying using GPU acceleration, applied to a large-scale dataset of Wikipedia articles. The process involves embedding generation, clustering, and approximate nearest neighbor search (GPU accelerated) to retrieve relevant articles based on input queries.

## Dataset

We utilize the Wikipedia dataset containing plain text from November 2020. You can download the dataset from Kaggle:

[Plain Text Wikipedia 202011 Dataset](https://www.kaggle.com/datasets/ltcmdrdata/plain-text-wikipedia-202011/data)

## Workflow (If you have access to CUDA5, skip to compilation and execution)
### (The below preprocessing is done and saved in CUDA5 /scratch/pvg2018/ if you don't have access, email pvg2018@nyu.edu or ax2119@nyu.edu or sc10670@nyu.edu)

1. **Embedding Generation**: Use `embedding.py` to generate vector embeddings for the Wikipedia articles.
2. **Clustering**: Apply K-means clustering with `cluster.py` to group the embeddings into 128 clusters. Save the clustered data in a designated folder.
3. **Data Conversion**: Convert the `.npy` files to `.bin` format using `convert_npy_bin.py` to ensure compatibility with C++.
4. **Query Processing**: Use pregenerated embeddings in the queries_data folder or create custom ones (See the Queries subsection for more info)
5. **Approximate Nearest Neighbor Search**: Compile and execute `IVF.cpp` to find the closest matching article to the query.

## Configurable Parameters

The following arguments need to be passed to the executable:
1. n_probe: Value from 1 to 128 which denotes how many top clusters can be chosen in the coarse search to do the fine search in
2. Which kernel mode: This defines which cuda kernel will run. It can either be "Atomic", or "NonAtomic". These are the 2 different types of kernels we use to compute the coarse and fine search
3. Sequential Search: This can be true or false. "true" stands for sequential search and "false" for non sequential search. This defines if each cluster is handled by a separate kernel or all the clusters are combined into one and a single kernel handles them all.
4. Use CUDA coarse: This can be true or false. This stands for using the CPU or GPU for the coarse search part (find the top n_probe cluster centroids).
5. Use CUDA fine: This can be true or false. This stands for using the CPU or GPU for the fine search part (find the top k closest elements in the top n_probe clusters).
6. ThreadsPerBlock: Can be any multiple of 32. Denotes the threads per block

## Input Queries
#### **Using Pre-uploaded Queries**
We have pre-generated embeddings for the following queries, saved as `.bin` files in the `queries_data` folder:

- `"What is learning rate in gradient descent?"` (`query1.bin`)
- `"What is Microbial biogeography?"` (`query2.bin`)
- `"Give me details about The Arch of Cabanes."` (`query3.bin`)
- `"Give me details about the history of the Taj Mahal."` (`query4.bin`)
- `"Tell me something about the labelling used on aid packages created and sent under the Marshall Plan."` (`query5.bin`)

You can use these queries by updating the query number in the code where the query file is specified.

#### **Generating a New Query**
If you want to process a new question:
1. Run the `test.py` script with the following command:
   ```bash
   python test.py --query "Your new question here"
   ```
2. This will generate a `.bin` file for your query and save it in the `queries_data` folder.
3. Update the query number in the code to point to the new query file.

#### **Example for Query Processing in IVF.cpp**
In your `IVF.cpp` code, make sure the path to the query file points to the desired query (e.g., `query1.bin` for the first pre-uploaded query or your new query file):
```cpp
std::string query_path = "queries_data/query1.bin"; // Update the file as needed
```

## Compilation and Execution

To compile and run the program on a CUDA-enabled machine:

```bash
ssh to cuda5.cims.nyu.edu
git clone https://github.com/PranavGrandhi/GPU_Accelerated_Vector_Indexing.git
cd GPU_Accelerated_Vector_Indexing

module load cuda-12.4
nvcc IVF.cpp cosine_similarity.cu -o IVF
./IVF --n_probe=30 --mode=Atomic --sequential_fine_search=false --use_cuda_coarse=true --use_cuda_fine=true --threadsperBlock=128 --print_results=true

Update flag values as needed
```

Upon execution, the program will output the article most relevant to the input query.

## Output

The program will display the title and content of the Wikipedia article that best matches the query "What is learning rate in gradient descent?", or any custom query



# Experiment Analysis

## Analysis

We can use the `run_multiple_configs.sh` file to run specific configurations multiple times and calculate the average running CPU/GPU time for each configuration. 

To execute an experiment:

1. **Prepare the Configuration File**: Each experiment has a corresponding `.txt` file listing all parameter configurations (e.g., `experiment1_config.txt`).

2. **Run the Script**: Use the following command:
   ```bash
   ./run_multiple_configs.sh <configuration_file> [<num_runs>]
   ```
   - Replace `<configuration_file>` with the name of the experiment configuration file (e.g., `experiment1_config.txt`).
   - Replace `<num_runs>` with the number of runs for averaging (default is 5).

3. **Output**: The script will display the configuration details, individual run times, and the average time for each configuration.

### Example Command
```bash
./run_multiple_configs.sh experiment1_config.txt 10
```

For more details, refer to the source code and comments within each script.

![ss](https://github.com/user-attachments/assets/2bfc5aef-edba-42e8-961f-c02ac976cb55)


