#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <algorithm>
#include <unordered_map>
#include <climits>
#include "json.hpp"       // For JSON parsing

using json = nlohmann::json;
using namespace std;
int num_clusters = 128;
int embedding_dim = 384;
int batch_size = INT_MAX;

// void computeCosineSimilarities(
//     const float* hostBatchVectors,
//     const float* hostQueryVector,
//     float* hostSimilarityScores,
//     size_t numVectors,
//     size_t vectorDim
// );

void computeCosineSimilaritiesOptimized(
    const float* hostBatchVectors,
    const float* hostQueryVector,
    float* hostSimilarityScores,
    size_t numVectors,
    size_t vectorDim,
    int threadsPerBlock = 256
);

void computeCosineSimilaritiesAtomicOptimized(
    const float* dataBatch,
    const float* queryVector,
    float* similarityScores,
    size_t numVectors,
    size_t dim,
    int threadsPerBlock = 256
);

// mapBack
class mapBack {
private:
    std::string data_dir;
    std::vector<std::pair<std::string, int>> idx2file;
    std::unordered_map<std::string, json> file_cache;

public:
    // Constructor
    mapBack(const std::string &data_dir) : data_dir(data_dir) {
        std::ifstream file_lengths_file(data_dir + "/file_lengths.json");
        if (!file_lengths_file.is_open()) {
            throw std::runtime_error("Could not open file_lengths.json");
        }

        json file_lengths_json;
        file_lengths_file >> file_lengths_json;
        file_lengths_file.close();

        // Iterate over the numerical keys of the JSON object
        int file_count = 0;
        for (const auto &[key, value] : file_lengths_json.items()) {
            // Each `value` is a list where:
            // value[0] = filename
            // value[1] = number of articles
            std::string filename = value[0];
            int num_articles = value[1];

            if (file_count%100 == 0) cout << file_count << " number of file articles mapped" << endl;
            file_count++;

            for (int i = 0; i < num_articles; ++i) {
                // if (i % 1000 == 0) {
                //     std::cout << i << " articles done for this filename: " << filename << ":: and num_articles: " << num_articles << std::endl;
                // }
                idx2file.emplace_back(filename, i);
            }
        }
    }

    // Reads a file and caches the result
    const json &read_file(const std::string &filename) {
        // Check if the file is already cached
        if (file_cache.find(filename) == file_cache.end()) {
            std::ifstream file(data_dir + "/wikidata/enwiki20201020/" + filename);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file: " + filename);
            }

            json json_data;
            file >> json_data;
            file.close();

            // Cache the file
            file_cache[filename] = json_data;
        }

        return file_cache[filename];
    }

    // Gets the text corresponding to a given index
    std::string get(int idx) {
        if (idx < 0 || idx >= idx2file.size()) {
            throw std::out_of_range("Index out of range");
        }

        const auto &[filename, offset] = idx2file[idx];

        const json &data = read_file(filename);

        if (offset < 0 || offset >= data.size()) {
            throw std::out_of_range("Offset out of range in file: " + filename);
        }

        return data[offset]["text"].get<std::string>();
    }
};


void computeCosineSimilaritiesCPU(
    const float* batchVectors,
    const float* queryVector,
    float* similarityScores,
    size_t numVectors,
    size_t vectorDim) 
{
    // Compute the norm of the query vector
    float queryNorm = 0.0f;
    for (size_t i = 0; i < vectorDim; ++i) 
    {
        float val = queryVector[i];
        queryNorm += val * val;
    }
    queryNorm = sqrtf(queryNorm);

    // Compute cosine similarity for each batch vector
    for (size_t vecIdx = 0; vecIdx < numVectors; ++vecIdx) 
    {
        float dotProduct = 0.0f;
        float batchNorm = 0.0f;
        for (size_t i = 0; i < vectorDim; ++i) {
            float batchVal = batchVectors[vecIdx * vectorDim + i];
            float queryVal = queryVector[i];
            dotProduct += batchVal * queryVal;
            batchNorm += batchVal * batchVal;
        }
        batchNorm = sqrtf(batchNorm);
        similarityScores[vecIdx] = dotProduct / (batchNorm * queryNorm + 1e-8f);
    }
}
class IVFIndex 
{
    public:
    IVFIndex
    (
        vector<vector<float>> cluster_embeddings, // 128 clusters - each cluster has x embeddings of size 384 each
        vector<vector<int>> cluster_mappings,
        vector<float> cluster_centroids,
        int n_probe
    ) : cluster_embeddings(cluster_embeddings),
        cluster_mappings(cluster_mappings),
        cluster_centroids(cluster_centroids),
        n_probe(n_probe) {}

    //returns the top k centroids
    vector<pair<float, int>> findSimilar(
    float* flattenedEmbeddings,
    float* query,
    int numEmbeddings,
    int vectorSize,
    int topK,
    int batchSize,
    bool useCuda,
    string mode,
    int threadsPerBlock = 256) 
    {
        if (batchSize == 0) 
        {
            batchSize = numEmbeddings;
        }

        // min heap to keep track of top k vectors
        typedef std::priority_queue<
            pair<float, int>,
            vector<pair<float, int>>,
            greater<pair<float, int>>
        > myheap;
        myheap heap;

        // loop over chunks in dataset
        for (int i = 0; i < numEmbeddings; i += batchSize) 
        {
            int currentBatchSize = min(batchSize, numEmbeddings - i);

            // get cosine similarity on this batch
            vector<float> scores(currentBatchSize);
            if(useCuda)
            {
                if(mode == "Atomic")
                {
                    computeCosineSimilaritiesAtomicOptimized(
                        flattenedEmbeddings + i * vectorSize,
                        query,
                        scores.data(),
                        currentBatchSize,
                        vectorSize,
                        threadsPerBlock
                    );
                }
                else if(mode == "NonAtomic")
                {
                    computeCosineSimilaritiesOptimized(
                    flattenedEmbeddings + i * vectorSize,
                    query,
                    scores.data(),
                    currentBatchSize,
                    vectorSize,
                    threadsPerBlock
                    );
                }
            }
            else
            {
                computeCosineSimilaritiesCPU(
                    flattenedEmbeddings + i * vectorSize,
                    query,
                    scores.data(),
                    currentBatchSize,
                    vectorSize
                );
            }

            // update heap to keep track of top k
            for (int j = 0; j < currentBatchSize; j++) 
            {
                if (heap.size() < topK) 
                {
                    heap.push(make_pair(scores[j], i + j));
                } 
                else if (make_pair(scores[j], i + j) > heap.top()) 
                {
                    heap.pop();
                    heap.push(make_pair(scores[j], i + j));
                }
            }
        }

        vector<pair<float, int>> result;
        myheap tempHeap = heap;
        while (!tempHeap.empty()) 
        {
            result.push_back(tempHeap.top());
            tempHeap.pop();
        }
        std::reverse(result.begin(), result.end());
        return result;
    }

    //Returns top k results from searching top n_probe centroids
    vector<pair<float, int>> search(vector<float>& query, int k, bool use_cuda_coarse, bool use_cuda_fine, string mode = "Atomic", bool sequential_fine_search = true, int threadsPerBlock = 256) 
    {
        // Find top centroids
    auto top_centroids = findSimilar(
        cluster_centroids.data(), 
        query.data(), 
        num_clusters, 
        embedding_dim, 
        n_probe, 
        batch_size, 
        use_cuda_coarse,
        mode,
        threadsPerBlock
    );

    if (sequential_fine_search)
    {
        // Original behavior: Process each cluster sequentially
        // Min-heap to store top k results
        auto cmp = [](const pair<float, int>& left, const pair<float, int>& right) 
        {
            return left.first > right.first; 
        };
        priority_queue<pair<float, int>, vector<pair<float, int>>, decltype(cmp)> min_heap(cmp);

        // Iterate over top centroids
        for (const auto& centroid : top_centroids) 
        {
            int cluster = centroid.second;

            // Find similar embeddings in the cluster
            int elements_in_cluster = cluster_embeddings[cluster].size() / embedding_dim;
            auto similarities = findSimilar(
                cluster_embeddings[cluster].data(), 
                query.data(), 
                elements_in_cluster, 
                embedding_dim, 
                k, 
                batch_size, 
                use_cuda_fine,
                mode,
                threadsPerBlock
            );

            for (const auto& sim : similarities) 
            {
                float score = sim.first;
                int idx = sim.second;
                int mapping_value = cluster_mappings[cluster][idx];

                if (min_heap.size() < k) 
                {
                    min_heap.emplace(score, mapping_value);
                } 
                else if (score > min_heap.top().first) 
                {
                    min_heap.pop();
                    min_heap.emplace(score, mapping_value);
                }
            }
        }

        // Extract results from the heap
        vector<pair<float, int>> results;
        while (!min_heap.empty()) 
        {
            results.push_back(min_heap.top());
            min_heap.pop();
        }
        reverse(results.begin(), results.end()); 
        return results;
    }
    else
    {
        // New behavior: Combine embeddings from top centroids and process once
        vector<float> combined_embeddings;
        vector<int> combined_mappings;
        int total_elements = 0;

        // First, calculate total number of embeddings to reserve space
        for (const auto& centroid : top_centroids)
        {
            int cluster = centroid.second;
            int elements_in_cluster = cluster_embeddings[cluster].size() / embedding_dim;
            total_elements += elements_in_cluster;
        }

        combined_embeddings.reserve(total_elements * embedding_dim);
        combined_mappings.reserve(total_elements);

        // Combine embeddings and mappings from top centroids
        for (const auto& centroid : top_centroids)
        {
            int cluster = centroid.second;
            const vector<float>& cluster_emb = cluster_embeddings[cluster];
            const vector<int>& cluster_map = cluster_mappings[cluster];

            combined_embeddings.insert(
                combined_embeddings.end(), 
                cluster_emb.begin(), 
                cluster_emb.end()
            );
            combined_mappings.insert(
                combined_mappings.end(), 
                cluster_map.begin(), 
                cluster_map.end()
            );
        }

        // Find similarities using the combined embeddings
        auto similarities = findSimilar(
            combined_embeddings.data(), 
            query.data(), 
            total_elements, 
            embedding_dim, 
            k, 
            batch_size, 
            use_cuda_fine,
            mode,
            threadsPerBlock
        );

        // Min-heap to store top k results
        auto cmp = [](const pair<float, int>& left, const pair<float, int>& right) 
        {
            return left.first > right.first; 
        };
        priority_queue<pair<float, int>, vector<pair<float, int>>, decltype(cmp)> min_heap(cmp);

        // Map similarities back to original indices
        for (const auto& sim : similarities) 
        {
            float score = sim.first;
            int idx = sim.second;
            int mapping_value = combined_mappings[idx];

            if (min_heap.size() < k) 
            {
                min_heap.emplace(score, mapping_value);
            } 
            else if (score > min_heap.top().first) 
            {
                min_heap.pop();
                min_heap.emplace(score, mapping_value);
            }
        }

        // Extract results from the heap
        vector<pair<float, int>> results;
        while (!min_heap.empty()) 
        {
            results.push_back(min_heap.top());
            min_heap.pop();
        }
        reverse(results.begin(), results.end()); 
        return results;
    }
}

    // Static method to load pretrained index
    static IVFIndex from_pretrained(const string& data_dir, int n_probe) 
    {
        // Load cluster mappings from JSON
        ifstream mapping_file(data_dir + "/cluster_mappings.json");
        json cluster_mappings_json;
        mapping_file >> cluster_mappings_json;
        mapping_file.close();

        // Convert JSON to vector of vectors
        vector<vector<int>> cluster_mappings =
            cluster_mappings_json.get<vector<vector<int>>>();

        int n_clusters = cluster_mappings.size();

        // Load cluster embeddings
        vector<vector<float>> cluster_embeddings(n_clusters);

        for (int i = 0; i < n_clusters; ++i) 
        {
            string filename = data_dir + "/cluster_embeddings_" + to_string(i) +".bin";

            // Read binary file
            ifstream file(filename, ios::binary | ios::ate);
            if (!file.is_open()) 
            {
                throw runtime_error("Failed to open file: " + filename);
            }

            // Get file size and calculate number of rows and columns
            streamsize file_size = file.tellg();
            file.seekg(0, ios::beg);

            int cols = 384; // Set this to the correct embedding dimension
            int rows = file_size / (sizeof(float) * cols);
            if (file_size % (sizeof(float) * cols) != 0) 
            {
                throw runtime_error("File size is not consistent with the expected float dimensions.");
            }

            // Read data into a single buffer
            vector<float> buffer(rows * cols);
            file.read(reinterpret_cast<char*>(buffer.data()), file_size);
            file.close();

            // Convert to vector of vectors
            cluster_embeddings[i].resize(rows * cols);
            cluster_embeddings[i] = buffer;
        }

        // Load cluster centroids
        string centroids_filename = data_dir + "/cluster_centroids.bin";
        ifstream centroids_file(centroids_filename, ios::binary | ios::ate);
        if (!centroids_file.is_open()) 
        {
            throw runtime_error("Failed to open file: " + centroids_filename);
        }

        // Get file size and calculate number of rows and columns
        streamsize centroids_file_size = centroids_file.tellg();
        centroids_file.seekg(0, ios::beg);

        int centroid_cols = 384; 
        int centroid_rows = centroids_file_size / (sizeof(float) * centroid_cols);
        if (centroids_file_size % (sizeof(float) * centroid_cols) != 0) 
        {
            throw runtime_error("Centroids file size is not consistent with the expected float dimensions.");
        }

        // Read data into a single buffer
        vector<float> centroids_buffer(centroid_rows * centroid_cols);
        centroids_file.read(reinterpret_cast<char*>(centroids_buffer.data()), centroids_file_size);
        centroids_file.close();

        // // Convert to vector of vectors
        // vector<vector<float>> cluster_centroids(centroid_rows, vector<float>(centroid_cols));
        // for (size_t r = 0; r < centroid_rows; ++r) 
        // {
        //     for (size_t c = 0; c < centroid_cols; ++c) 
        //     {
        //         cluster_centroids[r][c] = centroids_buffer[r * centroid_cols + c];
        //     }
        // }

        // Create and return IVFIndex instance
        return IVFIndex(cluster_embeddings, cluster_mappings, centroids_buffer, n_probe);
    }

    private:
    // Member variables
    vector<vector<float>> cluster_embeddings; 
    vector<vector<int>> cluster_mappings;
    vector<float> cluster_centroids; 
    int n_probe;

    // Placeholder for similarity_search (to be implemented separately)
    struct SimilaritySearch {
        vector<pair<float, int>> find_similar(
            const vector<vector<float>>& data,
            const vector<float>& query,
            int k,
            bool use_cuda
        ) {
            // Implementation goes here
            return {};
        }
    } similarity_search;
};

int main(int argc, char* argv[])
{
    int n_probe = 20;
    string mode = "NonAtomic";
    bool sequential_fine_search = true;
    bool use_cuda_coarse = false;
    bool use_cuda_fine = false;
    int threadsperBlock = 256;
    bool print_results = false;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg.find("--n_probe=") == 0) {
            string value = arg.substr(strlen("--n_probe="));
            try {
                n_probe = stoi(value);
            } catch (const invalid_argument&) {
                cerr << "Error: --n_probe must be an integer." << endl;
                return 1;
            }
        } else if (arg.find("--mode=") == 0) {
            mode = arg.substr(strlen("--mode="));
            if (mode != "Atomic" && mode != "NonAtomic") {
                cerr << "Error: --mode must be either 'Atomic' or 'NonAtomic'." << endl;
                return 1;
            }
        } else if (arg.find("--sequential_fine_search=") == 0) {
            string value = arg.substr(strlen("--sequential_fine_search="));
            if (value == "true" || value == "1") {
                sequential_fine_search = true;
            } else if (value == "false" || value == "0") {
                sequential_fine_search = false;
            } else {
                cerr << "Error: --sequential_fine_search must be 'true' or 'false'." << endl;
                return 1;
            }
        } else if (arg.find("--use_cuda_coarse=") == 0) {
            string value = arg.substr(strlen("--use_cuda_coarse="));
            if (value == "true" || value == "1") {
                use_cuda_coarse = true;
            } else if (value == "false" || value == "0") {
                use_cuda_coarse = false;
            } else {
                cerr << "Error: --use_cuda_coarse must be 'true' or 'false'." << endl;
                return 1;
            }
        } else if (arg.find("--use_cuda_fine=") == 0) {
            string value = arg.substr(strlen("--use_cuda_fine="));
            if (value == "true" || value == "1") {
                use_cuda_fine = true;
            } else if (value == "false" || value == "0") {
                use_cuda_fine = false;
            } else {
                cerr << "Error: --use_cuda_fine must be 'true' or 'false'." << endl;
                return 1;
            }
        } else if (arg.find("--threadsperBlock=") == 0) {
            string value = arg.substr(strlen("--threadsperBlock="));
            try {
                int threads = stoi(value);
                if (threads % 32 != 0) {
                    cerr << "Error: --threadsperBlock must be a multiple of 32." << endl;
                    return 1;
                }
                threadsperBlock = threads;
            } catch (const invalid_argument&) {
                cerr << "Error: --threadsperBlock must be a valid integer." << endl;
                return 1;
            } catch (const out_of_range&) {
                cerr << "Error: --threadsperBlock value is out of range." << endl;
                return 1;
            }
        } else if (arg.find("--print_results=") == 0) {
            string value = arg.substr(strlen("--print_results="));
            if (value == "true" || value == "1") {
                print_results = true;
            } else if (value == "false" || value == "0") {
                print_results = false;
            } else {
                cerr << "Error: --print_results must be 'true' or 'false'." << endl;
                return 1;
            }
        } 
        else {
            cerr << "Error: Unknown argument '" << arg << "'." << endl;
            return 1;
        }
    }

    // Display the parsed arguments (optional for debugging)
    cout << "n_probe: " << n_probe << endl;
    cout << "Mode: " << mode << endl;
    cout << "Sequential Fine Search: " << (sequential_fine_search ? "True" : "False") << endl;
    cout << "Use CUDA for Coarse Search: " << (use_cuda_coarse ? "True" : "False") << endl;
    cout << "Use CUDA for Fine Search: " << (use_cuda_fine ? "True" : "False") << endl;
    cout << "Threads per Block: " << threadsperBlock << endl;
    cout << "Print Results: " << (print_results ? "True" : "False") << endl;


    // Load pretrained index
    IVFIndex index = IVFIndex::from_pretrained("/scratch/pvg2018/cluster_data", n_probe);

    string filePath = "./queries_data/query1.bin";
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filePath);
    }

    // Get the file size
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Check that the file size is a multiple of the size of float
    if (fileSize % sizeof(float) != 0) {
        throw std::runtime_error("File size is not a multiple of float size.");
    }

    // Create a vector to hold the data
    std::vector<float> query(fileSize / sizeof(float));

    // Read the data into the vector
    if (!file.read(reinterpret_cast<char*>(query.data()), fileSize)) {
        throw std::runtime_error("Failed to read file: " + filePath);
    }

    // // store the DB mapping back the string from idx
    // cout << "Loading mapBack" << endl;
    // cout << "Loaded mapBack" << endl;
    if(print_results)
    {
        mapBack map_back("/scratch/pvg2018");
    }

    // Search
    int k = 5;
    vector<pair<float, int>> results, results2;

    // Measure time for GPU search
    auto start_time = chrono::high_resolution_clock::now();
    results = index.search(query, k, use_cuda_coarse, use_cuda_fine, mode, sequential_fine_search, threadsperBlock);
    auto end_time = chrono::high_resolution_clock::now();
    auto time_duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    // Print Results
    if (use_cuda_coarse || use_cuda_fine) cout << "GPU Results: " << endl;
    else cout << "CPU Results: " << endl;
    for (const auto& result : results) {
        if(print_results)
        {
            std::string text = map_back.get(result.second);
            std::string sub_text = text.substr(0, 200);
            cout << result.first << ", " << result.second << "::: Text: " << sub_text << endl;
        }
        else
            cout << result.first << ", " << result.second << endl;
    }
    cout << "Search Time: " << time_duration.count() << " ms" << endl;
}
