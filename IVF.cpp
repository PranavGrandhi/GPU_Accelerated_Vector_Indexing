#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <algorithm>
#include <unordered_map>
#include "json.hpp"       // For JSON parsing

using json = nlohmann::json;
using namespace std;
int num_clusters = 128;
int embedding_dim = 384;
int batch_size = 65536;

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
    size_t vectorDim
);

void computeCosineSimilaritiesAtomicOptimized(
    const float* dataBatch,
    const float* queryVector,
    float* similarityScores,
    size_t numVectors,
    size_t dim
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
    string mode) 
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
                        vectorSize
                    );
                }
                else if(mode == "NonAtomic")
                {
                    computeCosineSimilaritiesOptimized(
                    flattenedEmbeddings + i * vectorSize,
                    query,
                    scores.data(),
                    currentBatchSize,
                    vectorSize
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
    vector<pair<float, int>> search(vector<float>& query, int k, bool use_cuda, string mode = "Atomic") 
    {
        // Find top centroids
        auto top_centroids = findSimilar(cluster_centroids.data(), query.data(), num_clusters, embedding_dim, n_probe, batch_size, use_cuda, mode);

        // Min-heap to store top k results
        //similarity, index
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
            auto similarities = findSimilar(cluster_embeddings[cluster].data(), query.data(), elements_in_cluster, embedding_dim, k, batch_size, use_cuda , mode);

            for (const auto& sim : similarities) 
            {
                float score = sim.first;
                int idx = sim.second;
                int mapping_value = cluster_mappings[cluster][idx];

                if (min_heap.size() < k) 
                {
                    min_heap.emplace(score, mapping_value);
                } else if (score > min_heap.top().first) 
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
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <n_probe> <Atomic|NonAtomic>" << endl;
        return 1;
    }

    int n_probe = stoi(argv[1]);
    string mode = argv[2];

    if (mode != "Atomic" && mode != "NonAtomic") {
        cerr << "Error: Mode must be either 'Atomic' or 'NonAtomic'." << endl;
        return 1;
    }

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

    // store the DB mapping back the string from idx
    cout << "Loading mapBack" << endl;
    mapBack map_back("/scratch/pvg2018");
    cout << "Loaded mapBack" << endl;

    // Search
    int k = 5;
    vector<pair<float, int>> results, results2;

    // Measure time for GPU search
    auto start_gpu = chrono::high_resolution_clock::now();
    results = index.search(query, k, true, mode);
    auto end_gpu = chrono::high_resolution_clock::now();
    auto gpu_duration = chrono::duration_cast<chrono::milliseconds>(end_gpu - start_gpu);

    // Measure time for CPU search
    auto start_cpu = chrono::high_resolution_clock::now();
    results2 = index.search(query, k, false, mode);
    auto end_cpu = chrono::high_resolution_clock::now();
    auto cpu_duration = chrono::duration_cast<chrono::milliseconds>(end_cpu - start_cpu);

    // Print GPU Results
    cout << "GPU Results: " << endl;
    for (const auto& result : results) {
        std::string text = map_back.get(result.second);
        std::string sub_text = text.substr(0, 200);
        cout << result.first << ", " << result.second << "::: Text: " << sub_text << endl;
    }
    cout << "GPU Search Time: " << gpu_duration.count() << " ms" << endl;

    // Print CPU Results
    cout << "CPU Results: " << endl;
    for (const auto& result : results2) {
        std::string text = map_back.get(result.second);
        std::string sub_text = text.substr(0, 200);
        cout << result.first << ", " << result.second << "::: Text: " << sub_text << endl;
    }
    cout << "CPU Search Time: " << cpu_duration.count() << " ms" << endl;
}
