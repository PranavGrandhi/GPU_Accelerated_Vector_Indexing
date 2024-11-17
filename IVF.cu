#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <algorithm>
#include <unordered_map>
#include "cnpy.h"         // For loading .npy files
#include "json.hpp"       // For JSON parsing

using json = nlohmann::json;
using namespace std;

class IVFIndex 
{
    public:
    IVFIndex
    (
        const vector<vector<vector<float>>>& cluster_embeddings,
        const vector<vector<int>>& cluster_mappings,
        const vector<vector<float>>& cluster_centroids,
        int n_probe = 8
    ) : cluster_embeddings(cluster_embeddings),
        cluster_mappings(cluster_mappings),
        cluster_centroids(cluster_centroids),
        n_probe(n_probe) {}

    //Returns top k results
    vector<pair<float, int>> search(const vector<float>& query, int k, bool use_cuda = false) 
    {
        // Find top centroids
        auto top_centroids = similarity_search.find_similar(cluster_centroids, query, n_probe, use_cuda);

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
            auto similarities = similarity_search.find_similar(cluster_embeddings[cluster], query, k, use_cuda);

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
    static IVFIndex from_pretrained(const string& data_dir, int n_probe = 8) {
        // Load cluster mappings from JSON
        ifstream mapping_file(data_dir + "/cluster_mappings.json");
        json cluster_mappings_json;
        mapping_file >> cluster_mappings_json;
        mapping_file.close();

        // Convert JSON to vector of vectors
        vector<vector<int>> cluster_mappings =
            cluster_mappings_json.get<vector<vector<int>>>();

        size_t n_clusters = cluster_mappings.size();

        // Load cluster embeddings
        vector<vector<vector<float>>> cluster_embeddings(n_clusters);

        for (size_t i = 0; i < n_clusters; ++i) {
            string filename = data_dir + "/cluster_embeddings_" + to_string(i) + ".npy";
            cnpy::NpyArray arr = cnpy::npy_load(filename);
            float* data = arr.data<float>();
            size_t rows = arr.shape[0];
            size_t cols = arr.shape[1];

            // Convert to vector of vectors
            cluster_embeddings[i].resize(rows, vector<float>(cols));
            for (size_t r = 0; r < rows; ++r) {
                for (size_t c = 0; c < cols; ++c) {
                    cluster_embeddings[i][r][c] = data[r * cols + c];
                }
            }
        }

        // Load cluster centroids
        string centroids_filename = data_dir + "/cluster_centroids.npy";
        cnpy::NpyArray centroids_arr = cnpy::npy_load(centroids_filename);
        float* centroids_data = centroids_arr.data<float>();
        size_t centroid_rows = centroids_arr.shape[0];
        size_t centroid_cols = centroids_arr.shape[1];

        // Convert to vector of vectors
        vector<vector<float>> cluster_centroids(centroid_rows, vector<float>(centroid_cols));
        for (size_t r = 0; r < centroid_rows; ++r) {
            for (size_t c = 0; c < centroid_cols; ++c) {
                cluster_centroids[r][c] = centroids_data[r * centroid_cols + c];
            }
        }

        // Print info
        size_t n_embeddings = 0;
        for (const auto& embeddings : cluster_embeddings) {
            n_embeddings += embeddings.size();
        }
        size_t embed_dim = centroid_cols;

        cout << "embeddings: (" << n_embeddings << ", " << embed_dim << ")" << endl;
        cout << "centroids: (" << centroid_rows << ", " << centroid_cols << ")" << endl;

        // Create and return IVFIndex instance
        return IVFIndex(cluster_embeddings, cluster_mappings, cluster_centroids, n_probe);
    }

    private:
    // Member variables
    vector<vector<vector<float>>> cluster_embeddings;
    vector<vector<int>> cluster_mappings;
    vector<vector<float>> cluster_centroids;
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

void main()
{
    // Load pretrained index
    IVFIndex index = IVFIndex::from_pretrained("/scratch/pvg2018/embeddings_data");

    vector<float> query = {4.71796142e-03 -4.11833674e-02 -8.81804302e-02  3.31530534e-02
     3.53304818e-02  3.03912107e-02 -4.78145927e-02 -1.87749043e-02
     3.59381959e-02  1.19639495e-02  2.35260520e-02  5.83246835e-02
     1.06149353e-02  1.94831844e-02 -5.85400388e-02  5.14089204e-02
     2.60280762e-02  6.72893152e-02 -7.62769654e-02 -4.63880450e-02
    -2.80892793e-02 -1.62975844e-02  6.06718063e-02  1.05081415e-02
     4.26498950e-02 -3.97572778e-02 -4.61665764e-02 -6.83665927e-03
    -6.24483032e-03  2.62057427e-02  2.06426834e-03 -3.59356217e-02
     3.86018516e-03  3.31193842e-02 -1.19582824e-01  3.06038782e-02
     2.44444106e-02  2.19231006e-02  9.18838475e-03  2.85859201e-02
    -7.06080673e-03 -1.72228403e-02 -3.81311327e-02  2.24495791e-02
     1.07677087e-01  3.86632159e-02 -2.71681696e-03 -2.52348930e-02
    -2.17898432e-02  2.94948407e-02 -2.96229012e-02 -6.29498288e-02
    -7.68944323e-02 -1.96542293e-02 -2.64674448e-03  6.46483675e-02
     2.25822162e-02 -1.82778656e-03 -4.64642942e-02  2.24591568e-02
    -3.00803371e-02 -1.84306771e-01 -4.61819768e-02 -3.56384763e-03
    -2.94625503e-03 -7.71500692e-02  5.93792424e-02 -7.32133910e-02
     4.09004744e-04 -8.98774422e-04 -1.97180472e-02  4.76434529e-02
    -1.34360208e-03 -4.37828861e-02 -2.50164829e-02 -7.14039057e-02
     7.53066316e-02  8.62853825e-02  3.70947607e-02 -6.92933947e-02
     5.12001850e-02  7.85337836e-02  1.23697855e-02 -3.34290825e-02
     5.77131882e-02 -9.52319577e-02  1.20108180e-01 -7.13750673e-03
    -5.58444336e-02 -1.86205860e-02  2.39620972e-02 -3.82877290e-02
    -1.63556516e-01 -8.35323893e-03 -1.91235691e-02  2.11214507e-03
    -2.62620137e-03 -4.56207879e-02 -8.82594660e-02  2.83136871e-02
    -8.33365098e-02 -8.55714642e-03 -3.46265733e-02  6.54222816e-02
    -7.20986053e-02  1.91075318e-02  9.28883627e-03  8.73555541e-02
     5.26178926e-02  9.30725597e-03  3.34407277e-02  4.50378396e-02
    -3.10743079e-02  4.78980094e-02  7.29019428e-03 -4.54907455e-02
     1.87967550e-02 -9.04823020e-02 -2.33089924e-03  1.39475331e-01
    -4.95556742e-02 -1.49378795e-02 -5.71628008e-03 -3.87898907e-02
    -4.34617177e-02 -6.14252016e-02 -1.47800781e-02 -3.31835847e-33
    -1.89118891e-03  6.06292523e-02 -3.06440014e-02 -6.31866008e-02
     2.25274451e-02 -4.94068712e-02  5.08545386e-03  4.10955958e-02
     3.15615349e-02 -3.43625844e-02 -6.64357394e-02 -7.26243556e-02
    -6.77699049e-04  1.24246739e-01 -1.82866657e-04 -4.10066396e-02
    -1.08768404e-01  4.47033718e-02  9.38976780e-02 -8.00140053e-02
    -7.87570886e-03 -3.34761776e-02 -5.53382747e-02 -7.65657723e-02
    -3.31538394e-02  9.40804649e-03  7.07482994e-02  5.72187416e-02
    -1.62534900e-02  1.72545835e-02  3.98560911e-02  2.71782726e-02
    -7.00322539e-02 -6.16917089e-02  9.28796604e-02  3.38243954e-02
     6.42544404e-02  1.33180777e-02  3.45546305e-02 -4.03608195e-02
    -3.96933854e-02  3.25472057e-02  2.60827113e-02 -2.80733667e-02
    -1.13821268e-01 -5.66303059e-02  3.04314774e-03 -7.09795877e-02
    -5.00105508e-02 -5.08923121e-02 -5.78217469e-02 -1.22257015e-02
    -9.52049941e-02 -9.76814330e-03  8.55443180e-02  7.94016123e-02
    -1.33429347e-02  2.46728491e-02 -2.10588872e-02  8.88959616e-02
     5.93979564e-03  3.82781178e-02  7.56516084e-02  3.18807326e-02
     2.48839520e-03  1.02840990e-04 -1.14716580e-02  4.40467186e-02
     2.03275084e-02 -3.82036753e-02  5.37820905e-02  3.88998874e-02
     2.27163471e-02 -9.76141915e-02  4.42475304e-02  2.11911257e-02
     9.52628106e-02 -5.90108112e-02  3.23684141e-02  6.18857332e-02
    -6.69554919e-02  3.38510959e-04 -2.99580712e-02 -6.58242106e-02
    -7.39396065e-02  8.54555611e-03  5.99229671e-02 -5.89817055e-02
     3.73255834e-02  5.99558349e-04 -1.06686994e-01  1.21311372e-04
    -3.38193849e-02 -1.80716049e-02 -2.03161109e-02  1.02726929e-33
    -8.70907679e-03  1.79198626e-02 -3.72981392e-02  9.42171142e-02
     3.13945487e-02 -6.92430558e-03 -4.87107970e-02  9.00371224e-02
    -1.99255757e-02  1.13496484e-04 -1.79040097e-02  3.61161083e-02
    -5.93669526e-02  2.20363354e-03  2.85467394e-02  3.24478634e-02
    -5.17052859e-02  1.83267258e-02 -6.13363273e-03  5.97839104e-03
    -4.13149819e-02  6.15840293e-02 -1.35768671e-02  2.57283952e-02
    -8.02747719e-03 -4.75861989e-02 -2.29917676e-03  7.89342746e-02
    -2.94053350e-02 -2.66124886e-02  2.47671437e-02  6.31769048e-03
    -5.20781316e-02  5.29371984e-02 -5.09274600e-04  6.84673637e-02
     1.58770452e-03 -3.84701379e-02 -1.65390857e-02  1.01636477e-01
     2.98386700e-02  4.29023243e-02 -1.50098121e-02 -7.14213476e-02
     8.66238326e-02 -3.00390311e-02 -8.14390776e-04  1.25929853e-02
     2.49324888e-02  7.58107938e-03  5.40288612e-02 -2.09853426e-02
    -3.58602894e-03  2.46256422e-02 -6.63906038e-02 -1.55087691e-02
     2.32482981e-03 -1.51150441e-02  2.87643317e-02  1.76336542e-02
     3.60169746e-02  9.80754197e-02 -8.36313739e-02  6.27067760e-02
     2.42237691e-02  9.52268776e-04  1.26083530e-02  7.26221874e-02
     8.45862404e-02  1.24724254e-01  2.79294662e-02  5.58399931e-02
     3.84558216e-02  2.65187137e-02  1.30140129e-02 -8.13357383e-02
     1.76628660e-02  1.37231322e-02 -1.12273470e-01 -3.17181833e-02
     5.64920306e-02  8.16810969e-03 -1.40014403e-02  8.51793028e-03
    -1.71774477e-02 -4.55289930e-02  5.18445820e-02  3.64091359e-02
     4.29231003e-02 -5.50834239e-02  5.46129188e-03 -3.49287279e-02
    -3.90019035e-04 -9.80667695e-02 -9.21786651e-02 -1.67541359e-08
    -1.83742531e-02 -6.38678670e-02 -9.36740777e-04 -5.18199019e-02
     5.46660684e-02  3.45426798e-02 -1.90884545e-02  7.93279186e-02
    -8.40881243e-02 -3.11221578e-03  8.97776932e-02  3.46806459e-02
    -6.66726455e-02 -1.32514574e-02 -5.48177632e-03  1.09089755e-01
     4.68545966e-02  4.00946699e-02  5.74468374e-02  3.02230869e-03
     3.04729547e-02  5.31949624e-02  1.01671711e-01 -1.27678309e-02
     2.11286023e-02 -3.84246930e-02  1.04128085e-01  8.80592912e-02
     3.08186114e-02 -6.30761236e-02 -7.12716877e-02  7.30988905e-02
    -9.22983105e-04 -1.81112029e-02  2.72404384e-02  7.34237283e-02
     3.87391634e-02 -6.92170560e-02  1.98555551e-02  3.40501703e-02
    -4.16927151e-02 -6.49509253e-03  3.91647071e-02 -5.34128994e-02
     7.51126232e-03  7.32441992e-03 -5.71702085e-02 -7.89676234e-03
    -7.07877874e-02 -6.73123971e-02  7.27447569e-02  7.21522793e-02
    -4.85257851e-03  3.24073732e-02  4.85010408e-02 -4.07437496e-02
    -5.65540791e-03 -1.04320630e-01 -4.36480679e-02  4.79487553e-02
    -2.60277912e-02  3.63256745e-02  2.62984131e-02 -3.29946354e-02}; 

      // Search
      int k = 5;
      vector<pair<float, int>> results = index.search(query, k);

    // Print results
    for (const auto& result : results) {
        cout << result.first << ", " << result.second << endl;
    }
}
