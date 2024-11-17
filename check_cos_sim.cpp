// main.cpp

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath> 

// Include the header of the cosine similarity implementation
void computeCosineSimilarities(
    const float* hostBatchVectors,
    const float* hostQueryVector,
    float* hostSimilarityScores,
    size_t numVectors,
    size_t vectorDim
);

int main() {
    // Seed random number generator
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // Parameters
    size_t numVectors = 10000;  // Number of vectors in the batch
    size_t vectorDim = 512;     // Dimensionality of each vector

    // Allocate host memory
    std::vector<float> hostBatchVectors(numVectors * vectorDim);
    std::vector<float> hostQueryVector(vectorDim);
    std::vector<float> hostSimilarityScores(numVectors);

    // Initialize hostBatchVectors and hostQueryVector with random data
    for (size_t i = 0; i < numVectors * vectorDim; ++i) {
        hostBatchVectors[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    for (size_t i = 0; i < vectorDim; ++i) {
        hostQueryVector[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    // Compute cosine similarities using the GPU
    computeCosineSimilarities(
        hostBatchVectors.data(),
        hostQueryVector.data(),
        hostSimilarityScores.data(),
        numVectors,
        vectorDim
    );

    // Optionally, verify results by computing similarities on the CPU for a few vectors
    // For demonstration, we'll compute the cosine similarity for the first vector
    float dotProduct = 0.0f;
    float batchNorm = 0.0f;
    float queryNorm = 0.0f;

    for (size_t i = 0; i < vectorDim; ++i) {
        float batchVal = hostBatchVectors[i];
        float queryVal = hostQueryVector[i];

        dotProduct += batchVal * queryVal;
        batchNorm += batchVal * batchVal;
        queryNorm += queryVal * queryVal;
    }

    batchNorm = sqrt(batchNorm);
    queryNorm = sqrt(queryNorm);
    float cpuSimilarity = dotProduct / (batchNorm * queryNorm + 1e-8f);

    std::cout << "CPU computed similarity for the first vector: " << cpuSimilarity << std::endl;
    std::cout << "GPU computed similarity for the first vector: " << hostSimilarityScores[0] << std::endl;

    // Check if the results are close
    if (std::abs(cpuSimilarity - hostSimilarityScores[0]) < 1e-5f) {
        std::cout << "The GPU and CPU results match closely." << std::endl;
    } else {
        std::cout << "There is a significant difference between GPU and CPU results." << std::endl;
    }

    return 0;
}
