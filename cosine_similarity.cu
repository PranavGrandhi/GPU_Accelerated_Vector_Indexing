// cosine_similarity.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>


// Error checking macro
#define CUDA_CHECK(call)                                                        \
    {                                                                           \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - "\
                      << cudaGetErrorString(err) << std::endl;                  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    }

// Function to compute the norm of a vector on the host
__host__ float computeHostVectorNorm(const float* vector, size_t vectorSize) {
    float sum = 0.0f;
    for (size_t i = 0; i < vectorSize; ++i) {
        sum += vector[i] * vector[i];
    }
    return sqrt(sum);
}

// ORIGINAL
// CUDA kernel for computing cosine similarities
__global__ void cosineSimilarityKernel(
    const float* __restrict__ batchVectors,
    const float* __restrict__ queryVector,
    float* __restrict__ similarityScores,
    float queryNorm,
    size_t vectorDim
) {
    extern __shared__ float sharedMem[];
    float* sharedDot = sharedMem;
    float* sharedBatchNorm = sharedMem + blockDim.x;

    size_t tid = threadIdx.x;
    size_t globalVectorIdx = blockIdx.x;

    // Initialize shared memory
    sharedDot[tid] = 0.0f;
    sharedBatchNorm[tid] = 0.0f;

    __syncthreads();

    // Each thread computes partial sums
    for (size_t i = tid; i < vectorDim; i += blockDim.x) {
        float batchVal = batchVectors[globalVectorIdx * vectorDim + i];
        float queryVal = queryVector[i];

        sharedDot[tid] += batchVal * queryVal;
        sharedBatchNorm[tid] += batchVal * batchVal;
    }

    __syncthreads();

    // Reduce partial sums within the block
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedDot[tid] += sharedDot[tid + stride];
            sharedBatchNorm[tid] += sharedBatchNorm[tid + stride];
        }
        __syncthreads();
    }

    // Compute final similarity score
    if (tid == 0) {
        float batchNorm = sqrt(sharedBatchNorm[0]);
        float denominator = batchNorm * queryNorm + 1e-8f; // Avoid division by zero
        similarityScores[globalVectorIdx] = sharedDot[0] / denominator;
    }
}

// Host function to compute cosine similarities
void computeCosineSimilarities(
    const float* hostBatchVectors,
    const float* hostQueryVector,
    float* hostSimilarityScores,
    size_t numVectors,
    size_t vectorDim
) {
    // Device pointers
    float* deviceBatchVectors = nullptr;
    float* deviceQueryVector = nullptr;
    float* deviceSimilarityScores = nullptr;

    size_t batchSizeBytes = numVectors * vectorDim * sizeof(float);
    size_t querySizeBytes = vectorDim * sizeof(float);
    size_t scoresSizeBytes = numVectors * sizeof(float);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&deviceBatchVectors, batchSizeBytes));
    CUDA_CHECK(cudaMalloc(&deviceQueryVector, querySizeBytes));
    CUDA_CHECK(cudaMalloc(&deviceSimilarityScores, scoresSizeBytes));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(deviceBatchVectors, hostBatchVectors, batchSizeBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceQueryVector, hostQueryVector, querySizeBytes, cudaMemcpyHostToDevice));

    // Compute the norm of the query vector on the host
    float hostQueryNorm = 0.0f;
    for (size_t i = 0; i < vectorDim; ++i) {
        hostQueryNorm += hostQueryVector[i] * hostQueryVector[i];
    }
    hostQueryNorm = sqrt(hostQueryNorm);

    // Kernel configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = static_cast<int>(numVectors);
    size_t sharedMemSize = threadsPerBlock * 2 * sizeof(float); // sharedDot and sharedBatchNorm

    // Launch kernel
    cosineSimilarityKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
        deviceBatchVectors,
        deviceQueryVector,
        deviceSimilarityScores,
        hostQueryNorm,
        vectorDim
    );

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Synchronize device
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(hostSimilarityScores, deviceSimilarityScores, scoresSizeBytes, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(deviceBatchVectors));
    CUDA_CHECK(cudaFree(deviceQueryVector));
    CUDA_CHECK(cudaFree(deviceSimilarityScores));
}

// Non Atomic 2nd implimentation
__global__ void optimizedCosineSimilarityKernel(
    const float* __restrict__ batchVectors,
    const float* __restrict__ queryVector,
    float* __restrict__ similarityScores,
    float queryNorm,
    size_t numVectors,
    size_t vectorDim
) {
    extern __shared__ float sharedQuery[];

    // Load query vector into shared memory
    for (size_t i = threadIdx.x; i < vectorDim; i += blockDim.x) {
        sharedQuery[i] = queryVector[i];
    }
    __syncthreads();

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numVectors) {
        const float* batchVector = &batchVectors[idx * vectorDim];
        float dotProduct = 0.0f;
        float batchNorm = 0.0f;

        // Compute dot product and batch vector norm
        for (size_t i = 0; i < vectorDim; ++i) {
            float batchVal = batchVector[i];
            float queryVal = sharedQuery[i];

            dotProduct += batchVal * queryVal;
            batchNorm += batchVal * batchVal;
        }

        batchNorm = sqrtf(batchNorm);
        float denominator = batchNorm * queryNorm + 1e-8f;
        similarityScores[idx] = dotProduct / denominator;
    }
}

void computeCosineSimilaritiesOptimized(
    const float* batchVectors,
    const float* queryVector,
    float* similarityScores,
    size_t numVectors,
    size_t vectorDim
    int threadsPer_Block = 256;
) {
    // Compute the norm of the query vector on the host
    float queryNorm = 0.0f;
    for (size_t i = 0; i < vectorDim; ++i) {
        float val = queryVector[i];
        queryNorm += val * val;
    }
    queryNorm = sqrtf(queryNorm);

    // Device pointers
    float* d_batchVectors;
    float* d_queryVector;
    float* d_similarityScores;

    size_t batchSizeBytes = numVectors * vectorDim * sizeof(float);
    size_t querySizeBytes = vectorDim * sizeof(float);
    size_t scoresSizeBytes = numVectors * sizeof(float);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_batchVectors, batchSizeBytes));
    CUDA_CHECK(cudaMalloc(&d_queryVector, querySizeBytes));
    CUDA_CHECK(cudaMalloc(&d_similarityScores, scoresSizeBytes));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_batchVectors, batchVectors, batchSizeBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_queryVector, queryVector, querySizeBytes, cudaMemcpyHostToDevice));

    // Kernel configuration
    int threadsPerBlock = threadsPer_Block;
    int blocksPerGrid = std::min(65535, static_cast<int>((numVectors + threadsPerBlock - 1) / threadsPerBlock));
    size_t sharedMemSize = vectorDim * sizeof(float);

    // Launch kernel
    optimizedCosineSimilarityKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
        d_batchVectors,
        d_queryVector,
        d_similarityScores,
        queryNorm,
        numVectors,
        vectorDim
    );

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Synchronize device
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(similarityScores, d_similarityScores, scoresSizeBytes, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_batchVectors));
    CUDA_CHECK(cudaFree(d_queryVector));
    CUDA_CHECK(cudaFree(d_similarityScores));
}

// Atomic implementation
__global__ void computeDotAndNorm(
    const float* dataBatch,
    const float* queryVector,
    float* partialDots,
    float* partialNorms,
    size_t numVectors,
    size_t dim
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < numVectors * dim;
         index += blockDim.x * gridDim.x) {
        size_t vectorIndex = index / dim;
        size_t dimIndex = index % dim;
        atomicAdd(&partialDots[vectorIndex], dataBatch[index] * queryVector[dimIndex]);
        atomicAdd(&partialNorms[vectorIndex], dataBatch[index] * dataBatch[index]);
    }
}

__global__ void finalizeSimilarityScores(
    float* partialDots,
    const float* partialNorms,
    float normQuery,
    size_t numVectors
) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < numVectors;
         index += blockDim.x * gridDim.x) {
        partialDots[index] /= (sqrtf(partialNorms[index]) * normQuery + 1e-8f);
    }
}

// Host function for the CUDA implementation
void computeCosineSimilaritiesAtomicOptimized(
    const float* dataBatch,
    const float* queryVector,
    float* similarityScores,
    size_t numVectors,
    size_t dim,
    int threadsPer_Block = 256
) {
    // Compute the norm of the query vector on the host
    float normQuery = computeHostVectorNorm(queryVector, dim);

    // Allocate device memory
    float* deviceBatch;
    float* deviceQuery;
    float* devicePartialDots;
    float* devicePartialNorms;
    size_t sizeBatchVectors = numVectors * dim * sizeof(float);
    size_t sizeQueryVector = dim * sizeof(float);
    size_t sizeResults = numVectors * sizeof(float);

    CUDA_CHECK(cudaMalloc(&deviceBatch, sizeBatchVectors));
    CUDA_CHECK(cudaMalloc(&deviceQuery, sizeQueryVector));
    CUDA_CHECK(cudaMalloc(&devicePartialDots, sizeResults));
    CUDA_CHECK(cudaMalloc(&devicePartialNorms, sizeResults));

    CUDA_CHECK(cudaMemcpy(deviceBatch, dataBatch, sizeBatchVectors, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceQuery, queryVector, sizeQueryVector, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(devicePartialDots, 0, sizeResults));
    CUDA_CHECK(cudaMemset(devicePartialNorms, 0, sizeResults));

    // Execute kernels
    int threadsPerBlock = threadsPer_Block;
    int numBlocks = std::min(65535, static_cast<int>((numVectors * dim + threadsPerBlock - 1) / threadsPerBlock));
    computeDotAndNorm<<<numBlocks, threadsPerBlock>>>(
        deviceBatch,
        deviceQuery,
        devicePartialDots,
        devicePartialNorms,
        numVectors,
        dim
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    numBlocks = std::min(65535, static_cast<int>((numVectors + threadsPerBlock - 1) / threadsPerBlock));
    finalizeSimilarityScores<<<numBlocks, threadsPerBlock>>>(
        devicePartialDots,
        devicePartialNorms,
        normQuery,
        numVectors
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host and clean up
    CUDA_CHECK(cudaMemcpy(similarityScores, devicePartialDots, sizeResults, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(deviceBatch));
    CUDA_CHECK(cudaFree(deviceQuery));
    CUDA_CHECK(cudaFree(devicePartialDots));
    CUDA_CHECK(cudaFree(devicePartialNorms));
}
