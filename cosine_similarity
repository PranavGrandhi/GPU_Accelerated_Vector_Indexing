// cosine_similarity.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iostream>
#include <vector>

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
