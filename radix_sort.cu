/**
 * The file in which you will implement your GPU solutions!
 */

#include "algorithm_choices.h"

#include <chrono>    // for timing
#include <iostream>  // std::cout, std::endl

#include <algorithm>

#include "cuda_common.h"

namespace{
    int const MAX_WARPS = 32;
    int const MAX_LANES = 32;
    int const RADIX_DOMAIN= 10;
}

namespace csc485b {
namespace a1      {
namespace radix     {

/**
 * The CPU baseline benefits from warm caches because the data was generated on
 * the CPU. Run the data through the GPU once with some arbitrary logic to
 * ensure that the GPU cache is warm too and the comparison is more fair.
 */
__global__
void warm_the_gpu( element_t * data, std::size_t num_elements )
{
    int const th_id = blockIdx.x * blockDim.x + threadIdx.x;

    // We know this will never be true, because of the data generator logic,
    // but I doubt that the compiler will figure it out. Thus every element
    // should be read, but none of them should be modified.
    if( th_id < num_elements && data[ th_id ] > num_elements * 100 )
    {
        ++data[ th_id ]; // should not be possible.
    }
}


/**
 * Outputs:
 *   output_histogram: a histogram of size RADIX_DOMAIN that counts how many times a digit (of a certain place value) appears in d_data
 *   block_digit_counts: a flattened 2D array such that each "row" is the block number and each "column" is the digit. Used for calculating the correct offsets for ranking
 */
__global__
void histogram( element_t * data, element_t * output_histogram, element_t * block_digit_counts, std::size_t num_elements, int num_blocks, int place_value)
{
    // init shared memory as buckets
    __shared__ element_t smem_histogram [RADIX_DOMAIN]; 

    // get both global and local thread ids
    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    int const local_th_id = (warp_id * blockDim.x) + lane_id;
    int const global_th_id = (blockIdx.x * (blockDim.x * blockDim.y) + ((warp_id * blockDim.x) + lane_id));

    // init buckets to 0
    if (local_th_id < RADIX_DOMAIN){
        smem_histogram[local_th_id] = 0; //initialize buckets to 0
    }

    __syncthreads();

    // increment bucket at digit of the current place value
    if (global_th_id < num_elements){
        int digit = (data[global_th_id] / place_value) % 10; 
        atomicAdd(&smem_histogram[digit], 1);
    }

    __syncthreads();

    // store how many times each digit appears in this block inside block_digit_counts
    // increment the output histogram with what is in shared memory buckets
    if (local_th_id < RADIX_DOMAIN){
        block_digit_counts[local_th_id * num_blocks + blockIdx.x] = smem_histogram[local_th_id];
        atomicAdd(&output_histogram[local_th_id], smem_histogram[local_th_id]);
    }

    return;
}


/**
 * Outputs:
 * output_exclusive_scan: an exclusive prefix sum of the histogram
 */
__global__
void exclusive_scan( element_t * input_histogram, element_t * output_exclusive_scan, std::size_t num_elements)
{
    __shared__ element_t smem [RADIX_DOMAIN]; 

    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    int const local_th_id = (warp_id * blockDim.x) + lane_id; // only use local th_id because we only need 1 block

    // init smem
    if (local_th_id < RADIX_DOMAIN){
        smem[local_th_id] = input_histogram[local_th_id];
        // already read, so it's safe to clear
        input_histogram[local_th_id] = 0;
    }

    __syncthreads();

    for (int stride = 1; stride < num_elements; stride *= 2) {

        element_t new_value = 0;

        // every thread in block is reading old values
        if (local_th_id < num_elements) {
            new_value = smem[local_th_id];

            if ( local_th_id >= stride) {
                new_value += smem[local_th_id - stride];
            }
        }

        __syncthreads();   

        // every thread in block is writing new values
        if (local_th_id < num_elements) {
            smem[local_th_id] = new_value;
        }

        __syncthreads(); 
    }


    //copy back
    //shift right and append a 0 to make the prefix sum exclusive
    if (local_th_id < num_elements) {
        if (local_th_id == 0)
            output_exclusive_scan[0] = 0;
        else
            output_exclusive_scan[local_th_id] = smem[local_th_id - 1];
    }

    return;
}


/**
 * Outputs:
 * output_rank: an array where each element stores its rank among elements with the same digit in its block (0 indexed)
 */
__global__
void rank( element_t * data, element_t * output_rank, std::size_t num_elements, int place_value)
{
    //rank of an element's digit in a warp
    __shared__ int warp_count[MAX_WARPS][RADIX_DOMAIN];

    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    int const local_th_id = (warp_id * blockDim.x) + lane_id;
    int const global_th_id = (blockIdx.x * (blockDim.x * blockDim.y) + ((warp_id * blockDim.x) + lane_id));

    // init all warp/digit counts to 0
    if (local_th_id < MAX_WARPS * RADIX_DOMAIN) {
        warp_count[local_th_id / RADIX_DOMAIN][local_th_id % RADIX_DOMAIN] = 0;
    }

    __syncthreads();

    int digit = 0;
    int rank_in_warp = 0;

    // calculate the rank of an element's digit in a warp
    if (global_th_id < num_elements) {
        digit = (data[global_th_id] / place_value) % 10;

        unsigned lanes_same = __match_any_sync(__activemask(), digit); // mask of lanes in current warp with the same digit
        unsigned lanes_before = (1u << lane_id) - 1; // mask of lanes before the current lane
        unsigned lanes_same_before = lanes_same & lanes_before; // mask of lanes that are both before and the same digit as the current lane and warp

        // count them
        rank_in_warp = __popc(lanes_same_before);
        int count_in_warp = __popc(lanes_same);

        if (rank_in_warp == 0) {
            warp_count[warp_id][digit] = count_in_warp;
        }

    }

    __syncthreads();

    // calculate the rank of an element's digit in a block
    if (global_th_id < num_elements) {
        int prev_warp_count = 0;

        for (int warp = 0; warp < warp_id; warp++) {
            prev_warp_count += warp_count[warp][digit];
        }

        int rank_in_block = prev_warp_count + rank_in_warp;

        output_rank[global_th_id] = rank_in_block;
    }
}

/**
 * Outputs:
 * output: an array containing the final sorted version of our data
 */
__global__
void scatter( element_t * data, element_t *output, element_t * exclusive_scan_data, element_t * rank, element_t * block_digit_counts,
            std::size_t num_elements, std::size_t num_exclusive_scan_data, int num_blocks, int place_value)
{

    int warp_id = threadIdx.y;
    int lane_id = threadIdx.x;
    int const global_th_id = (blockIdx.x * (blockDim.x * blockDim.y) + ((warp_id * blockDim.x) + lane_id));

    if (global_th_id < num_elements){
        // read target data
        int target_data = data[global_th_id];
        int digit = (data[global_th_id] / place_value) % 10; 

        //sum of the times digit appears in all of the previous blocks to offset the local rank
        int earlier_blocks_count = 0;
        for (int block = 0; block < blockIdx.x; block++) {
            earlier_blocks_count += block_digit_counts[digit * num_blocks + block];
        }

        int target_index = exclusive_scan_data[digit] + rank[global_th_id] + earlier_blocks_count;

        //update output with target data
        output[target_index] = target_data;

    }

    return;
}


/**
 * Performs all the logic of allocating device vectors and copying host/input
 * vectors to the device. Times the opposing_sort() kernel with wall time,
 * but excludes set up and tear down costs such as mallocs, frees, and memcpies.
 */
void run_radix_gpu_soln( std::vector< element_t > data, std::size_t n )
{
    // Kernel launch configurations. Feel free to change these.
    // This is set to maximise the size of a thread block on a T4, but it hasn't
    // been tuned. It's not known if this is optimal.
    dim3 threads(MAX_LANES, MAX_WARPS);
    std::size_t const threads_per_block = threads.x * threads.y;
    std::size_t const num_blocks =  ( n + threads_per_block - 1 ) / threads_per_block;

    // Allocate arrays on the device/GPU
    element_t * d_data;
    element_t * d_output_histogram;
    element_t * d_output_exclusive_scan;
    element_t * d_output_rank;
    element_t * d_output;
    element_t * d_block_digit_counts;

    cudaMalloc( (void**) & d_data, sizeof( element_t ) * n );
    cudaMalloc( (void**) & d_output_histogram, sizeof( element_t ) * RADIX_DOMAIN);
    cudaMalloc( (void**) & d_output_exclusive_scan, sizeof( element_t ) * RADIX_DOMAIN);
    cudaMalloc( (void**) & d_output_rank, sizeof( element_t ) * n );
    cudaMalloc( (void**) & d_output, sizeof( element_t ) * n );
    cudaMalloc( (void**) & d_block_digit_counts, sizeof(element_t) * num_blocks * RADIX_DOMAIN);
    CHECK_ERROR("Allocating input array on device");

    // Copy the input from the host to the device/GPU, while setting other arrays to 0
    cudaMemcpy( d_data, data.data(), sizeof( element_t ) * n, cudaMemcpyHostToDevice );
    cudaMemset(d_output_histogram, 0, sizeof(element_t) * RADIX_DOMAIN); 
    CHECK_ERROR("Copying input array to device");

    // Warm the cache on the GPU for a more fair comparison
    warm_the_gpu<<< num_blocks, threads>>>( d_data, n );

    element_t max_value = *std::max_element(data.begin(), data.end());
    int last_place = 1;
    while (max_value / last_place >= 10){
        last_place *= 10;
    }

    // Time the execution of the kernel that you implemented
    auto const kernel_start = std::chrono::high_resolution_clock::now();


    // start host loop
    for (int place_value=1; place_value<=last_place; place_value*=10){
        histogram<<<num_blocks, threads>>>(d_data, d_output_histogram, d_block_digit_counts, n, num_blocks, place_value);
        exclusive_scan<<<1, threads>>>( d_output_histogram, d_output_exclusive_scan, RADIX_DOMAIN);
        rank<<<num_blocks, threads>>>(d_data, d_output_rank, n, place_value);
        scatter<<<num_blocks, threads>>>(d_data, d_output, d_output_exclusive_scan, d_output_rank, d_block_digit_counts, n, RADIX_DOMAIN, num_blocks, place_value);

        std::swap(d_data, d_output);
    }

    auto const kernel_end = std::chrono::high_resolution_clock::now();
    CHECK_ERROR("Executing kernel on device");

    // After the timer ends, copy the result back, free the device vector,
    // and echo out the timings and the results.
    cudaMemcpy( data.data(), d_data, sizeof( element_t ) * n, cudaMemcpyDeviceToHost );
    CHECK_ERROR("Transferring result back to host");

    cudaFree( d_data );
    cudaFree( d_output_histogram );
    cudaFree( d_output_exclusive_scan );
    cudaFree( d_output_rank );
    cudaFree( d_output );
    cudaFree( d_block_digit_counts );
    CHECK_ERROR("Freeing device memory");

    std::cout << "GPU Radix Solution time: "
              << std::chrono::duration_cast<std::chrono::nanoseconds>(kernel_end - kernel_start).count()
              << " ns" << std::endl;

    // for( auto const x : data ) std::cout << x << " "; std::cout << std::endl;
    std::cout << "Radix result sorted ascending: "
              << ( std::is_sorted( data.begin(), data.end() ) ? "yes" : "no" ) << std::endl;
}

} // namespace radix
} // namespace a1
} // namespace csc485b