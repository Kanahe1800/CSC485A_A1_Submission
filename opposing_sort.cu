#include "algorithm_choices.h"

#include <algorithm> // for sort
#include <chrono>    // for timing
#include <iostream>  // std::cout, std::endl

#include "cuda_common.h"

namespace csc485b {
namespace a1      {
namespace opposing     {

/**
 * The CPU baseline benefits from warm caches because the data was generated on
 * the CPU. Run the data through the GPU once with some arbitrary logic to
 * ensure that the GPU cache is warm too and the comparison is more fair.
 */
__global__
void warm_the_gpu( element_t * data, std::size_t invert_at_pos, std::size_t num_elements )
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
 * Your solution. Should match the CPU output.
 */

__global__
void single_block_opposing_sort( element_t * data, std::size_t inverse_at_pos, std::size_t num_elements, std::size_t num_elements_power_2) 
{
    int const number_of_threads = blockDim.x;
    int const th_id = threadIdx.x;
    __shared__ element_t s[ 2048 ];
    
    for ( std::size_t k = th_id; k < 2048; k += number_of_threads )// number of threads 
    {
        // https://stackoverflow.com/questions/20070374/how-do-i-set-a-uint32-to-its-maximum-value
        s [ k ] = ( k < num_elements ) ? data[ k ] : UINT32_MAX; // empty slots get the max value (padding)
    }
    __syncthreads();
    
    for (std::size_t stride_a = 1; stride_a < num_elements_power_2; stride_a <<= 1){
        for (std::size_t stride_b = stride_a; stride_b > 0; stride_b >>= 1){
            std::size_t const group = th_id / stride_b;
            std::size_t const offset = th_id % stride_b;
            std::size_t const i = group * 2 * stride_b + offset;

            std::size_t const l = i + stride_b;
            bool const desc = ( ( i / ( stride_a << 1 ) ) % 2 ) != 0;

            element_t const a  = s[ i ];
            element_t const b  = s[ l ];
            element_t const lo = min( a, b );
            element_t const hi = max( a, b );

            s[ i ] = desc ? hi : lo;
            s[ l ] = desc ? lo : hi;

            __syncthreads();
        }
    }
    for (std::size_t k = th_id; k < num_elements; k += number_of_threads ){
        int const dst = ( k < inverse_at_pos ) ? k : (num_elements - 1 + inverse_at_pos - k);
        data[ dst ] = s [ k ];
    }   
}

__global__
void opposing_sort( element_t * data, std::size_t invert_at_pos, std::size_t num_elements )
{
    int const th_id = blockIdx.x * blockDim.x + threadIdx.x;
    int const len = num_elements - invert_at_pos;
    if (th_id >= len / 2) { return; }

    int const p = invert_at_pos + th_id;
    int const q = num_elements - 1 - th_id;
    element_t const temp = data[p];
    data[ p ] = data[ q ];
    data[ q ] = temp;
    // output[ th_id ] = data[ th_id ];
}

__global__
void bitonic_step( element_t * data, std::size_t stride_a, std::size_t stride_b, std::size_t n ){
    std::size_t const t = blockIdx.x * blockDim.x + threadIdx.x;

    std::size_t const group = t / stride_b;
    std::size_t const offset = t % stride_b;
    std::size_t const i = group * 2 * stride_b + offset;
    if( i >= n ) { return; }

    std::size_t const l = i + stride_b;
    bool const desc = ( ( i / ( stride_a << 1 ) ) % 2 ) != 0;

    element_t const a  = data[ i ];
    element_t const b  = data[ l ];
    element_t const lo = min( a, b );
    element_t const hi = max( a, b );

    // https://www.w3schools.com/cpp/cpp_conditions_shorthand.asp
    data[ i ] = desc ? hi : lo;
    data[ l ] = desc ? lo : hi;
}

/**
 * Performs all the logic of allocating device vectors and copying host/data
 * vectors to the device. Times the opposing_sort() kernel with wall time,
 * but excludes set up and tear down costs such as mallocs, frees, and memcpies.
 */
void run_opposing_gpu_soln( std::vector< element_t > data, std::size_t switch_at, std::size_t n )
{
    // Kernel launch configurations. Feel free to change these.
    // This is set to maximise the size of a thread block on a T4, but it hasn't
    // been tuned. It's not known if this is optimal.
    std::size_t const threads_per_block = 1024;
    // std::size_t const num_blocks =  ( n + threads_per_block - 1 ) / threads_per_block;
    std::size_t N = 1;
    while( N < n ) N <<= 1;
    std::size_t const sort_blocks = ( N + threads_per_block - 1 ) / threads_per_block;
    // Allocate arrays on the device/GPU
    element_t * d_data;

    std::vector< element_t > expected = data;
    std::sort( expected.begin(), expected.end() );
    std::reverse( expected.begin() + switch_at, expected.end() );
    cudaMalloc( (void**) & d_data, sizeof( element_t ) * N );
    CHECK_ERROR("Allocating data array on device");

    // Copy the data from the host to the device/GPU
    cudaMemcpy( d_data, data.data(), sizeof( element_t ) * n, cudaMemcpyHostToDevice );
    CHECK_ERROR("Copying data array to device");
    // https://courses.cs.duke.edu/fall09/cps296.3/cuda_docs/html/group__CUDART__MEMORY_ge07c97b96efd09abaeb3ca3b5f8da4ee.html
    cudaMemset( d_data + n, 0xFF, sizeof( element_t ) * ( N - n ) ); 
    CHECK_ERROR("Padding data array on device");
    // Warm the cache on the GPU for a more fair comparison
    warm_the_gpu<<< sort_blocks, threads_per_block>>>( d_data, switch_at, n );
    cudaDeviceSynchronize();

    // Time the execution of the kernel that you implemented
    auto const kernel_start = std::chrono::high_resolution_clock::now();
    if (N <= threads_per_block * 2){
        single_block_opposing_sort<<< 1, threads_per_block >>>(d_data, switch_at, n, N);
    }
    else {
        for( std::size_t stride_a = 1; stride_a < N; stride_a <<= 1 ) {
            for( std::size_t stride_b = stride_a; stride_b > 0; stride_b >>= 1 ) {
                bitonic_step<<< sort_blocks, threads_per_block >>>( d_data, stride_a, stride_b, N );
            }
        }
        int tail_pairs = (n - switch_at) / 2;
        int const tail_blocks = ( tail_pairs + threads_per_block - 1 ) / threads_per_block;
        opposing_sort<<< tail_blocks, threads_per_block >>>( d_data, switch_at, n );
    }
    
    auto const kernel_end = std::chrono::high_resolution_clock::now();
    CHECK_ERROR("Executing kernel on device");

    // After the timer ends, copy the result back, free the device vector,
    // and echo out the timings and the results.
    cudaMemcpy( data.data(), d_data, sizeof( element_t ) * n, cudaMemcpyDeviceToHost );
    CHECK_ERROR("Transferring result back to host");
    cudaFree( d_data );
    CHECK_ERROR("Freeing device memory");

    std::cout << "GPU Solution time: "
              << std::chrono::duration_cast<std::chrono::nanoseconds>(kernel_end - kernel_start).count()
              << " ns" << std::endl;
    std::cout << "Opposing sort matches CPU: "
          << ( data == expected ? "yes" : "no" ) << std::endl;

    // for( auto const x : data ) std::cout << x << " "; std::cout << std::endl;
}

} // namespace opposing
} // namespace a1
} // namespace csc485b