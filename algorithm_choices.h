#pragma once

#include <vector>

#include "data_types.h"

namespace csc485b {
namespace a1 {
namespace cpu {

void run_cpu_baseline( std::vector< element_t > data, std::size_t switch_at, std::size_t n );

} // namespace cpu


namespace opposing {

void run_opposing_gpu_soln( std::vector< element_t > data, std::size_t switch_at, std::size_t n );

} // namespace opposing

// https://www.geeksforgeeks.org/cpp/namespace-c-set-3-creating-header-nesting-aliasing-accessing/
namespace radix {
    void run_radix_gpu_soln( std::vector< element_t > data, std::size_t n );
} // namespace radix
} // namespace a1
} // namespace csc485b