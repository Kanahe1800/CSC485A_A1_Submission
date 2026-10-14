# CSC485A A1: Opposing Sort and Radix Sort

## Requirements

- NVIDIA GPU with compute capability 7.0 or newer (e.g. T4)
- CUDA Toolkit (`nvcc`)
- Windows: Visual Studio Build Tools (provides `nmake` and the C++ compiler)

## Build

From the project folder, run:

    nmake

This runs:

    nvcc -O3 -std=c++20 -arch=native main.cu cpu_baseline.cu radix_sort.cu opposing_sort.cu -o solution.exe

On Google Colab (T4), replace `-arch=native` with `-arch=sm_75`.

To remove the build output:

    nmake clean

## Run

    solution.exe

On Linux or Colab:

    ./solution.exe

## Changing the input size

Set `n` in `main.cu`, then rebuild:

    std::size_t const n = 1 << 20;   // e.g. 1024, 2049, 1 << 20

## Output

The program prints the run time of:

1. the CPU baseline (opposing sort),
2. the GPU bitonic opposing sort,
3. the GPU radix sort (ascending),

followed by correctness checks for the GPU results.