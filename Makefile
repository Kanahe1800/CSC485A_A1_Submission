all:
	nvcc -O3 -std=c++20 -arch=native main.cu cpu_baseline.cu radix_sort.cu opposing_sort.cu -o solution.exe

clean:
	del /Q solution.exe solution.exp solution.lib 2>nul