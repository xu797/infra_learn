#include <iostream>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

#include "common.h"
#include "tester.h"

// using namespace nvcuda;
template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16>
__global__ void hgemm_naive_kernel(half *A, half *B, half *C, int M, int N, int K)
{
    const int NUM_K_TILES = div_ceil(K, WMMA_K);
    const int load_gemm_a_m = blockIdx.y * WMMA_M; //blockidx.y:[0, 32]
    const int load_gemm_b_n = blockIdx.x * WMMA_N; //blockidx.x:[0, 128]

    if(load_gemm_a_m >= M && load_gemm_b_n >= N)
    {
        return;
    }

    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
    nvcuda::wmma::fill_fragment(C_frag, 0.0);

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> A_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> B_frag;

#pragma unroll
    for(int k = 0; k < NUM_K_TILES; ++k)
    {
        nvcuda::wmma::load_matrix_sync(A_frag, A + load_gemm_a_m * K + k * WMMA_K, K);
        nvcuda::wmma::load_matrix_sync(B_frag, B + load_gemm_b_n + (k * WMMA_K) * N, N);

        nvcuda::wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
        // __syncthreads();
    }

    nvcuda::wmma::store_matrix_sync(C + load_gemm_a_m * N + load_gemm_b_n, C_frag, N, nvcuda::wmma::mem_row_major);

}

void hgemm_naive(half *A, half *B, half *C, int M, int N, int K)
{
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    dim3 block(32);
    dim3 grid(div_ceil(N, WMMA_N), div_ceil(M, WMMA_M)); //2048/16 = 128列, 512/16=32

    hgemm_naive_kernel<WMMA_M, WMMA_N, WMMA_K><<<grid, block>>>(A, B, C, M, N, K);
}

int main()
{
    Tester tester(512, 2048, 1024, 1, 10, 100, true);
    tester.evaluate(hgemm_naive, "hgemm_naive");
    return 0;
}