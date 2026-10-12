#include <iostream>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

#include "common.h"
#include "tester.h"

#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])

// using namespace nvcuda;
template <const int WMMA_M = 16, const int WMMA_N = 16, const int WMMA_K = 16, const int WARP_TILE_M = 4, const int WARP_TILE_N = 2>
__global__ void hgemm_v2_kernel(half *A, half *B, half *C, int M, int N, int K)
{
    //1block 256 threads,8 warp
    const int NUM_K_TILES = div_ceil(K, WMMA_K);


    //shared memory
    const int BM = WMMA_M * WARP_TILE_M;
    const int BK = WMMA_K;
    const int BN = WMMA_N * WARP_TILE_N;
    __shared__ half A_shared[BM][BK];
    __shared__ half B_shared[BK][BN];

    //warp
    const int tid = threadIdx.x;
    const int warp_id = threadIdx.x / 32;//[0-8]
    const int lane_id = threadIdx.x % 32;

    //shared memory 里面的warp级别的行和列，一共8个
    const int warp_m = warp_id / 2; //[0-3]
    const int warp_n = warp_id % 2; //[0-1]

    //load a_matrix:[64, 16], load b_matrix:[16, 32]
    //a_matrix: per thread 4 row, b_matrix: per thread 1 row, 2 col

    const int load_smem_a_m = tid / 4; //0-63
    const int load_smem_a_k = (tid % 4) * 4; // 0, 4, 8, 12

    const int load_smem_b_k = tid / 16; //0-16
    const int load_smem_b_n = (tid % 16) * 2;// 0 2 4 ...

    //global m, n
    const int load_gmem_a_m = blockIdx.y * WMMA_M * WARP_TILE_M + load_smem_a_m; //blockidx.y:[0, 8]
    const int load_gmem_b_n = blockIdx.x * WMMA_N * WARP_TILE_N + load_smem_b_n; //blockidx.x:[0, 64]


    if(load_gmem_a_m >= M && load_gmem_b_n >= N)
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
        int load_gmem_a_k  = k * WMMA_K + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;

        int load_gmem_b_k = k * WMMA_K + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
        
        LDST64BITS(A_shared[load_smem_a_m][load_smem_a_k]) = LDST64BITS(A[load_gmem_a_addr]);
        LDST32BITS(B_shared[load_smem_b_k][load_smem_b_n]) = LDST32BITS(B[load_gmem_b_addr]);

        __syncthreads();

        nvcuda::wmma::load_matrix_sync(A_frag, &A_shared[warp_m * WMMA_M][0], BK);
        nvcuda::wmma::load_matrix_sync(B_frag, &B_shared[0][warp_n * WMMA_N], BN);

        nvcuda::wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
        
        __syncthreads();
    }

    //现在实际有8个c_frag 因为fragment是warp级别的
    const int store_gmem_a_m = blockIdx.y * BM + warp_m * WMMA_M;
    const int store_gmem_a_n = blockIdx.x * BN + warp_n * WMMA_N;
    nvcuda::wmma::store_matrix_sync(C + store_gmem_a_m * N + store_gmem_a_n, C_frag, N, nvcuda::wmma::mem_row_major);

}

void hgemm_v2(half *A, half *B, half *C, int M, int N, int K)
{
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    const int WARP_TILE_M = 4;
    const int WARP_TILE_N = 2;

    dim3 block(256); //16 * 16
    dim3 grid(div_ceil(N, WMMA_N * WARP_TILE_N), div_ceil(M, WMMA_M * WARP_TILE_M)); //2048/16/2 = 64列, 512/16/4=8

    hgemm_v2_kernel<WMMA_M, WMMA_N, WMMA_K, WARP_TILE_M, WARP_TILE_N><<<grid, block>>>(A, B, C, M, N, K);
}

int main()
{
    Tester tester(512, 2048, 1024, 1, 10, 100, true);
    tester.evaluate(hgemm_v2, "hgemm_v2");
    return 0;
}