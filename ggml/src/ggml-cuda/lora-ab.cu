#include "lora-ab.cuh"
#include "vecdotq.cuh"

#include <cooperative_groups.h>

// Fused LoRA branch for one token: dst = B * q8(A * q8(x)). One cooperative launch, grid barrier between A and B.
// B weights are loaded before the barrier. Same q8_1 rounding and chunk dot as the mmvq path.

#define LORA_AB_THREADS 512
#define LORA_AB_WARPS   (LORA_AB_THREADS / 32)
#define LORA_AB_R       512 // rank
#define LORA_AB_LB      (LORA_AB_R / QK8_1) // chunks per B row, lanes per B row
#define LORA_AB_ROUNDS  8

struct lora_chunk {
    int16_t q[4];
    float   d2;
};

static __device__ __forceinline__ lora_chunk lora_load_chunk(const block_pq2_0 * row, const int c) {
    const block_pq2_0 * blk = row + (c >> 2);
    const int16_t     * qs  = (const int16_t *) blk->qs + (c & 3)*4;
    lora_chunk w;
    w.d2 = __half2float(blk->d);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        w.q[j] = qs[j];
    }
    return w;
}

static __device__ __forceinline__ float lora_dot_chunk(const lora_chunk & w, const block_q8_1 * y) {
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q  = w.q[j];
        const int u  = get_int_b4(y->qs, j*2 + 0);
        const int v  = get_int_b4(y->qs, j*2 + 1);
        const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
        const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
        sumi = ggml_cuda_dp4a(u, __byte_perm(qe, qo, 0x5140), sumi);
        sumi = ggml_cuda_dp4a(v, __byte_perm(qe, qo, 0x7362), sumi);
    }
    return w.d2 * __low2float(y->ds) * sumi;
}

// one warp quantizes four 32-element blocks per step, 8 lanes per block, 4 elements per lane
static __device__ __forceinline__ void lora_quantize(const float * x, block_q8_1 * y, const int nblk, const int warp, const int lane, const bool cg) {
    const int sub = lane & 7;
    for (int b = warp*4 + (lane >> 3); b < nblk; b += LORA_AB_WARPS*4) {
        const float4 v = cg ? __ldcg((const float4 *) (x + b*QK8_1) + sub) : ((const float4 *) (x + b*QK8_1))[sub];
        float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        float sum  = (v.x + v.y) + (v.z + v.w);
        amax = warp_reduce_max<8>(amax);
        sum  = warp_reduce_sum<8>(sum);
        const float d = amax / 127.0f;
        const float f = amax == 0.0f ? 0.0f : 1.0f;
        const int q0 = (int) (f*roundf(v.x/d)) & 0xFF;
        const int q1 = (int) (f*roundf(v.y/d)) & 0xFF;
        const int q2 = (int) (f*roundf(v.z/d)) & 0xFF;
        const int q3 = (int) (f*roundf(v.w/d)) & 0xFF;
        ((int *) y[b].qs)[sub] = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
        if (sub == 0) {
            y[b].ds = make_half2(__float2half(d), __float2half(sum));
        }
    }
}

template <int kch> // K / 32 / 32, chunks of A per lane
__global__ void __launch_bounds__(LORA_AB_THREADS)
lora_ab_pq2(const float * x, const block_pq2_0 * A, const block_pq2_0 * B, float * t, float * dst, const int K, const int N) {
    extern __shared__ block_q8_1 ys[]; // K/32 blocks of x, then R/32 blocks of t

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nw   = gridDim.x * (LORA_AB_THREADS / 32);
    const int gw   = blockIdx.x * (LORA_AB_THREADS / 32) + warp;

    // B rows: two per warp and round, lanes 0-15 and 16-31 (R/32 = 16 chunks per row)
    const int half = lane / LORA_AB_LB;
    const int cb   = lane % LORA_AB_LB;
    lora_chunk wb[LORA_AB_ROUNDS];
#pragma unroll
    for (int k = 0; k < LORA_AB_ROUNDS; ++k) {
        const int row = (gw + k*nw)*2 + half;
        if (row < N) {
            wb[k] = lora_load_chunk(B + (int64_t) row*(LORA_AB_R/QK_PQ2_0), cb);
        }
    }

    lora_quantize(x, ys, K/QK8_1, warp, lane, false);
    __syncthreads();

    // A rows, 4 per warp and round; the weights of the rows are loaded first
    const int krow = K/QK_PQ2_0;
    for (int r0 = gw*4; r0 < LORA_AB_R; r0 += nw*4) {
        lora_chunk wa[4][kch];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int j = 0; j < kch; ++j) {
                wa[i][j] = lora_load_chunk(A + (int64_t) (r0 + i)*krow, lane + 32*j);
            }
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float s = 0.0f;
#pragma unroll
            for (int j = 0; j < kch; ++j) {
                s += lora_dot_chunk(wa[i][j], ys + lane + 32*j);
            }
            s = warp_reduce_sum<WARP_SIZE>(s);
            if (lane == 0) {
                t[r0 + i] = s;
            }
        }
    }

    cooperative_groups::this_grid().sync();

    block_q8_1 * yt = ys + K/QK8_1;
    lora_quantize(t, yt, LORA_AB_LB, warp, lane, true);
    __syncthreads();

#pragma unroll
    for (int k = 0; k < LORA_AB_ROUNDS; ++k) {
        const int row = (gw + k*nw)*2 + half;
        float s = 0.0f;
        if (row < N) {
            s = lora_dot_chunk(wb[k], yt + cb);
        }
        s = warp_reduce_sum<LORA_AB_LB>(s);
        if (cb == 0 && row < N) {
            dst[row] = s;
        }
    }
}

bool ggml_cuda_lora_ab_supported(const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * x, const int cc, const int n_sm) {
    int coop = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, ggml_cuda_get_device()));
    const int64_t K = a->ne[0];
    const int64_t N = b->ne[1];
    return coop && GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING &&
        a->type == GGML_TYPE_PQ2_0 && b->type == GGML_TYPE_PQ2_0 && x->type == GGML_TYPE_F32 &&
        (K == 2048 || K == 4096) && a->ne[1] == LORA_AB_R && b->ne[0] == LORA_AB_R &&
        a->ne[2] == 1 && a->ne[3] == 1 && b->ne[2] == 1 && b->ne[3] == 1 &&
        x->ne[0] == K && x->ne[1] == 1 && x->ne[2] == 1 && x->ne[3] == 1 &&
        ggml_is_contiguous(x) && ggml_is_contiguous(a) && ggml_is_contiguous(b) &&
        N <= (int64_t) LORA_AB_ROUNDS*2*(n_sm/2)*GGML_CUDA_LORA_AB_BLOCKS_PER_SM*LORA_AB_WARPS;
}

void ggml_cuda_lora_ab(ggml_backend_cuda_context & ctx, const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * x, ggml_tensor * dst) {
    const int K    = a->ne[0];
    const int N    = b->ne[1];
    const int n_sm = ggml_cuda_info().devices[ctx.device].nsm;

    ggml_cuda_pool_alloc<float> t(ctx.pool(), LORA_AB_R);

    const float * xd = (const float *) x->data;
    const block_pq2_0 * ad = (const block_pq2_0 *) a->data;
    const block_pq2_0 * bd = (const block_pq2_0 *) b->data;
    float * td = t.get();
    float * dd = (float *) dst->data;

    const dim3   grid(n_sm*GGML_CUDA_LORA_AB_BLOCKS_PER_SM/2);
    const dim3   block(LORA_AB_THREADS);
    const size_t smem = (K/QK8_1 + LORA_AB_LB)*sizeof(block_q8_1);
    void * args[] = { &xd, &ad, &bd, &td, &dd, (void *) &K, (void *) &N };

    const void * fn = K == 2048 ? (const void *) lora_ab_pq2<2> : (const void *) lora_ab_pq2<4>;
    CUDA_CHECK(cudaLaunchCooperativeKernel(fn, grid, block, args, smem, ctx.stream()));
}
