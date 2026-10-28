// Phase breakdown for the eig2 (subspace) pipeline: cumulative
// stop-after-phase timing. Phase P kernel runs phases 0..P and exits;
// per-phase time is the difference of successive medians. GUE ensemble,
// same case seeding as bench_eig. Diagnostic tool — not shipped API.

#include "beig/hermitian_eig_gpu.cuh"
#include "ensembles.hpp"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using beig::cdouble;
using beig::gpu::dc;

// stop: 0 = input copy, 1 = +tridiag, 2 = +Sturm, 3 = +subspace,
// 4 = +back-transform (full eig2).
template <int stop>
__global__ void eig2_phase_kernel(const dc* __restrict__ Abatch, int n,
                                  uint64_t seed, double* __restrict__ wout,
                                  dc* __restrict__ vout, dc* carena,
                                  double* darena) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x, B = blockDim.x;
    __shared__ beig::gpu::ShCtx sh;
    beig::gpu::DevWork W =
        beig::gpu::make_work_d(carena + (size_t)b * beig::gpu::work_complex_count_d(n),
                               darena + (size_t)b * beig::gpu::work_double_count_d2(n),
                               n);
    double* gs = W.ds + 4 * n;
    double* G = gs + 3 * (size_t)n * n;
    const dc* A = Abatch + (size_t)b * n * n;
    for (int idx = tid; idx < n * n; idx += B) W.a[idx] = A[idx];
    __syncthreads();
    if (stop == 0) return;
    beig::gpu::tridiag_hermitian_d(W.a, n, W.d, W.e, W.vr, W.tau, W.phase,
                                   W.cs, sh);
    if (stop == 1) return;
    double* w = wout + (size_t)b * n;
    for (int k = tid; k < n; k += B)
        w[k] = beig::gpu::find_eigen_d(W.d, W.e, n, k);
    __syncthreads();
    if (stop == 2) return;
    beig::gpu::eigvecs_subspace_d(W.d, W.e, n, w,
                                  beig::gpu::mix64_d(seed + (uint64_t)b), W.z,
                                  gs, G, sh);
    if (stop == 3) return;
    beig::gpu::back_transform_d(W.vr, W.tau, W.phase, W.z, n,
                                vout + (size_t)b * n * n);
}

template <int stop>
double time_phase(const beig::gpu::Batch& B, int n, int batch, uint64_t seed,
                  int reps) {
    std::vector<double> ts;
    for (int i = 0; i < reps; ++i) {
        beig::gpu::cuda_check(cudaDeviceSynchronize(), "pre sync");
        const auto t0 = std::chrono::steady_clock::now();
        eig2_phase_kernel<stop><<<batch, beig::gpu::kThreads>>>(
            B.a, n, seed, B.w, B.v, B.carena, B.darena2);
        beig::gpu::cuda_check(cudaDeviceSynchronize(), "post sync");
        const auto t1 = std::chrono::steady_clock::now();
        ts.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    std::sort(ts.begin(), ts.end());
    return ts[ts.size() / 2];
}

} // namespace

int main(int argc, char** argv) {
    int n = 128, batch = 256, reps = 11;
    uint64_t seed = 0xBE16ULL;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--n=", 4) == 0) n = std::stoi(argv[i] + 4);
        else if (std::strncmp(argv[i], "--batch=", 8) == 0)
            batch = std::stoi(argv[i] + 8);
        else if (std::strncmp(argv[i], "--reps=", 7) == 0)
            reps = std::stoi(argv[i] + 7);
    }
    try {
        const uint64_t cseed = beig::mix64(seed + (uint64_t)n * 7919 + batch);
        std::vector<cdouble> h;
        h.reserve((size_t)batch * n * n);
        for (int b = 0; b < batch; ++b) {
            std::vector<cdouble> A =
                beigtest::random_hermitian(n, beig::mix64(cseed + (uint64_t)b));
            h.insert(h.end(), A.begin(), A.end());
        }
        beig::gpu::Batch B(h.data(), n, batch);
        B.solve2(cseed); // allocate darena2 + warmup

        const double t0 = time_phase<0>(B, n, batch, cseed, reps);
        const double t1 = time_phase<1>(B, n, batch, cseed, reps);
        const double t2 = time_phase<2>(B, n, batch, cseed, reps);
        const double t3 = time_phase<3>(B, n, batch, cseed, reps);
        const double t4 = time_phase<4>(B, n, batch, cseed, reps);
        std::printf("n=%d batch=%d median of %d (wall, ms; cumulative | phase)\n",
                    n, batch, reps);
        std::printf("copy   %8.3f | %8.3f\n", t0 * 1e3, t0 * 1e3);
        std::printf("tridiag%8.3f | %8.3f\n", t1 * 1e3, (t1 - t0) * 1e3);
        std::printf("sturm  %8.3f | %8.3f\n", t2 * 1e3, (t2 - t1) * 1e3);
        std::printf("subsp  %8.3f | %8.3f\n", t3 * 1e3, (t3 - t2) * 1e3);
        std::printf("backtr %8.3f | %8.3f\n", t4 * 1e3, (t4 - t3) * 1e3);
        std::printf("total  %8.3f\n", t4 * 1e3);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bench_phases failed: %s\n", e.what());
        return 1;
    }
    return 0;
}
