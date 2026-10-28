// Benchmark: beig GPU kernels (one block per matrix; per-vector inverse
// iteration and the subspace-iteration variant) vs the vendor batched
// eigensolver (cusolverDnXsyevBatched, the CUDA 13 batched path) and a naive
// per-matrix cusolverDnXsyevd loop. GUE ensembles, uniform n per case.
//
// Layout note: the input slabs are row-major Hermitian; the vendor calls
// read them as column-major, which is conj(A) — identical eigenvalues, and
// the eigenvector work is the same, so timings are representative. Our
// kernel copies A into its workspace and leaves the input untouched; the
// vendor overwrites A, so each vendor rep starts with an untimed
// device-to-device restore.
//
// Correctness precondition: our GPU eigenvalues must agree with the CPU
// reference (itself Eigen-gated) at 1e-12 * max(1, ||H||_F) on every matrix
// of every case before any timing runs; a mismatch aborts the benchmark.
//
// Per rep both a wall-clock time (sync-bracketed, includes host-side launch
// work — part of the vendor gap) and a CUDA-event GPU-stream time are
// recorded; raw reps are preserved in the JSON. Absolute times vary with
// machine state; report medians and observed spreads.

#include "beig/hermitian_eig_gpu.cuh"
#include "ensembles.hpp"

#include <cusolverDn.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using beig::cdouble;
using beig::gpu::dc;

void cu_check(cusolverStatus_t st, const char* what) {
    if (st != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error(std::string(what) + ": cusolver status " +
                                 std::to_string((int)st));
}

struct RngCase {
    int n, batch;
    std::vector<cdouble> h; // batch slabs of row-major n*n Hermitian
};

RngCase make_case(int n, int batch, uint64_t seed) {
    RngCase c;
    c.n = n;
    c.batch = batch;
    c.h.reserve((size_t)batch * n * n);
    for (int b = 0; b < batch; ++b) {
        std::vector<cdouble> A =
            beigtest::random_hermitian(n, beig::mix64(seed + (uint64_t)b));
        c.h.insert(c.h.end(), A.begin(), A.end());
    }
    return c;
}

// Gate: max |w_gpu - w_cpu| <= 1e-12 * max(1, ||H||_F) for every matrix,
// on both GPU paths.
void precondition(const RngCase& c, uint64_t seed) {
    double worst = 0.0, worst2 = 0.0;
    std::vector<double> w_gpu, w_gpu2;
    std::vector<cdouble> v_gpu, v_gpu2;
    beig::gpu::solve_batch_gpu(c.h.data(), c.n, c.batch, seed, w_gpu, v_gpu);
    beig::gpu::solve_batch_gpu2(c.h.data(), c.n, c.batch, seed, w_gpu2,
                                v_gpu2);
    for (int b = 0; b < c.batch; ++b) {
        const cdouble* A = c.h.data() + (size_t)b * c.n * c.n;
        const beig::EigResult r =
            beig::hermitian_eig(A, c.n, beig::mix64(seed + (uint64_t)b));
        double fro = 0.0;
        for (int i = 0; i < c.n * c.n; ++i) fro += std::norm(A[i]);
        const double scale = std::max(1.0, sqrt(fro));
        for (int i = 0; i < c.n; ++i) {
            worst = std::max(
                worst, std::abs(w_gpu[(size_t)b * c.n + i] - r.w[i]) / scale);
            worst2 = std::max(
                worst2,
                std::abs(w_gpu2[(size_t)b * c.n + i] - r.w[i]) / scale);
        }
    }
    std::printf("  precondition: worst GPU-vs-CPU eval %.3e, subspace %.3e "
                "(tol 1e-12)\n",
                worst, worst2);
    if (!(worst < 1e-12) || !(worst2 < 1e-12))
        throw std::runtime_error("GPU-vs-CPU precondition failed");
}

struct RepTimes {
    std::vector<double> wall_s, gpu_s;
    double wall_median = 0, wall_min = 0, wall_max = 0;
    double gpu_median = 0, gpu_min = 0, gpu_max = 0;
    void finish() {
        const auto stats = [](std::vector<double> v, double& med, double& mn,
                              double& mx) {
            std::sort(v.begin(), v.end());
            med = v[v.size() / 2];
            mn = v.front();
            mx = v.back();
        };
        stats(wall_s, wall_median, wall_min, wall_max);
        stats(gpu_s, gpu_median, gpu_min, gpu_max);
    }
};

template <typename F>
RepTimes time_reps(F&& f, int reps) {
    RepTimes r;
    cudaEvent_t ev0, ev1;
    beig::gpu::cuda_check(cudaEventCreate(&ev0), "event create");
    beig::gpu::cuda_check(cudaEventCreate(&ev1), "event create");
    for (int i = 0; i < reps; ++i) {
        beig::gpu::cuda_check(cudaDeviceSynchronize(), "pre-rep sync");
        const auto t0 = std::chrono::steady_clock::now();
        beig::gpu::cuda_check(cudaEventRecord(ev0), "event record");
        f();
        beig::gpu::cuda_check(cudaEventRecord(ev1), "event record");
        beig::gpu::cuda_check(cudaDeviceSynchronize(), "post-rep sync");
        const auto t1 = std::chrono::steady_clock::now();
        float ms = 0.0f;
        beig::gpu::cuda_check(cudaEventElapsedTime(&ms, ev0, ev1),
                              "event elapsed");
        r.wall_s.push_back(std::chrono::duration<double>(t1 - t0).count());
        r.gpu_s.push_back((double)ms * 1e-3);
    }
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    r.finish();
    return r;
}

std::string timestamp_utc() {
    const std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
    return buf;
}

std::vector<int> parse_list(const std::string& s) {
    std::vector<int> out;
    size_t pos = 0;
    while (pos <= s.size()) {
        const size_t comma = s.find(',', pos);
        out.push_back(std::stoi(s.substr(pos, comma - pos)));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return out;
}

struct CaseResult {
    int n, batch;
    RepTimes beig, beig2, vendor, loop;
};

void json_times(std::ofstream& f, const RepTimes& r, int indent) {
    const std::string pad(indent, ' ');
    f << pad << "{\n"
      << pad << "  \"wall_median_s\": " << r.wall_median << ",\n"
      << pad << "  \"wall_min_s\": " << r.wall_min << ",\n"
      << pad << "  \"wall_max_s\": " << r.wall_max << ",\n"
      << pad << "  \"gpu_median_s\": " << r.gpu_median << ",\n"
      << pad << "  \"gpu_min_s\": " << r.gpu_min << ",\n"
      << pad << "  \"gpu_max_s\": " << r.gpu_max << ",\n"
      << pad << "  \"wall_reps_s\": [";
    for (size_t i = 0; i < r.wall_s.size(); ++i)
        f << (i ? ", " : "") << r.wall_s[i];
    f << "],\n" << pad << "  \"gpu_reps_s\": [";
    for (size_t i = 0; i < r.gpu_s.size(); ++i)
        f << (i ? ", " : "") << r.gpu_s[i];
    f << "]\n" << pad << "}";
}

} // namespace

int main(int argc, char** argv) {
    std::vector<int> ns = {32, 64, 128};
    std::vector<int> batches = {64, 256, 1024};
    int reps = 11;
    uint64_t seed = 0xBE16ULL;
    std::string out_path = std::string(BEIG_DATA_DIR) + "/eig_gpu_bench.json";
    for (int i = 1; i < argc; ++i) {
        try {
            if (std::strncmp(argv[i], "--n=", 4) == 0)
                ns = parse_list(argv[i] + 4);
            else if (std::strncmp(argv[i], "--batch=", 8) == 0)
                batches = parse_list(argv[i] + 8);
            else if (std::strncmp(argv[i], "--reps=", 7) == 0)
                reps = std::stoi(argv[i] + 7);
            else if (std::strncmp(argv[i], "--seed=", 7) == 0)
                seed = std::stoull(argv[i] + 7, nullptr, 0);
            else if (std::strncmp(argv[i], "--out=", 6) == 0)
                out_path = argv[i] + 6;
            else
                throw std::invalid_argument(std::string("unknown argument: ") +
                                            argv[i]);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "bench_eig: %s\n", e.what());
            std::fprintf(stderr,
                         "usage: bench_eig [--n=32,64,128] "
                         "[--batch=64,256,1024] [--reps=11] [--seed=0x...] "
                         "[--out=PATH]\n");
            return 1;
        }
    }

    try {
        int dev = 0;
        beig::gpu::cuda_check(cudaGetDevice(&dev), "cudaGetDevice");
        cudaDeviceProp prop;
        beig::gpu::cuda_check(cudaGetDeviceProperties(&prop, dev),
                              "device props");
        int mem_clk = 0;
        beig::gpu::cuda_check(cudaDeviceGetAttribute(
                                  &mem_clk, cudaDevAttrMemoryClockRate, dev),
                              "memory clock");
        int driver_v = 0, runtime_v = 0;
        beig::gpu::cuda_check(cudaDriverGetVersion(&driver_v), "driver");
        beig::gpu::cuda_check(cudaRuntimeGetVersion(&runtime_v), "runtime");
        std::printf("gpu: %s sm_%d%d, %d SMs, %zu MiB, driver %d, runtime %d\n",
                    prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                    prop.totalGlobalMem >> 20, driver_v, runtime_v);
        std::printf("median of %d reps; wall = sync-bracketed host time, "
                    "gpu = CUDA-event stream time\n\n",
                    reps);

        cusolverDnHandle_t handle = nullptr;
        cu_check(cusolverDnCreate(&handle), "cusolverDnCreate");
        cusolverDnParams_t params = nullptr;
        cu_check(cusolverDnCreateParams(&params), "create params");

        std::printf("%4s %6s | %10s %10s | %10s %10s | %10s %10s\n", "n",
                    "batch", "beig ms", "beig gpu", "subs ms", "subs gpu",
                    "vendor ms", "vendor gpu");

        std::vector<CaseResult> results;
        for (const int n : ns)
            for (const int batch : batches) {
                const uint64_t cseed =
                    beig::mix64(seed + (uint64_t)n * 7919 + batch);
                const RngCase c = make_case(n, batch, cseed);
                precondition(c, cseed);

                const size_t asz = (size_t)batch * n * n;
                // Our kernel path: resident inputs, arena preallocated.
                beig::gpu::Batch B(c.h.data(), n, batch);

                // Vendor path: strided-batched XsyevBatched over the same
                // slabs (read as conj(A); eigenvalues identical).
                dc* a_vendor = nullptr;
                dc* a_pristine = nullptr;
                double* w_vendor = nullptr;
                beig::gpu::cuda_check(
                    cudaMalloc(&a_vendor, asz * sizeof(dc)), "vendor a");
                beig::gpu::cuda_check(
                    cudaMalloc(&a_pristine, asz * sizeof(dc)), "pristine a");
                beig::gpu::cuda_check(
                    cudaMalloc(&w_vendor, (size_t)batch * n * sizeof(double)),
                    "vendor w");
                beig::gpu::cuda_check(
                    cudaMemcpy(a_pristine, c.h.data(), asz * sizeof(dc),
                               cudaMemcpyHostToDevice),
                    "pristine upload");

                size_t dev_ws = 0, host_ws = 0;
                cu_check(cusolverDnXsyevBatched_bufferSize(
                             handle, params, CUSOLVER_EIG_MODE_VECTOR,
                             CUBLAS_FILL_MODE_LOWER, n, CUDA_C_64F, a_vendor,
                             n, CUDA_R_64F, w_vendor, CUDA_C_64F, &dev_ws,
                             &host_ws, batch),
                         "vendor bufferSize");
                void* ws_dev = nullptr;
                beig::gpu::cuda_check(cudaMalloc(&ws_dev, dev_ws ? dev_ws : 1),
                                      "vendor workspace");
                std::vector<char> ws_host(host_ws);
                std::vector<int> info(batch, -1);

                const auto restore = [&]() {
                    beig::gpu::cuda_check(
                        cudaMemcpy(a_vendor, a_pristine, asz * sizeof(dc),
                                   cudaMemcpyDeviceToDevice),
                        "vendor restore");
                };
                restore();
                cu_check(cusolverDnXsyevBatched(
                             handle, params, CUSOLVER_EIG_MODE_VECTOR,
                             CUBLAS_FILL_MODE_LOWER, n, CUDA_C_64F, a_vendor,
                             n, CUDA_R_64F, w_vendor, CUDA_C_64F, ws_dev,
                             dev_ws, ws_host.data(), host_ws, info.data(),
                             batch),
                         "vendor warmup");
                beig::gpu::cuda_check(cudaDeviceSynchronize(), "vendor sync");
                for (size_t i = 0; i < info.size(); ++i)
                    if (info[i] != 0)
                        throw std::runtime_error(
                            "vendor info nonzero at batch " +
                            std::to_string(i));

                // Naive per-matrix loop baseline (Xsyevd per slab).
                size_t l_dev = 0, l_host = 0;
                cu_check(cusolverDnXsyevd_bufferSize(
                             handle, params, CUSOLVER_EIG_MODE_VECTOR,
                             CUBLAS_FILL_MODE_LOWER, n, CUDA_C_64F, a_vendor,
                             n, CUDA_R_64F, w_vendor, CUDA_C_64F, &l_dev,
                             &l_host),
                         "loop bufferSize");
                void* l_ws = nullptr;
                beig::gpu::cuda_check(cudaMalloc(&l_ws, l_dev ? l_dev : 1),
                                      "loop workspace");
                std::vector<char> l_host_ws(l_host);
                int l_info = -1;

                CaseResult cr;
                cr.n = n;
                cr.batch = batch;

                cr.beig = time_reps(
                    [&]() {
                        beig::gpu::eig_batch_kernel<<<batch, beig::gpu::kThreads>>>(
                            B.a, n, cseed, B.w, B.v, B.carena, B.darena);
                    },
                    reps);

                B.solve2(cseed); // allocates darena2; also the warmup
                cr.beig2 = time_reps(
                    [&]() {
                        beig::gpu::eig2_batch_kernel<<<batch, beig::gpu::kThreads>>>(
                            B.a, n, cseed, B.w, B.v, B.carena, B.darena2);
                    },
                    reps);

                cr.vendor = time_reps(
                    [&]() {
                        restore();
                        cu_check(cusolverDnXsyevBatched(
                                     handle, params, CUSOLVER_EIG_MODE_VECTOR,
                                     CUBLAS_FILL_MODE_LOWER, n, CUDA_C_64F,
                                     a_vendor, n, CUDA_R_64F, w_vendor,
                                     CUDA_C_64F, ws_dev, dev_ws,
                                     ws_host.data(), host_ws, info.data(),
                                     batch),
                                 "vendor rep");
                    },
                    reps);

                cr.loop = time_reps(
                    [&]() {
                        restore();
                        for (int b = 0; b < batch; ++b)
                            cusolverDnXsyevd(
                                handle, params, CUSOLVER_EIG_MODE_VECTOR,
                                CUBLAS_FILL_MODE_LOWER, n, CUDA_C_64F,
                                a_vendor + (size_t)b * n * n, n, CUDA_R_64F,
                                w_vendor + (size_t)b * n, CUDA_C_64F, l_ws,
                                l_dev, l_host_ws.data(), l_host, &l_info);
                    },
                    reps);

                results.push_back(cr);
                std::printf("%4d %6d | %10.3f %10.3f | "
                            "%10.3f %10.3f | %10.3f %10.3f\n",
                            n, batch, cr.beig.wall_median * 1e3,
                            cr.beig.gpu_median * 1e3,
                            cr.beig2.wall_median * 1e3,
                            cr.beig2.gpu_median * 1e3,
                            cr.vendor.wall_median * 1e3,
                            cr.vendor.gpu_median * 1e3);
                std::printf("            loop wall %.3f ms; wall spread ms "
                            "[min,max]: beig [%.3f,%.3f] subs [%.3f,%.3f] "
                            "vendor [%.3f,%.3f] loop [%.3f,%.3f]\n",
                            cr.loop.wall_median * 1e3,
                            cr.beig.wall_min * 1e3, cr.beig.wall_max * 1e3,
                            cr.beig2.wall_min * 1e3, cr.beig2.wall_max * 1e3,
                            cr.vendor.wall_min * 1e3,
                            cr.vendor.wall_max * 1e3, cr.loop.wall_min * 1e3,
                            cr.loop.wall_max * 1e3);

                cudaFree(a_vendor);
                cudaFree(a_pristine);
                cudaFree(w_vendor);
                cudaFree(ws_dev);
                cudaFree(l_ws);
            }

        cusolverDnDestroyParams(params);
        cusolverDnDestroy(handle);

        std::ofstream f(out_path);
        if (!f) throw std::runtime_error("cannot open " + out_path);
        f.precision(17);
        f << "{\n"
          << "  \"timestamp_utc\": \"" << timestamp_utc() << "\",\n"
          << "  \"seed\": \"" << seed << "\",\n"
          << "  \"gpu\": \"" << prop.name << "\",\n"
          << "  \"compute_capability\": \"" << prop.major << "." << prop.minor
          << "\",\n"
          << "  \"sm_count\": " << prop.multiProcessorCount << ",\n"
          << "  \"total_global_mem\": " << prop.totalGlobalMem << ",\n"
          << "  \"memory_clock_khz\": " << mem_clk << ",\n"
          << "  \"driver_version\": " << driver_v << ",\n"
          << "  \"runtime_version\": " << runtime_v << ",\n"
          << "  \"nvcc\": \"" << __CUDACC_VER_MAJOR__ << "."
          << __CUDACC_VER_MINOR__ << "." << __CUDACC_VER_BUILD__ << "\",\n"
          << "  \"build_type\": \"" << BEIG_BUILD_TYPE << "\",\n"
          << "  \"n_reps\": " << reps << ",\n"
          << "  \"ensemble\": \"GUE, seeded; row-major slabs, vendor reads "
             "as conj(A) (identical eigenvalues)\",\n"
          << "  \"cases\": [\n";
        for (size_t i = 0; i < results.size(); ++i) {
            const CaseResult& c = results[i];
            f << "    {\"n\": " << c.n << ", \"batch\": " << c.batch
              << ", \"precondition\": \"gpu-vs-cpu eval < 1e-12*scale, both "
                 "gpu paths\",\n"
              << "     \"wall_speedup_vs_vendor\": "
              << c.vendor.wall_median / c.beig.wall_median
              << ", \"gpu_speedup_vs_vendor\": "
              << c.vendor.gpu_median / c.beig.gpu_median
              << ", \"subspace_wall_speedup_vs_vendor\": "
              << c.vendor.wall_median / c.beig2.wall_median
              << ", \"subspace_gpu_speedup_vs_vendor\": "
              << c.vendor.gpu_median / c.beig2.gpu_median
              << ", \"subspace_vs_pervector_wall\": "
              << c.beig.wall_median / c.beig2.wall_median
              << ", \"wall_speedup_vs_loop\": "
              << c.loop.wall_median / c.beig.wall_median << ",\n"
              << "     \"beig\": ";
            json_times(f, c.beig, 5);
            f << ",\n     \"beig_subspace\": ";
            json_times(f, c.beig2, 5);
            f << ",\n     \"vendor_xsyevbatched\": ";
            json_times(f, c.vendor, 5);
            f << ",\n     \"naive_xsyevd_loop\": ";
            json_times(f, c.loop, 5);
            f << "}" << (i + 1 < results.size() ? "," : "") << "\n";
        }
        f << "  ]\n}\n";
        if (!f) throw std::runtime_error("write failed: " + out_path);
        std::printf("\nJSON -> %s\n", out_path.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bench_eig failed: %s\n", e.what());
        return 1;
    }
    return 0;
}
