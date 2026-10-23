// Physics checks for the zigzag-ribbon test case, plus the batch API.
//   - H(k) is bit-exactly Hermitian, zero diagonal, 4W-2 nonzeros.
//   - At k = pi the Bloch bond t(1 + e^{-ik}) vanishes: the spectrum is
//     exactly {0 (twice, edge modes), +/-|t| (W-1 each)}.
//   - Chiral (sublattice) symmetry: eigenvalues pair as +/-E.
//   - solve_batch output is bitwise reproducible from the base seed:
//     item b equals a single solve seeded with mix64(seed + b).

#include "beig/hermitian_eig.hpp"
#include "ensembles.hpp"

#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>
#include <vector>

namespace {

using beig::cdouble;
int failures = 0;

void check(bool ok, const char* name, const std::string& detail) {
    std::printf("[%s] %s: %s\n", ok ? "pass" : "FAIL", name, detail.c_str());
    if (!ok) ++failures;
}

} // namespace

int main() {
    char buf[256];
    const double pi = std::numbers::pi;

    // ---------- structure ----------
    const int W = 32;
    const int n = 2 * W;
    const auto H = beigtest::zigzag_ribbon(W, 0.7);

    bool herm = true, zero_diag = true;
    int nnz = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            const cdouble h = H[(size_t)i * n + j];
            if (h != 0.0) ++nnz;
            if (h != conj(H[(size_t)j * n + i])) herm = false;
            if (i == j && h != 0.0) zero_diag = false;
        }
    std::snprintf(buf, sizeof buf, "nnz = %d (expect %d)", nnz, 4 * W - 2);
    check(nnz == 4 * W - 2, "ribbon bond count", buf);
    check(herm, "ribbon H(k) bit-exact Hermitian", "H[i][j] == conj(H[j][i])");
    check(zero_diag, "ribbon zero on-site", "diag = 0");

    // ---------- k = pi: exact edge modes + dimer spectrum ----------
    const auto Hpi = beigtest::zigzag_ribbon(W, pi);
    const beig::EigResult rp = beig::hermitian_eig(Hpi.data(), n, 0xA11CE);
    int n_zero = 0, n_plus = 0, n_minus = 0;
    double w_edge = 0.0, w_dimer = 0.0;
    for (double lam : rp.w) {
        if (std::abs(lam) < 1e-9) {
            ++n_zero;
            w_edge = std::max(w_edge, std::abs(lam));
        } else if (std::abs(lam - 1.0) < 1e-9) {
            ++n_plus;
            w_dimer = std::max(w_dimer, std::abs(lam - 1.0));
        } else if (std::abs(lam + 1.0) < 1e-9) {
            ++n_minus;
            w_dimer = std::max(w_dimer, std::abs(lam + 1.0));
        }
    }
    std::snprintf(buf, sizeof buf,
                  "%d zero modes (expect 2), max|edge E| = %.3e", n_zero,
                  w_edge);
    check(n_zero == 2, "k=pi zigzag edge zero modes", buf);
    std::snprintf(buf, sizeof buf,
                  "%d at +|t|, %d at -|t| (expect %d each), max dev %.3e",
                  n_plus, n_minus, W - 1, w_dimer);
    check(n_plus == W - 1 && n_minus == W - 1, "k=pi dimer levels at +/-|t|",
          buf);

    // ---------- chiral symmetry at a generic k ----------
    const beig::EigResult rc = beig::hermitian_eig(H.data(), n, 0xC11A1);
    double sym = 0.0;
    for (int i = 0; i < n; ++i)
        sym = std::max(sym, std::abs(rc.w[i] + rc.w[n - 1 - i]));
    std::snprintf(buf, sizeof buf, "max|E_i + E_{n-1-i}| = %.3e", sym);
    check(sym < 1e-12, "chiral +/-E pairing at k=0.7", buf);

    // ---------- batch API: 9 k-points, W = 16, bitwise seed reproducibility
    const int Wb = 16, nb = 2 * Wb, npts = 9;
    const uint64_t seed = 0xBA7C4;
    std::vector<std::vector<cdouble>> mats(npts);
    std::vector<beig::BatchItem> items(npts);
    std::vector<std::vector<double>> ws(npts);
    std::vector<std::vector<cdouble>> vs(npts);
    for (int j = 0; j < npts; ++j) {
        const double k = -pi + 2.0 * pi * j / (npts - 1);
        mats[j] = beigtest::zigzag_ribbon(Wb, k);
        ws[j].resize(nb);
        vs[j].resize((size_t)nb * nb);
        items[j] = beig::BatchItem{nb, mats[j].data(), ws[j].data(),
                                   vs[j].data()};
    }
    beig::solve_batch(items.data(), npts, seed);

    bool same = true;
    for (int j = 0; j < npts; ++j) {
        const beig::EigResult single = beig::hermitian_eig(
            mats[j].data(), nb, beig::mix64(seed + (uint64_t)j));
        if (single.w != ws[j] || single.v != vs[j]) same = false;
    }
    check(same, "batch == per-matrix solves at mix64(seed + b)",
          "bitwise over 9 k-points, n = 32");

    std::printf("%s (%d failure%s)\n",
                failures == 0 ? "ALL CHECKS PASSED" : "FAILURES", failures,
                failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
