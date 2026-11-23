#ifndef PARALLEL_APPROXIMATE_NN_INTERNAL_H_
#define PARALLEL_APPROXIMATE_NN_INTERNAL_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

struct IVFImpl {
    std::function<void()> train;
    std::function<void()> add;
    std::function<void()> search;
};

// The data buffers must outlive the returned callbacks.
IVFImpl make_faiss_impl(const std::vector<float>& database,
    const std::vector<float>& queries, std::vector<std::int64_t>& neighbors,
    std::size_t dimension, std::size_t nlist, std::size_t nprobe,
    std::size_t k, int threads);
IVFImpl make_customized_impl(const std::vector<float>& database,
    const std::vector<float>& queries, std::vector<std::int64_t>& neighbors,
    std::size_t dimension, std::size_t nlist, std::size_t nprobe,
    std::size_t k, int threads);

namespace customized {

// XXX: Benchmark SIMD or a library implementation for squared L2.
static inline double squared_l2(const float* lhs, const float* rhs,
    std::size_t dimension) {
    double distance = 0.0;
    for (std::size_t i = 0; i < dimension; ++i) {
        const double diff = static_cast<double>(lhs[i]) - rhs[i];
        distance += diff * diff;
    }
    return distance;
}

std::vector<float> sample_rows(const std::vector<float>& database,
    std::size_t dimension, std::size_t nlist);
std::vector<float> init_centroids(const std::vector<float>& sample,
    std::size_t dimension, std::size_t nlist);
void lloyd(const std::vector<float>& sample, std::size_t dimension,
    std::size_t nlist, std::vector<float>& centroids);

std::vector<std::vector<std::size_t>> build_ivf(
    const std::vector<float>& database, const std::vector<float>& centroids,
    std::size_t dimension, std::size_t nlist);
void search_ivf(const std::vector<float>& database,
    const std::vector<float>& queries, const std::vector<float>& centroids,
    const std::vector<std::vector<std::size_t>>& ivf, std::size_t dimension,
    std::size_t nlist, std::size_t nprobe, std::size_t k,
    std::vector<std::int64_t>& neighbors);

}  // namespace customized

#endif  // PARALLEL_APPROXIMATE_NN_INTERNAL_H_
