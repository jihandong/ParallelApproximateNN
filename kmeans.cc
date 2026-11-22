#include "internal.h"

#include <parlay/parallel.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <utility>

#if DEBUG_MODE
#include <chrono>
#include <cstdio>
#endif

namespace customized {

std::vector<float> sample_rows(const std::vector<float>& database,
    std::size_t dimension, std::size_t nlist) {
    constexpr std::size_t rows_per_center = 64;
    const auto database_rows = database.size() / dimension;
    const auto sample_count = nlist > database_rows / rows_per_center
        ? database_rows
        : nlist * rows_per_center;

    std::vector<std::size_t> ids(database_rows);
    std::iota(ids.begin(), ids.end(), std::size_t{0});
    std::mt19937 rng(42);

    // In-place Fisher-Yates shuffle.
    for (std::size_t i = 0; i < sample_count; ++i) {
        std::uniform_int_distribution<std::size_t> pick(i, database_rows - 1);
        std::swap(ids[i], ids[pick(rng)]);
    }

    std::vector<float> sample(sample_count * dimension);
    // XXX: Row copies can run in parallel, but minor benefits perhaps
    for (std::size_t i = 0; i < sample_count; ++i) {
        std::copy_n(database.data() + ids[i] * dimension, dimension,
            sample.data() + i * dimension);
    }
    return sample;
}

std::vector<float> init_centroids(const std::vector<float>& sample,
    std::size_t dimension, std::size_t nlist) {
    constexpr std::size_t block_size = 256;
    const auto sample_count = sample.size() / dimension;
    const auto block_count = (sample_count + block_size - 1) / block_size;
    std::vector<float> centroids(nlist * dimension);
    std::vector<double> weights(
        sample_count, std::numeric_limits<double>::infinity()); // distances
    std::vector<double> block_sums(block_count);
    std::vector<bool> selected(sample_count);

#if DEBUG_MODE
    using Clock = std::chrono::steady_clock;
    Clock::duration weights_time{};
    Clock::duration prefix_time{};
    Clock::duration pick_time{};
#endif

    std::mt19937 rng(43);
    auto row =
        std::uniform_int_distribution<std::size_t>(0, sample_count - 1)(rng);
    for (std::size_t t = 0; t < nlist; ++t) {
        auto* centroid = centroids.data() + t * dimension;
        std::copy_n(sample.data() + row * dimension, dimension, centroid);
        selected[row] = true;
        if (t + 1 == nlist) break;

        // Refresh the weights and total them within each block.
#if DEBUG_MODE
        const auto weights_start = Clock::now();
#endif
        parlay::blocked_for(0, sample_count, block_size,
            [&](std::size_t block, std::size_t begin, std::size_t end) {
                double sum = 0.0;
                for (std::size_t i = begin; i < end; ++i) {
                    const auto* point = sample.data() + i * dimension;
                    // XXX: Benchmark SIMD or batched kernels for squared L2.
                    double distance = 0.0;
                    for (std::size_t j = 0; j < dimension; ++j) {
                        const double diff =
                            static_cast<double>(point[j]) - centroid[j];
                        distance += diff * diff;
                    }
                    weights[i] = std::min(weights[i], distance);
                    sum += weights[i];
                }
                block_sums[block] = sum;
            });

#if DEBUG_MODE
        weights_time += Clock::now() - weights_start;
        const auto prefix_start = Clock::now();
#endif
        // XXX: Evaluate a parallel scan for larger block counts.
        std::partial_sum(
            block_sums.begin(), block_sums.end(), block_sums.begin());
#if DEBUG_MODE
        prefix_time += Clock::now() - prefix_start;
        const auto pick_start = Clock::now();
#endif
        const auto total = block_sums.back();
        // Select uniformly among unused rows if all weights vanish.
        if (total == 0.0) {
            const auto remaining = sample_count - (t + 1);
            auto rank = std::uniform_int_distribution<std::size_t>(
                0, remaining - 1)(rng);
            for (std::size_t i = 0; i < sample_count; ++i) {
                if (selected[i]) continue;
                if (rank == 0) {
                    row = i;
                    break;
                }
                --rank;
            }
#if DEBUG_MODE
            pick_time += Clock::now() - pick_start;
#endif
            continue;
        }

        // Pick next centroid, distances as prob weights
        const auto draw = std::uniform_real_distribution<double>(0.0, total)(rng);
        const auto block = static_cast<std::size_t>(
            std::upper_bound(block_sums.begin(), block_sums.end(), draw) -
            block_sums.begin());
        const auto local_draw = draw - (block == 0 ? 0.0 : block_sums[block - 1]);
        const auto begin = block * block_size;
        const auto end = std::min(begin + block_size, sample_count);
        double sum = 0.0;
        for (std::size_t i = begin; i < end; ++i) {
            if (weights[i] == 0.0) continue;
            row = i;  // Fall back to the last positive row if the draw rounds up.
            sum += weights[i];
            if (local_draw < sum) break;
        }
#if DEBUG_MODE
        pick_time += Clock::now() - pick_start;
#endif
    }

#if DEBUG_MODE
    using Milliseconds = std::chrono::duration<double, std::milli>;
    std::fprintf(stderr,
        "K-means++ profiling (ms): weights=%.6f, prefix_sum=%.6f, pick=%.6f\n",
        Milliseconds(weights_time).count(), Milliseconds(prefix_time).count(),
        Milliseconds(pick_time).count());
#endif
    return centroids;
}

}  // namespace customized
