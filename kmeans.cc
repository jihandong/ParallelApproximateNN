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
    const auto nb_samples = nlist > database_rows / rows_per_center
        ? database_rows
        : nlist * rows_per_center;

    std::vector<std::size_t> ids(database_rows);
    std::iota(ids.begin(), ids.end(), std::size_t{0});
    std::mt19937 rng(42);

    // In-place Fisher-Yates shuffle.
    for (std::size_t i = 0; i < nb_samples; ++i) {
        std::uniform_int_distribution<std::size_t> pick(i, database_rows - 1);
        std::swap(ids[i], ids[pick(rng)]);
    }

    std::vector<float> sample(nb_samples * dimension);
    // XXX: Row copies can run in parallel, but minor benefits perhaps
    for (std::size_t i = 0; i < nb_samples; ++i) {
        std::copy_n(database.data() + ids[i] * dimension, dimension,
            sample.data() + i * dimension);
    }
    return sample;
}

std::vector<float> init_centroids(const std::vector<float>& sample,
    std::size_t dimension, std::size_t nlist) {
    constexpr std::size_t block_size = 256;
    const auto nb_samples = sample.size() / dimension;
    const auto nb_blocks = (nb_samples + block_size - 1) / block_size;
    std::vector<float> centroids(nlist * dimension);
    std::vector<double> weights(
        nb_samples, std::numeric_limits<double>::infinity()); // distances
    std::vector<double> block_sums(nb_blocks);
    std::vector<bool> selected(nb_samples);

#if DEBUG_MODE
    using Clock = std::chrono::steady_clock;
    Clock::duration weights_time{};
    Clock::duration prefix_time{};
    Clock::duration pick_time{};
#endif

    std::mt19937 rng(43);
    auto row =
        std::uniform_int_distribution<std::size_t>(0, nb_samples - 1)(rng);
    for (std::size_t t = 0; t < nlist; ++t) {
        auto* centroid = centroids.data() + t * dimension;
        std::copy_n(sample.data() + row * dimension, dimension, centroid);
        selected[row] = true;
        if (t + 1 == nlist) break;

        // Refresh the weights and total them within each block.
#if DEBUG_MODE
        const auto weights_start = Clock::now();
#endif
        parlay::blocked_for(0, nb_samples, block_size,
            [&](std::size_t block, std::size_t begin, std::size_t end) {
                double sum = 0.0;
                for (std::size_t i = begin; i < end; ++i) {
                    const auto* point = sample.data() + i * dimension;
                    const auto distance = squared_l2(point, centroid, dimension);
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
            const auto remaining = nb_samples - (t + 1);
            auto rank = std::uniform_int_distribution<std::size_t>(
                0, remaining - 1)(rng);
            for (std::size_t i = 0; i < nb_samples; ++i) {
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
            std::upper_bound(block_sums.begin(), block_sums.end(), draw) - block_sums.begin());
        const auto local_draw = draw - (block == 0 ? 0.0 : block_sums[block - 1]);
        const auto begin = block * block_size;
        const auto end = std::min(begin + block_size, nb_samples);
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

void lloyd(const std::vector<float>& sample, std::size_t dimension,
    std::size_t nlist, std::vector<float>& centroids) {
    constexpr std::size_t max_iterations = 20;
    constexpr double min_improv = 0.0001;
    constexpr std::size_t min_block_size = 256;
    constexpr std::size_t max_blocks = 64;
    const auto nb_samples = sample.size() / dimension;
    const auto all_centroids = nlist * dimension;
    const auto block_size = std::max(min_block_size,
        (nb_samples + max_blocks - 1) / max_blocks);
    const auto nb_blocks = (nb_samples + block_size - 1) / block_size;

    std::vector<std::size_t> block_counts(nb_blocks * nlist); // BxC
    std::vector<double> block_sums(nb_blocks * all_centroids); // BxCxD
    std::vector<double> block_losses(nb_blocks); // B
    // NOTE: B for Sample Blocks, C for Centroids, D for embedding dimensions.
    double prev_loss = 0.0;
    int small_improvs = 0; // counters

#if DEBUG_MODE
    using Clock = std::chrono::steady_clock;
    Clock::duration assign_time{}, update_time{};
#endif

    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
#if DEBUG_MODE
        const auto assign_start = Clock::now();
#endif
        // Assign rows against the same centroids and accumulate per-block stats.
        parlay::blocked_for(0, nb_samples, block_size,
            [&](std::size_t block, std::size_t begin, std::size_t end) {
                auto* counts = block_counts.data() + block * nlist;
                auto* sums = block_sums.data() + block * all_centroids;
                std::fill_n(counts, nlist, std::size_t{0});
                std::fill_n(sums, all_centroids, 0.0);
                double loss = 0.0;
                // XXX: parallelize the tripple loop might bring speedup
                for (std::size_t i = begin; i < end; ++i) {
                    const auto* point = sample.data() + i * dimension;
                    std::size_t nearest = 0;
                    double best_dist = std::numeric_limits<double>::infinity();
                    for (std::size_t j = 0; j < nlist; ++j) {
                        const auto* centroid = centroids.data() + j * dimension;
                        const auto distance = squared_l2(point, centroid, dimension);
                        if (distance < best_dist) {
                            nearest = j;
                            best_dist = distance;
                        }
                    }

                    ++counts[nearest];
                    auto* sum = sums + nearest * dimension;
                    for (std::size_t j = 0; j < dimension; ++j)
                        sum[j] += point[j];
                    loss += best_dist;
                }
                block_losses[block] = loss;
            });

#if DEBUG_MODE
        assign_time += Clock::now() - assign_start;
        const auto update_start = Clock::now();
#endif
        // Merge in-block centroid losses into block0.
        // XXX: SIMD or parallelism
        parlay::parallel_for(0, nlist, [&](std::size_t j) {
            const auto offset = j * dimension;
            auto* sum = block_sums.data() + offset;
            auto count = block_counts[j];
            for (std::size_t block = 1; block < nb_blocks; ++block) {
                count += block_counts[block * nlist + j];
                const auto* block_sum =
                    block_sums.data() + block * all_centroids + offset;
                for (std::size_t k = 0; k < dimension; ++k)
                    sum[k] += block_sum[k];
            }
            if (count == 0) return;
            auto* centroid = centroids.data() + offset;
            for (std::size_t k = 0; k < dimension; ++k)
                centroid[k] = static_cast<float>(sum[k] / count);
        });
#if DEBUG_MODE
        update_time += Clock::now() - update_start;
#endif

        const auto loss =
            std::accumulate(block_losses.begin(), block_losses.end(), 0.0);
        if (loss == 0.0) break;
        if (iteration > 0 && (prev_loss - loss) / prev_loss < min_improv) {
            if (++small_improvs == 2) break;
        } else {
            small_improvs = 0;
        }
        prev_loss = loss;
    }

#if DEBUG_MODE
    using Milliseconds = std::chrono::duration<double, std::milli>;
    std::fprintf(stderr, "Lloyd profiling (ms): assign=%.6f, update=%.6f\n",
        Milliseconds(assign_time).count(), Milliseconds(update_time).count());
#endif
}

}  // namespace customized
