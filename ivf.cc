#include "internal.h"

#include <parlay/parallel.h>

#include <iostream>
#include <limits>

#if DEBUG_MODE
#include <chrono>
#include <cstdio>
#endif

namespace customized {

std::vector<std::vector<std::size_t>> build_ivf(
    const std::vector<float>& database, const std::vector<float>& centroids,
    std::size_t dimension, std::size_t nlist) {
#if DEBUG_MODE
    using Clock = std::chrono::steady_clock;
    const auto assign_start = Clock::now();
#endif

    const auto nb_samples = database.size() / dimension;
    std::vector<std::size_t> buckets(nb_samples); // NOTE: can be really large
    parlay::parallel_for(0, nb_samples, [&](std::size_t i) {
        const auto* row = database.data() + i * dimension;
        std::size_t nearest = 0;
        double best_dist = std::numeric_limits<double>::infinity();
        for (std::size_t j = 0; j < nlist; ++j) {
            const auto* centroid = centroids.data() + j * dimension;
            const auto distance = squared_l2(row, centroid, dimension);
            if (distance < best_dist) {
                nearest = j;
                best_dist = distance;
            }
        }
        buckets[i] = nearest;
    });

#if DEBUG_MODE
    const auto group_start = Clock::now();
#endif

    // XXX: bucket filling can be parallel, benefits uncertain
    std::vector<std::vector<std::size_t>> inverted_lists(nlist);
    for (std::size_t i = 0; i < nb_samples; ++i)
        inverted_lists[buckets[i]].push_back(i);

#if DEBUG_MODE
    const auto group_end = Clock::now();
    using Millis = std::chrono::duration<double, std::milli>;
    std::printf("IVF Add profiling (ms): assign=%.6f, group=%.6f\n",
        Millis(group_start - assign_start).count(),
        Millis(group_end - group_start).count());
#endif
    return inverted_lists;
}

void search_ivf() {
    std::cout << "IVF search is not implemented.\n";
}

}  // namespace customized
