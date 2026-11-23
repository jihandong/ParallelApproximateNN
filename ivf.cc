#include "internal.h"

#include <parlay/parallel.h>

#include <algorithm>
#include <limits>
#include <queue>
#include <utility>

#if DEBUG_MODE
#include <chrono>
#include <cstdio>
#include <numeric>
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

void search_ivf(const std::vector<float>& database,
    const std::vector<float>& queries, const std::vector<float>& centroids,
    const std::vector<std::vector<std::size_t>>& ivf, std::size_t dimension,
    std::size_t nlist, std::size_t nprobe, std::size_t k,
    std::vector<std::int64_t>& neighbors) {
    using Candidate = std::pair<double, std::size_t>;
    const auto nb_queries = queries.size() / dimension;
#if DEBUG_MODE
    using Clock = std::chrono::steady_clock;
    std::vector<Clock::duration> probe_times(nb_queries), scan_times(nb_queries);
    // XXX: Shall we use atomic add instead of final sum?
#endif

    parlay::parallel_for(0, nb_queries, [&](std::size_t i) {
#if DEBUG_MODE
        const auto probe_start = Clock::now();
#endif
        const auto* query = queries.data() + i * dimension;
        std::vector<Candidate> probes(nlist);
        for (std::size_t j = 0; j < nlist; ++j) {
            const auto* centroid = centroids.data() + j * dimension;
            probes[j] = {squared_l2(query, centroid, dimension), j};
        }
        // XXX: might be optimizable
        std::partial_sort(probes.begin(), probes.begin() + nprobe, probes.end());

#if DEBUG_MODE
        const auto scan_start = Clock::now();
        probe_times[i] = scan_start - probe_start;
#endif
        // XXX: max-heap impl, not the best algorithm for top-k
        std::priority_queue<Candidate> best;
        for (std::size_t j = 0; j < nprobe; ++j) {
            for (const auto id : ivf[probes[j].second]) {
                const auto* row = database.data() + id * dimension;
                const Candidate candidate{squared_l2(query, row, dimension), id};
                if (best.size() < k) {
                    best.push(candidate);
                } else if (candidate < best.top()) {
                    best.pop();
                    best.push(candidate);
                }
            }
        }

        // Write nearest IDs first and mark missing results.
        auto* result = neighbors.data() + i * k;
        std::fill(result + best.size(), result + k, -1);
        while (!best.empty()) {
            result[best.size() - 1] = static_cast<std::int64_t>(best.top().second);
            best.pop();
        }
#if DEBUG_MODE
        scan_times[i] = Clock::now() - scan_start;
#endif
    });

#if DEBUG_MODE
    using Millis = std::chrono::duration<double, std::milli>;
    const auto probe_time = std::accumulate(
        probe_times.begin(), probe_times.end(), Clock::duration{});
    const auto scan_time = std::accumulate(
        scan_times.begin(), scan_times.end(), Clock::duration{});
    std::printf("IVF Search profiling: probe=%.6f, scan=%.6f\n",
        Millis(probe_time).count(), Millis(scan_time).count());
#endif
}

}  // namespace customized
