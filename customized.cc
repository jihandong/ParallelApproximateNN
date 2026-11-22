#include "internal.h"

#include <memory>
#include <utility>

#if DEBUG_MODE
#include <iostream>
#endif

IVFImpl make_customized_impl(const std::vector<float>& database,
    const std::vector<float>& queries, std::vector<std::int64_t>& neighbors,
    std::size_t dimension, std::size_t nlist, std::size_t nprobe,
    std::size_t k) {
    const auto trained_centroids = std::make_shared<std::vector<float>>();
    const auto ivf = std::make_shared<std::vector<std::vector<std::size_t>>>();
    return {
        .train = [&database, dimension, nlist, trained_centroids] {
            const auto sample = customized::sample_rows(database, dimension, nlist);
            auto centroids = customized::init_centroids(sample, dimension, nlist);
            customized::lloyd(sample, dimension, nlist, centroids);
#if DEBUG_MODE
            std::clog << "Sampled " << sample.size() / dimension
                << " rows for K-means training.\n"
                << "Initialized " << nlist
                << " centroids.\n";
#endif
            *trained_centroids = std::move(centroids);
        },
        .add = [&database, dimension, nlist, trained_centroids, ivf] {
            *ivf = customized::build_ivf(database,
                *trained_centroids, dimension, nlist);
        },
        .search = [&database, &queries, &neighbors, dimension, nlist, nprobe, k,
                       trained_centroids, ivf] {
            customized::search_ivf(database, queries, *trained_centroids, *ivf,
                dimension, nlist, nprobe, k, neighbors);
        }
    };
}
