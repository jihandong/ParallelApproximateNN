#include "internal.h"

#if DEBUG_MODE
#include <iostream>
#endif

IVFImpl make_customized_impl(const std::vector<float>& database,
    const std::vector<float>& queries, std::vector<std::int64_t>& neighbors,
    std::size_t dimension, std::size_t nlist, std::size_t nprobe,
    std::size_t k) {
    (void)queries;
    (void)neighbors;
    (void)nprobe;
    (void)k;

    return {
        .train = [&database, dimension, nlist] {
            const auto sample =
                customized::sample_training_rows(database, dimension, nlist);
#if DEBUG_MODE
            std::clog << "Sampled " << sample.size() / dimension
                << " rows for K-means training.\n";
#endif
            // TODO: Initialize and refine the centers from the sample for add.
        },
        // TODO: Build the inverted lists from the database and trained centers.
        .add = customized::build_ivf_index,
        // TODO: Search the built index and write IDs into the result buffer.
        .search = customized::search_ivf
    };
}
