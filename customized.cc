#include "internal.h"

#include <iostream>

IVFImpl make_customized_impl(const std::vector<float>& database,
    std::size_t dimension, std::size_t nlist) {
    return {
        .train = [&database, dimension, nlist] {
            const auto sample =
                customized::sample_training_rows(database, dimension, nlist);
            // TODO: Initialize and refine the centers from the sample for add.
            std::cout << "Sampled " << sample.size() / dimension
                << " rows for K-means training.\n";
        },
        // TODO: Build the inverted lists from the database and trained centers.
        .add = customized::build_ivf_index,
        // TODO: Search the built index and write IDs into the result buffer.
        .search = customized::search_ivf
    };
}
