#include "internal.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <utility>

namespace customized {

std::vector<float> sample_training_rows(const std::vector<float>& database,
    std::size_t dimension, std::size_t nlist) {
    constexpr std::size_t rows_per_center = 64;
    const auto database_rows = database.size() / dimension;
    const auto sample_rows = nlist > database_rows / rows_per_center
        ? database_rows
        : nlist * rows_per_center;

    std::vector<std::size_t> ids(database_rows);
    std::iota(ids.begin(), ids.end(), std::size_t{0});
    std::mt19937 rng(42);

    // In-place Fisher-Yates shuffle.
    for (std::size_t i = 0; i < sample_rows; ++i) {
        std::uniform_int_distribution<std::size_t> pick(i, database_rows - 1);
        std::swap(ids[i], ids[pick(rng)]);
    }

    std::vector<float> sample(sample_rows * dimension);
    // XXX: Row copies can run in parallel, but minor benefits perhaps
    for (std::size_t i = 0; i < sample_rows; ++i) {
        std::copy_n(database.data() + ids[i] * dimension, dimension,
            sample.data() + i * dimension);
    }
    return sample;
}

}  // namespace customized
