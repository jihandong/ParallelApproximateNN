#include "internal.h"

#include <cblas.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <omp.h>

#include <cstdint>
#include <memory>
#include <vector>

IVFImpl make_faiss_impl(const std::vector<float>& database,
    const std::vector<float>& queries, std::vector<std::int64_t>& neighbors,
    std::size_t dimension, std::size_t nlist, std::size_t nprobe,
    std::size_t k) {
    // Limit both Faiss's OpenMP work and its BLAS calls to one thread.
    omp_set_num_threads(1);
    openblas_set_num_threads(1);

    const auto database_rows = database.size() / dimension;
    const auto query_rows = queries.size() / dimension;

    auto quantizer = std::make_unique<faiss::IndexFlatL2>(dimension);
    auto index = std::make_shared<faiss::IndexIVFFlat>(
        quantizer.get(), dimension, nlist, faiss::METRIC_L2);
    index->own_fields = true;  // The index takes ownership of the quantizer.
    quantizer.release();
    index->nprobe = nprobe;

    return {
        .train = [index, &database, database_rows] {
            index->train(database_rows, database.data());
        },
        .add = [index, &database, database_rows] {
            index->add(database_rows, database.data());
        },
        .search = [index, &queries, &neighbors, query_rows, k,
            distances = std::vector<float>(query_rows * k)]() mutable {
            index->search(query_rows, queries.data(), k, distances.data(),
                neighbors.data());
        },
    };
}
