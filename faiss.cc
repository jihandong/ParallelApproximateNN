#include <cblas.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <omp.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

std::function<void(const float*, std::int64_t*)> make_faiss_search(
    const std::vector<float>& database, std::size_t dimension,
    std::size_t nlist, std::size_t nprobe, std::size_t query_rows,
    std::size_t k) {
  // Limit both Faiss's OpenMP work and its BLAS calls to one thread.
  omp_set_num_threads(1);
  openblas_set_num_threads(1);

  const auto database_rows = database.size() / dimension;

  auto quantizer = std::make_unique<faiss::IndexFlatL2>(dimension);
  auto impl = std::make_shared<faiss::IndexIVFFlat>(quantizer.get(), dimension,
                                                    nlist, faiss::METRIC_L2);
  impl->own_fields = true;  // The index takes ownership of the quantizer.
  quantizer.release();
  impl->nprobe = nprobe;
  impl->train(database_rows, database.data());
  impl->add(database_rows, database.data());

  return [impl, query_rows, k, distances = std::vector<float>(query_rows * k)](
             const float* queries, std::int64_t* neighbors) mutable {
    impl->search(query_rows, queries, k, distances.data(), neighbors);
  };
}
