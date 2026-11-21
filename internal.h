#ifndef PARALLEL_APPROXIMATE_NN_INTERNAL_H_
#define PARALLEL_APPROXIMATE_NN_INTERNAL_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

std::function<void(const float*, std::int64_t*)> make_faiss_search(
    const std::vector<float>& database, std::size_t dimension, std::size_t nlist,
    std::size_t nprobe, std::size_t query_rows, std::size_t k);
int run_customized();

namespace customized {

void train_kmeans();
void build_ivf_index();
void search_ivf();

}  // namespace customized

#endif  // PARALLEL_APPROXIMATE_NN_INTERNAL_H_
