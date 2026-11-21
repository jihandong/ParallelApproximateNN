#ifndef PARALLEL_APPROXIMATE_NN_INTERNAL_H_
#define PARALLEL_APPROXIMATE_NN_INTERNAL_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

struct IVFImpl {
    std::function<void()> train;
    std::function<void()> add;
    std::function<void()> search;
};

// The data buffers must outlive the returned callbacks.
IVFImpl make_faiss_impl(const std::vector<float>& database,
    const std::vector<float>& queries, std::vector<std::int64_t>& neighbors,
    std::size_t dimension, std::size_t nlist, std::size_t nprobe,
    std::size_t k);
IVFImpl make_customized_impl(const std::vector<float>& database,
    const std::vector<float>& queries, std::vector<std::int64_t>& neighbors,
    std::size_t dimension, std::size_t nlist, std::size_t nprobe,
    std::size_t k);

namespace customized {

std::vector<float> sample_training_rows(const std::vector<float>& database,
    std::size_t dimension, std::size_t nlist);
void build_ivf_index();
void search_ivf();

}  // namespace customized

#endif  // PARALLEL_APPROXIMATE_NN_INTERNAL_H_
