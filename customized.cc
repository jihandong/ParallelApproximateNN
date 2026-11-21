#include "internal.h"

IVFImpl make_customized_impl() {
    return {
        .train = customized::train_kmeans,
        .add = customized::build_ivf_index,
        .search = customized::search_ivf
    };
}
