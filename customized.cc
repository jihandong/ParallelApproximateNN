#include "internal.h"

int run_customized() {
  customized::train_kmeans();
  customized::build_ivf_index();
  customized::search_ivf();
  return 0;
}
