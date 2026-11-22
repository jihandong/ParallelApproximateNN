cmake_minimum_required(VERSION 3.20)
project(ParallelApproximateNN LANGUAGES CXX)

find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(faiss CONFIG REQUIRED)
find_package(Parlay CONFIG REQUIRED)

# The baseline calls OpenBLAS directly to set its thread count.
set(BLA_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)
find_path(OPENBLAS_INCLUDE_DIR NAMES openblas_config.h
    PATH_SUFFIXES openblas openblas-pthread openblas-openmp REQUIRED)

add_executable(ann main.cc faiss.cc customized.cc kmeans.cc ivf.cc)
target_compile_features(ann PRIVATE cxx_std_20)
target_compile_definitions(ann PRIVATE DEBUG_MODE=$<CONFIG:Debug>)
target_include_directories(ann PRIVATE "${OPENBLAS_INCLUDE_DIR}")
target_link_libraries(ann PRIVATE faiss OpenMP::OpenMP_CXX BLAS::BLAS
    Parlay::parlay)

# Stage the local UCRT64 DLLs next to ann.exe.
if(WIN32 AND ANN_RUNTIME_DIR)
    file(COPY
        "${ANN_RUNTIME_DIR}/libopenblas.dll"
        "${ANN_RUNTIME_DIR}/libgomp-1.dll"
        "${ANN_RUNTIME_DIR}/libgfortran-5.dll"
        "${ANN_RUNTIME_DIR}/libquadmath-0.dll"
        DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
endif()
