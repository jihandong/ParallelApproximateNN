cmake_minimum_required(VERSION 3.20)
project(ParallelApproximateNN LANGUAGES CXX)

find_package(OpenMP REQUIRED COMPONENTS CXX)

# The baseline calls OpenBLAS directly to set its thread count.
set(BLA_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)
find_path(OPENBLAS_INCLUDE_DIR NAMES openblas_config.h
    PATH_SUFFIXES openblas openblas-pthread openblas-openmp REQUIRED)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    include(FetchContent)

    find_package(faiss CONFIG QUIET)
    if(NOT TARGET faiss)
        set(FAISS_ENABLE_GPU OFF)
        set(FAISS_ENABLE_PYTHON OFF)
        set(FAISS_ENABLE_MKL OFF)
        set(FAISS_ENABLE_EXTRAS OFF)
        set(BUILD_TESTING OFF)
        FetchContent_Declare(faiss
            GIT_REPOSITORY https://github.com/facebookresearch/faiss.git
            GIT_TAG v1.14.0
            GIT_SHALLOW TRUE)
        FetchContent_MakeAvailable(faiss)
    endif()

    find_package(Parlay CONFIG QUIET)
    if(NOT TARGET Parlay::parlay)
        FetchContent_Declare(parlaylib
            GIT_REPOSITORY https://github.com/cmuparlay/parlaylib.git
            GIT_TAG 51017699dcc421f80479cdb238d3092233ad0d26)
        FetchContent_MakeAvailable(parlaylib)
        add_library(Parlay::parlay ALIAS parlay)
    endif()
else()
    find_package(faiss CONFIG REQUIRED)
    find_package(Parlay CONFIG REQUIRED)
endif()

add_executable(ann main.cc faiss.cc customized.cc kmeans.cc ivf.cc)
target_compile_features(ann PRIVATE cxx_std_20)
target_compile_definitions(ann PRIVATE DEBUG_MODE=$<CONFIG:Debug>)
target_include_directories(ann PRIVATE "${OPENBLAS_INCLUDE_DIR}")
target_link_libraries(ann PRIVATE faiss OpenMP::OpenMP_CXX BLAS::BLAS
    Parlay::parlay)

# MinGW GCC emits duplicate Parlay TLS initializers across translation units.
if(MINGW AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set_target_properties(ann PROPERTIES UNITY_BUILD ON UNITY_BUILD_MODE GROUP)
    set_source_files_properties(kmeans.cc ivf.cc PROPERTIES UNITY_GROUP parlay)
endif()

# Stage the local UCRT64 DLLs next to ann.exe.
if(WIN32 AND ANN_RUNTIME_DIR)
    file(COPY
        "${ANN_RUNTIME_DIR}/libopenblas.dll"
        "${ANN_RUNTIME_DIR}/libgomp-1.dll"
        "${ANN_RUNTIME_DIR}/libgfortran-5.dll"
        "${ANN_RUNTIME_DIR}/libquadmath-0.dll"
        DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
endif()
