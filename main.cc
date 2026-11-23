#include "internal.h"

#include <faiss/utils/utils.h>

#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

constexpr std::size_t top_k = 10;

// Command-line params and argument parsing.
struct Params {
    std::string_view implementation = "faiss";
    std::size_t nlist = 256;
    std::size_t nprobe = 8;
    int threads = 1;
};

std::size_t positive_integer(std::string_view value, std::string_view option) {
    std::size_t number;
    const auto* end = value.data() + value.size();
    const auto result = std::from_chars(value.data(), end, number);
    if (result.ec != std::errc{} || result.ptr != end || number == 0) {
        throw std::runtime_error(
            "Expected a positive integer for " + std::string(option));
    }
    return number;
}

Params parse_params(int argc, char** argv) {
    Params params;
    for (int i = 2; i < argc; i += 2) {
        const std::string_view option = argv[i];
        if (i + 1 == argc) {
            throw std::runtime_error(
                "Missing value for " + std::string(option));
        }
        const std::string_view value = argv[i + 1];
        if (option == "--impl") {
            params.implementation = value;
        } else if (option == "--nlist") {
            params.nlist = positive_integer(value, option);
        } else if (option == "--nprobe") {
            params.nprobe = positive_integer(value, option);
        } else if (option == "--threads") {
            const auto threads = positive_integer(value, option);
            if (!std::in_range<int>(threads)) {
                throw std::runtime_error("Thread count is too large for --threads");
            }
            params.threads = static_cast<int>(threads);
        } else {
            throw std::runtime_error("Unknown option: " + std::string(option));
        }
    }
    if (params.nprobe > params.nlist) {
        throw std::runtime_error("--nprobe must be at most --nlist");
    }
    return params;
}

// Input helpers for the prepared dataset.
std::size_t read_metadata(const std::string& metadata, const std::string& key) {
    // The data preparation script writes a flat JSON object of +integers.
    const std::regex field("\"" + key + "\"\\s*:\\s*([1-9][0-9]*)\\s*[,}]");
    std::smatch match;
    if (!std::regex_search(metadata, match, field)) {
        throw std::runtime_error("Missing or invalid metadata field: " + key);
    }
    return std::stoull(match[1].str());
}

template <class T>  // matrix element value type.
std::vector<T> read_matrix(
    const fs::path& path, std::size_t rows, std::size_t columns) {
    // Prepared matrices are headerless and use 32-bit little-endian elements.
    static_assert(sizeof(T) == 4 && std::endian::native == std::endian::little);
    const auto bytes = fs::file_size(path);
    const auto count = bytes / sizeof(T);
    if (bytes % sizeof(T) != 0 || count % columns != 0 ||
        count / columns != rows) {
        throw std::runtime_error(
            "Unexpected matrix file size: " + path.string());
    }
    std::vector<T> values(count);
    std::ifstream input(path, std::ios::binary);
    if (!input.read(reinterpret_cast<char*>(values.data()), bytes)) {
        throw std::runtime_error("Could not read " + path.string());
    }
    return values;
}

// Validate result IDs and compare them with the reference answers.
double check_results(const fs::path& reference_path,
    const std::vector<std::int64_t>& neighbors, std::size_t db_rows,
    std::size_t query_rows, std::size_t reference_columns) {
    if (neighbors.size() % top_k != 0 ||
        neighbors.size() / top_k != query_rows) {
        throw std::runtime_error("Expected exactly 10 neighbor IDs per query");
    }
    const auto references = read_matrix<std::int32_t>(
        reference_path, query_rows, reference_columns);
    std::vector<std::int64_t> expected(top_k);
    std::size_t matches = 0;
    for (std::size_t row = 0; row < query_rows; ++row) {
        const auto* predicted = neighbors.data() + row * top_k;
        for (const auto* id = predicted; id != predicted + top_k; ++id) {
            if (*id < 0 || std::cmp_greater_equal(*id, db_rows)) {
                throw std::runtime_error(
                    "Out-of-range neighbor ID at query " + std::to_string(row));
            }
            if (std::find(predicted, id, *id) != id) {
                throw std::runtime_error(
                    "Duplicate neighbor ID at query " + std::to_string(row));
            }
        }
        // Widen the first 10 reference IDs to Faiss's 64-bit ID type.
        std::copy_n(references.data() + row * reference_columns, top_k,
            expected.data());
        matches += faiss::ranklist_intersection_size(
            top_k, expected.data(), top_k, predicted);
    }
    return static_cast<double>(matches) / neighbors.size();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ann <dataset directory> [--impl faiss|customized]"
            " [--nlist 256] [--nprobe 8] [--threads 1]\n";
        return 2;
    }

    try {
        // Parse the params and select the search implementation.
        const auto params = parse_params(argc, argv);
        const bool use_customized = params.implementation == "customized";
        if (!use_customized && params.implementation != "faiss") {
            throw std::runtime_error("Unknown implementation: " +
                std::string(params.implementation));
        }

        // Read the dataset and check that they support the requested search.
        const fs::path directory = argv[1];
        const auto metadata_path = directory / "metadata.json";
        std::ifstream metadata_file(metadata_path);
        if (!metadata_file) {
            throw std::runtime_error(
                "Could not open " + metadata_path.string());
        }
        const std::string metadata{
            std::istreambuf_iterator<char>(metadata_file),
            std::istreambuf_iterator<char>()};
        const auto dimension = read_metadata(metadata, "dimension");
        const auto db_rows = read_metadata(metadata, "database_rows");
        const auto query_rows = read_metadata(metadata, "query_rows");
        if (params.nlist > db_rows) {
            throw std::runtime_error("--nlist must be at most the db size");
        }
        const auto reference_columns =
            read_metadata(metadata, "neighbors_per_query");
        if (db_rows < top_k || reference_columns < top_k) {
            throw std::runtime_error(
                "Too few db rows or reference IDs for top-10");
        }

        // Load the database and query vectors.
        const auto db =
            read_matrix<float>(directory / "database.f32", db_rows, dimension);
        const auto queries = read_matrix<float>(
            directory / "queries.f32", query_rows, dimension);

        // Bind the in-memory data and allocate buffers before timing the phases.
        std::vector<std::int64_t> neighbors(query_rows * top_k);
        const auto make_impl =
            use_customized ? make_customized_impl : make_faiss_impl;
        const auto impl = make_impl(db, queries, neighbors, dimension,
            params.nlist, params.nprobe, top_k, params.threads);
        using Clock = std::chrono::steady_clock;
        using Seconds = std::chrono::duration<double>;
        const auto train_start = Clock::now();
        impl.train();
        const auto add_start = Clock::now();
        impl.add();
        const auto search_start = Clock::now();
        impl.search();
        const auto search_end = Clock::now();

        const auto train_elapsed = Seconds(add_start - train_start);
        const auto add_elapsed = Seconds(search_start - add_start);
        const auto search_elapsed = Seconds(search_end - search_start);

        // Check the returned IDs outside the timed run.
        const auto recall = check_results(directory / "neighbors.i32",
            neighbors, db_rows, query_rows, reference_columns);

        std::cout << "Train: " << train_elapsed.count() << " seconds\n"
            << "Add: " << add_elapsed.count() << " seconds\n"
            << query_rows << " queries in " << search_elapsed.count()
            << " seconds\n"
            << "Recall@" << top_k << ": " << std::fixed
            << std::setprecision(2) << recall * 100 << "%\n";

    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
