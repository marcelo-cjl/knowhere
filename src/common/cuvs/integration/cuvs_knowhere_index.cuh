/**
 * SPDX-FileCopyrightText: Copyright (c) 2023,NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <cmath>
#include <cstdint>
#include <cuvs/core/bitset.hpp>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <fstream>
#include <iostream>
#include <istream>
#include <limits>
#include <ostream>
#include <raft/core/copy.cuh>
#include <raft/core/device_resources_manager.hpp>
#include <raft/core/device_setter.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/serialize.hpp>
#include <raft/linalg/normalize.cuh>
#include <tuple>
#include <type_traits>

#include "common/cuvs/integration/cuvs_knowhere_index.hpp"
#include "common/cuvs/proto/cuvs_index.hpp"
#include "common/cuvs/proto/cuvs_index_kind.hpp"
#include "knowhere/comp/index_param.h"

namespace cuvs_knowhere {
namespace detail {

// This helper struct maps the generic type of cuVS index to the specific
// instantiation of that index used within knowhere.
template <bool B, cuvs_proto::cuvs_index_kind IndexKind, typename DataType>
struct cuvs_index_type_mapper : std::false_type {};

template <typename DataType>
struct cuvs_index_type_mapper<true, cuvs_proto::cuvs_index_kind::brute_force, DataType> : std::true_type {
    using data_type = DataType;
    using indexing_type = cuvs_indexing_t<cuvs_proto::cuvs_index_kind::brute_force>;
    using type = cuvs_proto::cuvs_index<cuvs::neighbors::brute_force::index, data_type, float>;
    using underlying_index_type = typename type::vector_index_type;
    using index_params_type = typename type::index_params_type;
    using search_params_type = typename type::search_params_type;
};
template <typename DataType>
struct cuvs_index_type_mapper<true, cuvs_proto::cuvs_index_kind::ivf_flat, DataType> : std::true_type {
    using data_type = DataType;
    using indexing_type = cuvs_indexing_t<cuvs_proto::cuvs_index_kind::ivf_flat>;
    using type = cuvs_proto::cuvs_index<cuvs::neighbors::ivf_flat::index, data_type, indexing_type>;
    using underlying_index_type = typename type::vector_index_type;
    using index_params_type = typename type::index_params_type;
    using search_params_type = typename type::search_params_type;
};
template <typename DataType>
struct cuvs_index_type_mapper<true, cuvs_proto::cuvs_index_kind::ivf_pq, DataType> : std::true_type {
    using data_type = DataType;
    using indexing_type = cuvs_indexing_t<cuvs_proto::cuvs_index_kind::ivf_pq>;
    using type = cuvs_proto::cuvs_index<cuvs::neighbors::ivf_pq::index, indexing_type>;
    using underlying_index_type = typename type::vector_index_type;
    using index_params_type = typename type::index_params_type;
    using search_params_type = typename type::search_params_type;
};
template <typename DataType>
struct cuvs_index_type_mapper<true, cuvs_proto::cuvs_index_kind::cagra, DataType> : std::true_type {
    using data_type = DataType;
    using indexing_type = cuvs_indexing_t<cuvs_proto::cuvs_index_kind::cagra>;
    using type = cuvs_proto::cuvs_index<cuvs::neighbors::cagra::index, data_type, indexing_type>;
    using underlying_index_type = typename type::vector_index_type;
    using index_params_type = typename type::index_params_type;
    using search_params_type = typename type::search_params_type;
};
template <typename DataType>
struct cuvs_index_type_mapper<true, cuvs_proto::cuvs_index_kind::vamana, DataType> : std::true_type {
    using data_type = DataType;
    using indexing_type = cuvs_indexing_t<cuvs_proto::cuvs_index_kind::vamana>;
    using type = cuvs_proto::cuvs_index<cuvs::neighbors::vamana::index, data_type, indexing_type>;
    using underlying_index_type = typename type::vector_index_type;
    using index_params_type = typename type::index_params_type;
    using search_params_type = typename type::search_params_type;
};

template <typename U, typename V>
struct check_valid_entry {
    __device__ __host__
    check_valid_entry(U max_distance, V max_id)
        : max_distance_(max_distance), max_id_(max_id) {
    }
    __device__ thrust::tuple<V, U>
    operator()(V id, U distance) {
        if (distance >= max_distance_ || id >= max_id_)
            return thrust::tuple<V, U>(V{-1}, distance);
        if (distance < 0)
            return thrust::tuple<V, U>(id, U{0});
        return thrust::tuple<V, U>(id, distance);
    }

 private:
    U max_distance_;
    V max_id_;
};

}  // namespace detail

template <cuvs_proto::cuvs_index_kind IndexKind, typename DataType>
using cuvs_index_t = typename detail::cuvs_index_type_mapper<true, IndexKind, DataType>::type;

template <cuvs_proto::cuvs_index_kind IndexKind>
using cuvs_index_params_t = typename detail::cuvs_index_type_mapper<true, IndexKind, float>::index_params_type;
template <cuvs_proto::cuvs_index_kind IndexKind>
using cuvs_search_params_t = typename detail::cuvs_index_type_mapper<true, IndexKind, float>::search_params_type;

// Metrics are passed between knowhere and cuVS as strings to avoid tight
// coupling between the implementation details of either one.
[[nodiscard]] inline auto
metric_string_to_cuvs_distance_type(std::string const& metric_string) {
    auto result = cuvs::distance::DistanceType::L2Expanded;
    if (metric_string == "L2") {
        result = cuvs::distance::DistanceType::L2Expanded;
    } else if (metric_string == "COSINE") {
        result = cuvs::distance::DistanceType::InnerProduct;
    } else if (metric_string == "L2SqrtExpanded") {
        result = cuvs::distance::DistanceType::L2SqrtExpanded;
    } else if (metric_string == "CosineExpanded") {
        result = cuvs::distance::DistanceType::CosineExpanded;
    } else if (metric_string == "L1") {
        result = cuvs::distance::DistanceType::L1;
    } else if (metric_string == "L2Unexpanded") {
        result = cuvs::distance::DistanceType::L2Unexpanded;
    } else if (metric_string == "L2SqrtUnexpanded") {
        result = cuvs::distance::DistanceType::L2SqrtUnexpanded;
    } else if (metric_string == "IP") {
        result = cuvs::distance::DistanceType::InnerProduct;
    } else if (metric_string == "Linf") {
        result = cuvs::distance::DistanceType::Linf;
    } else if (metric_string == "Canberra") {
        result = cuvs::distance::DistanceType::Canberra;
    } else if (metric_string == "LpUnexpanded") {
        result = cuvs::distance::DistanceType::LpUnexpanded;
    } else if (metric_string == "CorrelationExpanded") {
        result = cuvs::distance::DistanceType::CorrelationExpanded;
    } else if (metric_string == "JACCARD") {
        result = cuvs::distance::DistanceType::JaccardExpanded;
    } else if (metric_string == "HellingerExpanded") {
        result = cuvs::distance::DistanceType::HellingerExpanded;
    } else if (metric_string == "Haversine") {
        result = cuvs::distance::DistanceType::Haversine;
    } else if (metric_string == "BrayCurtis") {
        result = cuvs::distance::DistanceType::BrayCurtis;
    } else if (metric_string == "JensenShannon") {
        result = cuvs::distance::DistanceType::JensenShannon;
    } else if (metric_string == "HAMMING") {
        result = cuvs::distance::DistanceType::BitwiseHamming;
    } else if (metric_string == "KLDivergence") {
        result = cuvs::distance::DistanceType::KLDivergence;
    } else if (metric_string == "RusselRaoExpanded") {
        result = cuvs::distance::DistanceType::RusselRaoExpanded;
    } else if (metric_string == "DiceExpanded") {
        result = cuvs::distance::DistanceType::DiceExpanded;
    } else if (metric_string == "Precomputed") {
        result = cuvs::distance::DistanceType::Precomputed;
    } else {
        RAFT_FAIL("Unrecognized metric type %s", metric_string.c_str());
    }
    return result;
}

[[nodiscard]] inline auto
codebook_string_to_cuvs_codebook_gen(std::string const& codebook_string) {
    auto result = cuvs::neighbors::ivf_pq::codebook_gen::PER_SUBSPACE;
    if (codebook_string == "PER_SUBSPACE") {
        result = cuvs::neighbors::ivf_pq::codebook_gen::PER_SUBSPACE;
    } else if (codebook_string == "PER_CLUSTER") {
        result = cuvs::neighbors::ivf_pq::codebook_gen::PER_CLUSTER;
    } else {
        RAFT_FAIL("Unrecognized codebook type %s", codebook_string.c_str());
    }
    return result;
}
[[nodiscard]] inline auto
build_algo_string_to_cagra_build_algo(std::string const& algo_string, int intermediate_graph_degree,
                                      int nn_descent_niter, cuvs::distance::DistanceType metric) {
    std::variant<std::monostate, cuvs::neighbors::cagra::graph_build_params::ivf_pq_params,
                 cuvs::neighbors::cagra::graph_build_params::nn_descent_params,
                 cuvs::neighbors::cagra::graph_build_params::iterative_search_params>
        result = cuvs::neighbors::cagra::graph_build_params::ivf_pq_params();
    if (algo_string == "IVF_PQ") {
        result = cuvs::neighbors::cagra::graph_build_params::ivf_pq_params();
    } else if (algo_string == "NN_DESCENT") {
        auto nn_desc_params =
            cuvs::neighbors::cagra::graph_build_params::nn_descent_params(intermediate_graph_degree, metric);
        nn_desc_params.max_iterations = nn_descent_niter;
        result = nn_desc_params;
    } else if (algo_string == "ITERATIVE") {
        result = cuvs::neighbors::cagra::graph_build_params::iterative_search_params();
    } else {
        RAFT_FAIL("Unrecognized CAGRA build algo %s", algo_string.c_str());
    }
    return result;
}

[[nodiscard]] inline auto
search_algo_string_to_cagra_search_algo(std::string const& algo_string) {
    auto result = cuvs::neighbors::cagra::search_algo::AUTO;
    if (algo_string == "SINGLE_CTA") {
        result = cuvs::neighbors::cagra::search_algo::SINGLE_CTA;
    } else if (algo_string == "MULTI_CTA") {
        result = cuvs::neighbors::cagra::search_algo::MULTI_CTA;
    } else if (algo_string == "MULTI_KERNEL") {
        result = cuvs::neighbors::cagra::search_algo::MULTI_KERNEL;
    } else if (algo_string == "AUTO") {
        result = cuvs::neighbors::cagra::search_algo::AUTO;
    } else {
        RAFT_FAIL("Unrecognized CAGRA search algo %s", algo_string.c_str());
    }
    return result;
}

[[nodiscard]] inline auto
hashmap_mode_string_to_cagra_hashmap_mode(std::string const& mode_string) {
    auto result = cuvs::neighbors::cagra::hash_mode::AUTO;
    if (mode_string == "HASH") {
        result = cuvs::neighbors::cagra::hash_mode::HASH;
    } else if (mode_string == "SMALL") {
        result = cuvs::neighbors::cagra::hash_mode::SMALL;
    } else if (mode_string == "AUTO") {
        result = cuvs::neighbors::cagra::hash_mode::AUTO;
    } else {
        RAFT_FAIL("Unrecognized CAGRA hash mode %s", mode_string.c_str());
    }
    return result;
}

[[nodiscard]] inline auto
dtype_string_to_cuda_dtype(std::string const& dtype_string) {
    auto result = CUDA_R_32F;
    if (dtype_string == "CUDA_R_16F") {
        result = CUDA_R_16F;
    } else if (dtype_string == "CUDA_C_16F") {
        result = CUDA_C_16F;
    } else if (dtype_string == "CUDA_R_16BF") {
        result = CUDA_R_16BF;
    } else if (dtype_string == "CUDA_R_32F") {
        result = CUDA_R_32F;
    } else if (dtype_string == "CUDA_C_32F") {
        result = CUDA_C_32F;
    } else if (dtype_string == "CUDA_R_64F") {
        result = CUDA_R_64F;
    } else if (dtype_string == "CUDA_C_64F") {
        result = CUDA_C_64F;
    } else if (dtype_string == "CUDA_R_8I") {
        result = CUDA_R_8I;
    } else if (dtype_string == "CUDA_C_8I") {
        result = CUDA_C_8I;
    } else if (dtype_string == "CUDA_R_8U") {
        result = CUDA_R_8U;
    } else if (dtype_string == "CUDA_C_8U") {
        result = CUDA_C_8U;
    } else if (dtype_string == "CUDA_R_32I") {
        result = CUDA_R_32I;
    } else if (dtype_string == "CUDA_C_32I") {
        result = CUDA_C_32I;
#if __CUDACC_VER_MAJOR__ >= 12
    } else if (dtype_string == "CUDA_R_8F_E4M3") {
        result = CUDA_R_8F_E4M3;
    } else if (dtype_string == "CUDA_R_8F_E5M2") {
        result = CUDA_R_8F_E5M2;
#endif
    } else {
        RAFT_FAIL("Unrecognized dtype %s", dtype_string.c_str());
    }
    return result;
}

// Given a generic config without cuVS symbols, convert to cuVS index build
// parameters
template <cuvs_proto::cuvs_index_kind IndexKind>
[[nodiscard]] auto
config_to_index_params(cuvs_knowhere_config const& raw_config) {
    RAFT_EXPECTS(raw_config.index_type == IndexKind, "Incorrect index type for this index");
    auto config = validate_cuvs_knowhere_config(raw_config);
    auto result = cuvs_index_params_t<IndexKind>{};

    result.metric = metric_string_to_cuvs_distance_type(config.metric_type);
    result.metric_arg = config.metric_arg;

    if constexpr (IndexKind == cuvs_proto::cuvs_index_kind::ivf_flat ||
                  IndexKind == cuvs_proto::cuvs_index_kind::ivf_pq) {
        result.n_lists = *(config.nlist);
        result.kmeans_n_iters = *(config.kmeans_n_iters);
        result.kmeans_trainset_fraction = *(config.kmeans_trainset_fraction);
        result.conservative_memory_allocation = *(config.conservative_memory_allocation);
        result.add_data_on_build = config.add_data_on_build;
    }
    if constexpr (IndexKind == cuvs_proto::cuvs_index_kind::ivf_flat) {
        result.adaptive_centers = *(config.adaptive_centers);
    }
    if constexpr (IndexKind == cuvs_proto::cuvs_index_kind::ivf_pq) {
        result.pq_dim = *(config.m);
        result.pq_bits = *(config.nbits);
        result.codebook_kind = codebook_string_to_cuvs_codebook_gen(*(config.codebook_kind));
        result.force_random_rotation = *(config.force_random_rotation);
    }
    if constexpr (IndexKind == cuvs_proto::cuvs_index_kind::cagra) {
        result.intermediate_graph_degree = *(config.intermediate_graph_degree);
        result.graph_degree = *(config.graph_degree);
        result.attach_dataset_on_build = config.add_data_on_build;
        // TODO(mide): add compression
        result.graph_build_params = build_algo_string_to_cagra_build_algo(
            *(config.build_algo), result.intermediate_graph_degree, *(config.nn_descent_niter), result.metric);
    }
    if constexpr (IndexKind == cuvs_proto::cuvs_index_kind::vamana) {
        // cuVS Vamana only supports L2. For COSINE, we use L2 after normalization
        // (normalized L2 distance is equivalent to COSINE similarity for ranking).
        result.metric = cuvs::distance::DistanceType::L2Expanded;
        result.graph_degree = *(config.graph_degree);
        result.visited_size = *(config.visited_size);
        result.vamana_iters = *(config.vamana_iters);
        result.alpha = *(config.alpha);
        result.max_fraction = *(config.max_fraction);
        result.batch_base = *(config.batch_base);
        result.queue_size = *(config.queue_size);
        result.reverse_batchsize = 1000000;
    }

    return result;
}

// Given a generic config without cuVS symbols, convert to cuVS index search
// parameters
template <cuvs_proto::cuvs_index_kind IndexKind>
[[nodiscard]] auto
config_to_search_params(cuvs_knowhere_config const& raw_config) {
    RAFT_EXPECTS(raw_config.index_type == IndexKind, "Incorrect index type for this index");
    auto config = validate_cuvs_knowhere_config(raw_config);
    auto result = cuvs_search_params_t<IndexKind>{};
    if constexpr (IndexKind == cuvs_proto::cuvs_index_kind::ivf_flat ||
                  IndexKind == cuvs_proto::cuvs_index_kind::ivf_pq) {
        result.n_probes = *(config.nprobe);
    }
    if constexpr (IndexKind == cuvs_proto::cuvs_index_kind::ivf_pq) {
        result.lut_dtype = dtype_string_to_cuda_dtype(*(config.lookup_table_dtype));
        result.internal_distance_dtype = dtype_string_to_cuda_dtype(*(config.internal_distance_dtype));
        result.preferred_shmem_carveout = *(config.preferred_shmem_carveout);
    }
    if constexpr (IndexKind == cuvs_proto::cuvs_index_kind::cagra) {
        result.max_queries = *(config.max_queries);
        result.itopk_size = *(config.itopk_size);
        result.max_iterations = *(config.max_iterations);
        result.algo = search_algo_string_to_cagra_search_algo(*(config.search_algo));
        result.team_size = *(config.team_size);
        result.search_width = *(config.search_width);
        result.min_iterations = *(config.min_iterations);
        result.thread_block_size = *(config.thread_block_size);
        result.hashmap_mode = hashmap_mode_string_to_cagra_hashmap_mode(*(config.hashmap_mode));
        result.hashmap_min_bitlen = *(config.hashmap_min_bitlen);
        result.hashmap_max_fill_rate = *(config.hashmap_max_fill_rate);
        result.persistent = *(config.persistent);
    }
    if constexpr (IndexKind == cuvs_proto::cuvs_index_kind::vamana) {
        result.max_queries = *(config.max_queries);
        result.itopk_size = *(config.itopk_size);
        result.max_iterations = *(config.max_iterations);
        result.algo = search_algo_string_to_cagra_search_algo(*(config.search_algo));
        result.team_size = *(config.team_size);
        result.search_width = *(config.search_width);
        result.min_iterations = *(config.min_iterations);
        result.thread_block_size = *(config.thread_block_size);
        result.hashmap_mode = hashmap_mode_string_to_cagra_hashmap_mode(*(config.hashmap_mode));
        result.hashmap_min_bitlen = *(config.hashmap_min_bitlen);
        result.hashmap_max_fill_rate = *(config.hashmap_max_fill_rate);
        result.persistent = *(config.persistent);
    }
    return result;
}

inline auto const&
get_device_resources_without_mempool(int device_id = raft::device_setter::get_current_device()) {
    auto thread_local res = std::vector<raft::device_resources>([]() {
        int device_count;
        RAFT_CUDA_TRY(cudaGetDeviceCount(&device_count));
        return device_count;
    }());

    return res[device_id];
}

inline auto
select_device_id() {
    auto static device_count = []() {
        auto result = 0;
        RAFT_CUDA_TRY(cudaGetDeviceCount(&result));
        RAFT_EXPECTS(result != 0, "No CUDA devices found");
        return result;
    }();
    auto static index_counter = std::atomic<int>{0};
    // Use round-robin assignment to distribute indexes across devices
    auto result = index_counter.fetch_add(1) % device_count;
    return result;
}

// This struct is used to connect knowhere to a cuVS index. The implementation
// is provided here, but this header should never be directly included in
// another knowhere header. This ensures that cuVS symbols are not exposed in
// any knowhere header.
template <cuvs_proto::cuvs_index_kind IndexKind, typename DataType>
struct cuvs_knowhere_index<IndexKind, DataType>::impl {
    auto static constexpr index_kind = IndexKind;
    using data_type = typename cuvs_data_type_mapper<DataType>::data_type;
    using indexing_type = cuvs_indexing_t<index_kind>;
    using input_indexing_type = cuvs_input_indexing_t<index_kind>;
    using cuvs_index_type = cuvs_index_t<index_kind, data_type>;

    impl() {
    }

    auto
    is_trained() const {
        return index_.has_value();
    }
    [[nodiscard]] auto
    size() const {
        auto result = std::int64_t{};
        if (is_trained()) {
            result = index_->size();
        }
        return result;
    }
    [[nodiscard]] auto
    dim() const {
        auto result = std::int64_t{};
        if (is_trained()) {
            result = index_->dim();
        }
        return result;
    }

    void
    train(cuvs_knowhere_config const& config, data_type const* data, knowhere_indexing_type row_count,
          knowhere_indexing_type feature_count) {
        if constexpr (std::is_same_v<data_type, std::uint8_t>) {
            // The input feature_count represents the number of bits. Change it to number of bytes
            feature_count = feature_count / (8 * sizeof(data_type));
        }
        auto scoped_device = raft::device_setter{device_id};
        auto index_params = config_to_index_params<index_kind>(config);
        if constexpr (index_kind == cuvs_proto::cuvs_index_kind::ivf_flat ||
                      index_kind == cuvs_proto::cuvs_index_kind::ivf_pq) {
            index_params.n_lists = std::min(knowhere_indexing_type(index_params.n_lists), row_count);
        }
        auto const& res = get_device_resources_without_mempool();
        auto host_data = raft::make_host_matrix_view(data, row_count, feature_count);
        if (config.metric_type == knowhere::metric::COSINE) {
            auto device_data = raft::make_device_matrix<data_type, input_indexing_type>(res, row_count, feature_count);
            auto device_data_view = device_data.view();
            raft::copy(res, device_data_view, host_data);
            raft::linalg::row_normalize<raft::linalg::L2Norm>(res, raft::make_const_mdspan(device_data_view),
                                                              device_data_view);
            auto host_data_view = raft::make_host_matrix_view(const_cast<data_type*>(data), row_count, feature_count);
            raft::copy(res, host_data_view, device_data_view);
            res.sync_stream();
        }

        if (config.cache_dataset_on_device) {
            device_dataset_storage =
                raft::make_device_matrix<data_type, input_indexing_type>(res, row_count, feature_count);
            auto device_data = device_dataset_storage->view();
            raft::copy(res, device_data, host_data);
            index_ = cuvs_index_type::template build<data_type, indexing_type, input_indexing_type>(
                res, index_params, raft::make_const_mdspan(device_data));
        } else {
            index_ = cuvs_index_type::template build<data_type, indexing_type, input_indexing_type>(
                res, index_params, raft::make_const_mdspan(host_data));
        }

        // Auto-serialize graph to file if dataset_name is provided
        if (config.dataset_name.has_value() && !config.dataset_name->empty()) {
            if constexpr (index_kind == cuvs_proto::cuvs_index_kind::cagra ||
                          index_kind == cuvs_proto::cuvs_index_kind::vamana) {
                std::string index_type_str;
                if constexpr (index_kind == cuvs_proto::cuvs_index_kind::cagra) {
                    index_type_str = "cagra";
                } else if constexpr (index_kind == cuvs_proto::cuvs_index_kind::vamana) {
                    index_type_str = "vamana";
                }
                std::string filename = *config.dataset_name + "_" + index_type_str + "_graph.bin";
                try {
                    serialize_graph_to_file(filename);
                } catch (const std::exception& e) {
                    // Log warning but don't fail the training
                    std::cerr << "Warning: Failed to auto-serialize graph to " << filename << ": " << e.what()
                              << std::endl;
                }
            }
        }
    }

    auto
    search(cuvs_knowhere_config const& config, data_type const* data, knowhere_indexing_type row_count,
           knowhere_indexing_type feature_count, knowhere_bitset_data_type const* bitset_data,
           knowhere_bitset_indexing_type bitset_byte_size, knowhere_bitset_indexing_type bitset_size) const {
        if constexpr (std::is_same_v<data_type, std::uint8_t>) {
            // The input feature_count represents the number of bits. Change it to number of bytes
            feature_count = feature_count / (8 * sizeof(data_type));
        }
        auto scoped_device = raft::device_setter{device_id};
        auto const& res = raft::device_resources_manager::get_device_resources();
        auto k = knowhere_indexing_type(config.k);
        auto search_params = config_to_search_params<index_kind>(config);

        auto host_data = raft::make_host_matrix_view(data, row_count, feature_count);
        auto device_data_storage =
            raft::make_device_matrix<data_type, input_indexing_type>(res, row_count, feature_count);
        raft::copy(res, device_data_storage.view(), host_data);

        if (config.metric_type == knowhere::metric::COSINE) {
            auto device_data_view = device_data_storage.view();
            raft::linalg::row_normalize<raft::linalg::L2Norm>(res, raft::make_const_mdspan(device_data_view),
                                                              device_data_view);
        }

        auto device_bitset = std::optional<
            cuvs::core::bitset<knowhere_bitset_internal_data_type, knowhere_bitset_internal_indexing_type>>{};
        if (bitset_data != nullptr && bitset_byte_size != 0) {
            device_bitset =
                cuvs::core::bitset<knowhere_bitset_internal_data_type, knowhere_bitset_internal_indexing_type>(
                    res, bitset_size);
            raft::copy(res,
                       raft::make_device_vector_view<knowhere_bitset_data_type, knowhere_bitset_indexing_type>(
                           reinterpret_cast<knowhere_bitset_data_type*>(device_bitset->data()), bitset_byte_size),
                       raft::make_host_vector_view(bitset_data, bitset_byte_size));
            if (device_bitset) {
                device_bitset->flip(res);
            }
        }

        auto output_size = row_count * k;
        auto ids = std::unique_ptr<knowhere_indexing_type[]>(new knowhere_indexing_type[output_size]);
        auto distances = std::unique_ptr<knowhere_distance_type[]>(new knowhere_distance_type[output_size]);

        auto host_ids = raft::make_host_matrix_view(ids.get(), row_count, k);
        auto host_distances = raft::make_host_matrix_view(distances.get(), row_count, k);

        auto device_ids_storage = raft::make_device_matrix<indexing_type, input_indexing_type>(res, row_count, k);
        auto device_distances_storage =
            raft::make_device_matrix<knowhere_distance_type, input_indexing_type>(res, row_count, k);
        auto device_ids = device_ids_storage.view();
        auto device_distances = device_distances_storage.view();

        RAFT_EXPECTS(index_, "Index has not yet been trained");
        auto dataset_view = device_dataset_storage
                                ? std::make_optional(device_dataset_storage->view())
                                : std::optional<raft::device_matrix_view<const data_type, input_indexing_type>>{};

        if (device_bitset) {
            // cuVS is using uint32_t as filter datatype. Set the original nbits to knowhere's bitset data
            // type to make them compatible.
            auto bitset_view = device_bitset->view();
            bitset_view.set_original_nbits(sizeof(knowhere_bitset_data_type) * 8);
            cuvs_index_type::search(
                res, *index_, search_params, raft::make_const_mdspan(device_data_storage.view()), device_ids,
                device_distances, config.refine_ratio, input_indexing_type{}, dataset_view,
                cuvs::neighbors::filtering::bitset_filter<knowhere_bitset_internal_data_type,
                                                          knowhere_bitset_internal_indexing_type>{bitset_view});
        } else {
            cuvs_index_type::search(res, *index_, search_params, raft::make_const_mdspan(device_data_storage.view()),
                                    device_ids, device_distances, config.refine_ratio, input_indexing_type{},
                                    dataset_view);
        }

        auto device_knowhere_ids_storage =
            std::optional<raft::device_matrix<knowhere_indexing_type, input_indexing_type>>{};
        auto device_knowhere_ids = [&device_knowhere_ids_storage, &res, row_count, k, device_ids]() {
            if constexpr (std::is_signed_v<indexing_type>) {
                return device_ids;
            } else {
                device_knowhere_ids_storage =
                    raft::make_device_matrix<knowhere_indexing_type, input_indexing_type>(res, row_count, k);
                raft::copy(res, device_knowhere_ids_storage->view(), device_ids);
                return device_knowhere_ids_storage->view();
            }
        }();

        auto max_distance =
            std::nextafter(std::numeric_limits<knowhere_distance_type>::max(), knowhere_distance_type{0});
        auto device_post_process = detail::check_valid_entry<knowhere_distance_type, knowhere_indexing_type>{
            max_distance, knowhere_indexing_type(size())};
        thrust::transform(
            raft::resource::get_thrust_policy(res),
            thrust::device_ptr<knowhere_indexing_type>(device_knowhere_ids.data_handle()),
            thrust::device_ptr<knowhere_indexing_type>(device_knowhere_ids.data_handle() + device_knowhere_ids.size()),
            thrust::device_ptr<knowhere_distance_type>(device_distances.data_handle()),
            thrust::make_zip_iterator(
                thrust::make_tuple(thrust::device_ptr<knowhere_indexing_type>(device_knowhere_ids.data_handle()),
                                   thrust::device_ptr<knowhere_distance_type>(device_distances.data_handle()))),
            device_post_process);

        raft::copy(res, host_ids, device_knowhere_ids);
        raft::copy(res, host_distances, device_distances);
        return std::make_tuple(ids.release(), distances.release());
    }
    void
    range_search() const {
        RAFT_FAIL("Range search not yet implemented for RAFT indexes");
    }
    void
    get_vector_by_id() const {
        RAFT_FAIL("Vector reconstruction not yet implemented for RAFT indexes");
    }
    void
    serialize(std::ostream& os) const {
        auto scoped_device = raft::device_setter{device_id};
        auto const& res = get_device_resources_without_mempool();
        RAFT_EXPECTS(index_, "Index has not yet been trained");
        cuvs_index_type::template serialize<data_type, indexing_type>(res, os, *index_);
        if (device_dataset_storage) {
            raft::serialize_scalar(res, os, true);
            raft::serialize_scalar(res, os, device_dataset_storage->extent(0));
            raft::serialize_scalar(res, os, device_dataset_storage->extent(1));
            raft::serialize_mdspan(res, os, device_dataset_storage->view());
        } else {
            raft::serialize_scalar(res, os, false);
        }
    }

    void
    serialize_to_hnswlib(std::ostream& os) const {
        // only cagra can save to hnswlib format
        if constexpr (index_kind == cuvs_proto::cuvs_index_kind::cagra) {
            auto scoped_device = raft::device_setter{device_id};
            auto const& res = get_device_resources_without_mempool();
            RAFT_EXPECTS(index_, "Index has not yet been trained");
            cuvs_index_type::template serialize_to_hnswlib<data_type, indexing_type>(res, os, *index_);
            raft::serialize_scalar(res, os, false);
        }
    }

    void
    serialize_graph_to_file(const std::string& filename) const {
        if constexpr (index_kind == cuvs_proto::cuvs_index_kind::cagra ||
                      index_kind == cuvs_proto::cuvs_index_kind::vamana) {
            auto scoped_device = raft::device_setter{device_id};
            auto const& res = get_device_resources_without_mempool();
            RAFT_EXPECTS(index_, "Index has not yet been trained");

            // Get the graph from the index
            auto graph_view = index_->get_vector_index().graph();

            int64_t num_nodes = graph_view.extent(0);
            int64_t graph_degree = graph_view.extent(1);
            size_t graph_size = num_nodes * graph_degree;

            // Allocate host memory and copy graph
            std::vector<indexing_type> host_graph(graph_size);
            RAFT_CUDA_TRY(cudaMemcpy(host_graph.data(), graph_view.data_handle(), graph_size * sizeof(indexing_type),
                                     cudaMemcpyDeviceToHost));
            raft::resource::sync_stream(res);

            // Write graph to file
            std::ofstream out(filename, std::ios::binary);
            if (!out) {
                throw std::runtime_error("Failed to open file for writing: " + filename);
            }

            // Write header: [num_nodes, graph_degree]
            out.write(reinterpret_cast<const char*>(&num_nodes), sizeof(int64_t));
            out.write(reinterpret_cast<const char*>(&graph_degree), sizeof(int64_t));

            // Write graph data
            out.write(reinterpret_cast<const char*>(host_graph.data()), graph_size * sizeof(indexing_type));

            out.close();
        } else {
            throw std::runtime_error("Graph serialization is only supported for CAGRA and Vamana indexes");
        }
    }

    void
    serialize_graph_to_fbin_format(const std::string& filename) const {
        if constexpr (index_kind == cuvs_proto::cuvs_index_kind::cagra ||
                      index_kind == cuvs_proto::cuvs_index_kind::vamana) {
            auto scoped_device = raft::device_setter{device_id};
            auto const& res = get_device_resources_without_mempool();
            RAFT_EXPECTS(index_, "Index has not yet been trained");

            // Get the graph from the index
            auto graph_view = index_->get_vector_index().graph();

            uint32_t num_vertices = static_cast<uint32_t>(graph_view.extent(0));
            uint32_t graph_degree = static_cast<uint32_t>(graph_view.extent(1));
            size_t graph_size = static_cast<size_t>(num_vertices) * graph_degree;

            // Allocate host memory and copy graph
            std::vector<indexing_type> host_graph(graph_size);
            RAFT_CUDA_TRY(cudaMemcpy(host_graph.data(), graph_view.data_handle(), graph_size * sizeof(indexing_type),
                                     cudaMemcpyDeviceToHost));
            raft::resource::sync_stream(res);

            // Get entry point
            uint32_t entry_point = 0;
            if constexpr (index_kind == cuvs_proto::cuvs_index_kind::vamana) {
                entry_point = static_cast<uint32_t>(index_->get_vector_index().medoid());
            } else {
                // CAGRA: compute entry point as the nearest point to the centroid
                entry_point = compute_centroid_nearest();
            }

            // Invalid neighbor value for indexing_type (uint32_t -> UINT32_MAX, uint64_t -> UINT64_MAX)
            constexpr indexing_type invalid_neighbor = std::numeric_limits<indexing_type>::max();

            // Build CSR format: filter invalid neighbors
            std::vector<uint32_t> indices(num_vertices + 1);
            std::vector<int32_t> neighbors;
            neighbors.reserve(graph_size);

            indices[0] = 0;
            for (uint32_t u = 0; u < num_vertices; ++u) {
                size_t row_offset = static_cast<size_t>(u) * graph_degree;
                for (uint32_t k = 0; k < graph_degree; ++k) {
                    indexing_type neighbor = host_graph[row_offset + k];
                    // Filter invalid neighbors (invalid value can be anywhere in the neighbor list)
                    if (neighbor != invalid_neighbor) {
                        neighbors.push_back(static_cast<int32_t>(neighbor));
                    }
                }
                indices[u + 1] = static_cast<uint32_t>(neighbors.size());
            }

            // Write graph to file in CSR format
            std::ofstream out(filename, std::ios::binary);
            if (!out) {
                throw std::runtime_error("Failed to open file for writing: " + filename);
            }

            // Write header: [num_vertices (uint32), entry_point (uint32)]
            out.write(reinterpret_cast<const char*>(&num_vertices), sizeof(uint32_t));
            out.write(reinterpret_cast<const char*>(&entry_point), sizeof(uint32_t));

            // Write CSR indices
            out.write(reinterpret_cast<const char*>(indices.data()), indices.size() * sizeof(uint32_t));

            // Write neighbors
            out.write(reinterpret_cast<const char*>(neighbors.data()), neighbors.size() * sizeof(int32_t));

            out.close();
        } else {
            throw std::runtime_error(
                "Cardinal format graph serialization is only supported for CAGRA and Vamana indexes");
        }
    }

    uint32_t
    compute_centroid_nearest() const {
        if constexpr (index_kind == cuvs_proto::cuvs_index_kind::cagra) {
            auto const& res = get_device_resources_without_mempool();
            auto dataset_view = index_->get_vector_index().dataset();

            // Check if dataset is available
            if (dataset_view.data_handle() == nullptr || dataset_view.extent(0) == 0) {
                throw std::runtime_error(
                    "Cannot compute entry point: dataset is not available in CAGRA index. "
                    "Please ensure the index was built with dataset included.");
            }

            int64_t num_vectors = dataset_view.extent(0);
            int64_t dim = dataset_view.extent(1);

            // Copy dataset to host
            std::vector<data_type> host_dataset(num_vectors * dim);
            RAFT_CUDA_TRY(cudaMemcpy2D(host_dataset.data(), dim * sizeof(data_type), dataset_view.data_handle(),
                                       dataset_view.stride(0) * sizeof(data_type), dim * sizeof(data_type), num_vectors,
                                       cudaMemcpyDeviceToHost));
            raft::resource::sync_stream(res);

            // Compute centroid (mean of all vectors)
            std::vector<double> centroid(dim, 0.0);
            for (int64_t i = 0; i < num_vectors; ++i) {
                for (int64_t j = 0; j < dim; ++j) {
                    centroid[j] += static_cast<double>(host_dataset[i * dim + j]);
                }
            }
            for (int64_t j = 0; j < dim; ++j) {
                centroid[j] /= num_vectors;
            }

            // Find the nearest point to centroid (using L2 distance)
            uint32_t nearest_id = 0;
            double min_dist = std::numeric_limits<double>::max();
            for (int64_t i = 0; i < num_vectors; ++i) {
                double dist = 0.0;
                for (int64_t j = 0; j < dim; ++j) {
                    double diff = centroid[j] - static_cast<double>(host_dataset[i * dim + j]);
                    dist += diff * diff;
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    nearest_id = static_cast<uint32_t>(i);
                }
            }
            return nearest_id;
        }
        return 0;
    }

    auto static deserialize(std::istream& is)
        -> std::unique_ptr<typename cuvs_knowhere_index<index_kind, DataType>::impl> {
        auto static device_count = []() {
            auto result = 0;
            RAFT_CUDA_TRY(cudaGetDeviceCount(&result));
            RAFT_EXPECTS(result != 0, "No CUDA devices found");
            return result;
        }();
        // The lazy allocation mode cannot completely eliminate uneven distribution, but it can alleviate it well.
        int new_device_id = 0;
        size_t free, total;
        size_t max_free = 0;
        for (int i = 0; i < device_count; ++i) {
            auto scoped_device = raft::device_setter{i};
            RAFT_CUDA_TRY(cudaMemGetInfo(&free, &total));
            if (max_free < free) {
                max_free = free;
                new_device_id = i;
            }
        }
        auto scoped_device = raft::device_setter{new_device_id};
        auto const& res = get_device_resources_without_mempool();
        auto des_index = cuvs_index_type::template deserialize<data_type, indexing_type>(res, is);

        auto dataset = std::optional<raft::device_matrix<data_type, input_indexing_type>>{};
        auto has_dataset = raft::deserialize_scalar<bool>(res, is);
        if (has_dataset) {
            auto rows = raft::deserialize_scalar<input_indexing_type>(res, is);
            auto cols = raft::deserialize_scalar<input_indexing_type>(res, is);
            dataset = raft::make_device_matrix<data_type, input_indexing_type>(res, rows, cols);
            raft::deserialize_mdspan(res, is, dataset->view());
            if constexpr (index_kind == cuvs_proto::cuvs_index_kind::brute_force ||
                          index_kind == cuvs_proto::cuvs_index_kind::cagra) {
                cuvs_index_type::template update_dataset<data_type, input_indexing_type>(
                    res, des_index, raft::make_const_mdspan(dataset->view()));
            }
        }
        return std::make_unique<typename cuvs_knowhere_index<index_kind, DataType>::impl>(
            std::move(des_index), new_device_id, std::move(dataset));
    }

    void
    synchronize(bool is_without_mempool = false) const {
        auto scoped_device = raft::device_setter{device_id};
        if (is_without_mempool) {
            get_device_resources_without_mempool().sync_stream();

        } else {
            raft::device_resources_manager::get_device_resources().sync_stream();
        }
    }
    impl(cuvs_index_type&& index, int new_device_id,
         std::optional<raft::device_matrix<data_type, input_indexing_type>>&& dataset)
        : index_{std::move(index)}, device_id{new_device_id}, device_dataset_storage{std::move(dataset)} {
    }

 private:
    std::optional<cuvs_index_type> index_ = std::nullopt;
    int device_id = select_device_id();
    std::optional<raft::device_matrix<data_type, input_indexing_type>> device_dataset_storage = std::nullopt;
};

template <cuvs_proto::cuvs_index_kind IndexKind, typename DataType>
cuvs_knowhere_index<IndexKind, DataType>::cuvs_knowhere_index()
    : pimpl{new cuvs_knowhere_index<IndexKind, DataType>::impl()} {
}

template <cuvs_proto::cuvs_index_kind IndexKind, typename DataType>
cuvs_knowhere_index<IndexKind, DataType>::~cuvs_knowhere_index<IndexKind, DataType>() = default;

template <cuvs_proto::cuvs_index_kind IndexKind, typename DataType>
cuvs_knowhere_index<IndexKind, DataType>::cuvs_knowhere_index(cuvs_knowhere_index<IndexKind, DataType>&& other)
    : pimpl{std::move(other.pimpl)} {
}

template <cuvs_proto::cuvs_index_kind IndexKind, typename DataType>
cuvs_knowhere_index<IndexKind, DataType>&
cuvs_knowhere_index<IndexKind, DataType>::operator=(cuvs_knowhere_index<IndexKind, DataType>&& other) {
    pimpl = std::move(other.pimpl);
    return *this;
}

template <cuvs_proto::cuvs_index_kind IndexKind, typename DataType>
bool
cuvs_knowhere_index<IndexKind, DataType>::is_trained() const {
    return pimpl->is_trained();
}

template <cuvs_proto::cuvs_index_kind IndexKind, typename DataType>
std::int64_t
cuvs_knowhere_index<IndexKind, DataType>::size() const {
    return pimpl->size();
}

template <cuvs_proto::cuvs_index_kind IndexKind, typename DataType>
std::int64_t
cuvs_knowhere_index<IndexKind, DataType>::dim() const {
    return pimpl->dim();
}

template <cuvs_proto::cuvs_index_kind IndexKind, typename DataType>
void
cuvs_knowhere_index<IndexKind, DataType>::train(cuvs_knowhere_config const& config, data_type const* data,
                                                knowhere_indexing_type row_count,
                                                knowhere_indexing_type feature_count) {
    return pimpl->train(config, data, row_count, feature_count);
}

template <cuvs_proto::cuvs_index_kind IndexKind, typename DataType>
std::tuple<knowhere_indexing_type*, knowhere_distance_type*>
cuvs_knowhere_index<IndexKind, DataType>::search(cuvs_knowhere_config const& config, data_type const* data,
                                                 knowhere_indexing_type row_count, knowhere_indexing_type feature_count,
                                                 knowhere_bitset_data_type const* bitset_data,
                                                 knowhere_bitset_indexing_type bitset_byte_size,
                                                 knowhere_bitset_indexing_type bitset_size) const {
    return pimpl->search(config, data, row_count, feature_count, bitset_data, bitset_byte_size, bitset_size);
}

template <cuvs_proto::cuvs_index_kind IndexKind, typename DataType>
void
cuvs_knowhere_index<IndexKind, DataType>::range_search() const {
    return pimpl->range_search();
}

template <cuvs_proto::cuvs_index_kind IndexKind, typename DataType>
void
cuvs_knowhere_index<IndexKind, DataType>::get_vector_by_id() const {
    return pimpl->get_vector_by_id();
}

template <cuvs_proto::cuvs_index_kind IndexKind, typename DataType>
void
cuvs_knowhere_index<IndexKind, DataType>::serialize(std::ostream& os) const {
    return pimpl->serialize(os);
}

template <cuvs_proto::cuvs_index_kind IndexKind, typename DataType>
void
cuvs_knowhere_index<IndexKind, DataType>::serialize_to_hnswlib(std::ostream& os) const {
    return pimpl->serialize_to_hnswlib(os);
}

template <cuvs_proto::cuvs_index_kind IndexKind, typename DataType>
void
cuvs_knowhere_index<IndexKind, DataType>::serialize_graph_to_file(const std::string& filename) const {
    return pimpl->serialize_graph_to_file(filename);
}

template <cuvs_proto::cuvs_index_kind IndexKind, typename DataType>
void
cuvs_knowhere_index<IndexKind, DataType>::serialize_graph_to_fbin_format(const std::string& filename) const {
    return pimpl->serialize_graph_to_fbin_format(filename);
}

template <cuvs_proto::cuvs_index_kind IndexKind, typename DataType>
cuvs_knowhere_index<IndexKind, DataType>
cuvs_knowhere_index<IndexKind, DataType>::deserialize(std::istream& is) {
    return cuvs_knowhere_index<IndexKind, DataType>(cuvs_knowhere_index<IndexKind, DataType>::impl::deserialize(is));
}

template <cuvs_proto::cuvs_index_kind IndexKind, typename DataType>
void
cuvs_knowhere_index<IndexKind, DataType>::synchronize(bool is_without_mempool) const {
    return pimpl->synchronize(is_without_mempool);
}

}  // namespace cuvs_knowhere
