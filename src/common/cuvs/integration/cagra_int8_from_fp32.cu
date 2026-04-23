/**
 * SPDX-FileCopyrightText: Copyright (c) 2023,NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include "common/cuvs/integration/cagra_int8_from_fp32.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/copy.cuh>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/device_setter.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/serialize.hpp>
#include <raft/linalg/normalize.cuh>

#include "common/cuvs/integration/cuvs_knowhere_config.hpp"
#include "knowhere/comp/index_param.h"

namespace cuvs_knowhere {

namespace {

cuvs::distance::DistanceType
map_metric(const std::string& metric_type) {
    if (metric_type == knowhere::metric::L2) {
        return cuvs::distance::DistanceType::L2Expanded;
    }
    if (metric_type == knowhere::metric::COSINE) {
        // Cosine is implemented as L2 over L2-normalized vectors.
        return cuvs::distance::DistanceType::L2Expanded;
    }
    if (metric_type == knowhere::metric::IP) {
        return cuvs::distance::DistanceType::InnerProduct;
    }
    throw std::runtime_error("unsupported metric for cagra int8-from-fp32 build: " + metric_type);
}

}  // namespace

void
build_and_serialize_cagra_int8_fp32(std::ostream& os, const cuvs_knowhere_config& config, float* fp32_data,
                                    int64_t rows, int64_t cols) {
    auto scoped_device = raft::device_setter{0};
    raft::device_resources res;

    const auto metric = map_metric(config.metric_type);

    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
    };
    auto t0 = clk::now();

    // 1) If COSINE, L2-normalize fp32_data in place so the int8 quantized graph
    //    and the fp32 CPU search use the same vector directions.
    if (config.metric_type == knowhere::metric::COSINE) {
        auto device_data = raft::make_device_matrix<float, int64_t>(res, rows, cols);
        auto device_view = device_data.view();
        auto host_const = raft::make_host_matrix_view<const float, int64_t>(fp32_data, rows, cols);
        raft::copy(res, device_view, host_const);
        raft::linalg::row_normalize<raft::linalg::NormType::L2Norm>(res, raft::make_const_mdspan(device_view),
                                                                    device_view);
        auto host_mut = raft::make_host_matrix_view<float, int64_t>(fp32_data, rows, cols);
        raft::copy(res, host_mut, device_view);
        raft::resource::sync_stream(res);
    }
    auto t_norm = clk::now();

    // 2) Global symmetric int8 quantization.
    const int64_t total = rows * cols;
    float max_abs = 0.0f;
    for (int64_t i = 0; i < total; ++i) {
        float v = std::fabs(fp32_data[i]);
        if (v > max_abs) {
            max_abs = v;
        }
    }
    float scale = max_abs > 0.0f ? (max_abs / 127.0f) : 1.0f;

    std::vector<int8_t> int8_data(static_cast<size_t>(total));
    for (int64_t i = 0; i < total; ++i) {
        float q = std::round(fp32_data[i] / scale);
        if (q > 127.0f) {
            q = 127.0f;
        } else if (q < -128.0f) {
            q = -128.0f;
        }
        int8_data[static_cast<size_t>(i)] = static_cast<int8_t>(q);
    }
    auto t_quant = clk::now();

    // 3) Build int8 CAGRA index. Default graph build algorithm is NN_DESCENT; if
    //    IVF_PQ was requested we let cuVS pick defaults (no per-field override here).
    cuvs::neighbors::cagra::index_params int8_params;
    int8_params.metric = metric;
    int8_params.intermediate_graph_degree = static_cast<uint32_t>(config.intermediate_graph_degree.value_or(128));
    int8_params.graph_degree = static_cast<uint32_t>(config.graph_degree.value_or(64));
    int8_params.attach_dataset_on_build = config.add_data_on_build;
    auto nn_desc_params = cuvs::neighbors::cagra::graph_build_params::nn_descent_params(
        int8_params.intermediate_graph_degree, metric);
    nn_desc_params.max_iterations = config.nn_descent_niter.value_or(20);
    int8_params.graph_build_params = nn_desc_params;

    auto host_int8_view = raft::make_host_matrix_view<const int8_t, int64_t>(int8_data.data(), rows, cols);
    auto int8_index = cuvs::neighbors::cagra::build(res, int8_params, host_int8_view);
    raft::resource::sync_stream(res);
    auto t_build = clk::now();

    // 4) Build an fp32 CAGRA index that reuses the int8-built graph but carries
    //    fp32 data on device. The index constructor stores a reference to the
    //    device dataset; ensure its lifetime covers the serialize call.
    auto device_fp32 = raft::make_device_matrix<float, int64_t>(res, rows, cols);
    auto host_fp32_const = raft::make_host_matrix_view<const float, int64_t>(fp32_data, rows, cols);
    raft::copy(res, device_fp32.view(), host_fp32_const);
    raft::resource::sync_stream(res);

    auto int8_graph_view = int8_index.graph();  // device_matrix_view<uint32_t, int64_t>

    cuvs::neighbors::cagra::index<float, uint32_t> fp32_index(
        res, metric, raft::make_const_mdspan(device_fp32.view()), raft::make_const_mdspan(int8_graph_view));
    auto t_wrap = clk::now();

    // 5) Write the 24-byte prefix that knowhere's cuvs_index.hpp::serialize_to_hnswlib
    //    adds before the raw cuVS hnswlib blob: [metric_type, data_size, dim] as
    //    size_t triples. hnswlib::loadIndex reads these first three fields, so they
    //    must match the fp32 dataset we ship.
    size_t metric_type_sz = 0;
    if (fp32_index.metric() == cuvs::distance::DistanceType::L2Expanded) {
        metric_type_sz = 0;
    } else if (fp32_index.metric() == cuvs::distance::DistanceType::InnerProduct) {
        metric_type_sz = 1;
    } else if (fp32_index.metric() == cuvs::distance::DistanceType::CosineExpanded) {
        metric_type_sz = 2;
    }
    os.write(reinterpret_cast<char*>(&metric_type_sz), sizeof(metric_type_sz));
    size_t data_size_sz = static_cast<size_t>(cols) * sizeof(float);
    os.write(reinterpret_cast<char*>(&data_size_sz), sizeof(data_size_sz));
    size_t dim_sz = static_cast<size_t>(cols);
    os.write(reinterpret_cast<char*>(&dim_sz), sizeof(dim_sz));

    // 6) Write the cuVS hnswlib-format blob (graph + fp32 dataset). Pass the fp32
    //    host view explicitly so serialize_to_hnswlib embeds the fp32 rows.
    cuvs::neighbors::cagra::serialize_to_hnswlib(
        res, os, fp32_index,
        std::make_optional<raft::host_matrix_view<const float, int64_t, raft::row_major>>(host_fp32_const));
    raft::serialize_scalar(res, os, false);
    auto t_ser = clk::now();

    std::cout << "[CAGRA_INT8_TIMING] cosine_norm=" << ms(t0, t_norm)
              << "ms quantize=" << ms(t_norm, t_quant)
              << "ms cuvs_int8_build=" << ms(t_quant, t_build)
              << "ms fp32_wrap=" << ms(t_build, t_wrap)
              << "ms serialize_hnswlib=" << ms(t_wrap, t_ser)
              << "ms total=" << ms(t0, t_ser) << "ms" << std::endl;
}

}  // namespace cuvs_knowhere
