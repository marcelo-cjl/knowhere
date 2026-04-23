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
#include "common/cuvs/integration/vamana_int8_from_fp32.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/vamana.hpp>
#include <raft/core/copy.cuh>
#include <raft/core/device_resources.hpp>
#include <raft/core/device_resources_manager.hpp>
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
    if (metric_type == knowhere::metric::L2 || metric_type == knowhere::metric::COSINE) {
        // Cosine is implemented as L2 of L2-normalized vectors.
        return cuvs::distance::DistanceType::L2Expanded;
    }
    if (metric_type == knowhere::metric::IP) {
        return cuvs::distance::DistanceType::InnerProduct;
    }
    throw std::runtime_error("unsupported metric for vamana int8-from-fp32 build: " + metric_type);
}

}  // namespace

void
build_and_serialize_vamana_int8_fp32(std::ostream& os, const cuvs_knowhere_config& config, float* fp32_data,
                                     int64_t rows, int64_t cols) {
    auto scoped_device = raft::device_setter{0};
    raft::device_resources res;

    // 1) If COSINE, L2-normalize fp32_data in place so that CPU-side distances
    //    and the int8-quantized graph build agree on the same direction.
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

    // 2) Global symmetric int8 quantization: int8 = round(fp32 / scale),
    //    scale = max(|fp32|) / 127.
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

    // 3) Build cuVS int8 vamana from the quantized host data.
    cuvs::neighbors::vamana::index_params params;
    params.metric = map_metric(config.metric_type);
    params.graph_degree = static_cast<uint32_t>(config.graph_degree.value_or(32));
    params.visited_size = static_cast<uint32_t>(config.visited_size.value_or(64));
    params.vamana_iters = config.vamana_iters.value_or(1.0f);
    params.alpha = config.alpha.value_or(1.2f);
    params.max_fraction = config.max_fraction.value_or(0.06f);
    params.batch_base = config.batch_base.value_or(2.0f);
    params.queue_size = static_cast<uint32_t>(config.queue_size.value_or(127));

    // cuVS vamana dereferences the dataset pointer on device, so we must feed a
    // device mdspan (host mdspan triggers cudaErrorIllegalAddress).
    auto host_int8_view = raft::make_host_matrix_view<const int8_t, int64_t>(int8_data.data(), rows, cols);
    auto device_int8 = raft::make_device_matrix<int8_t, int64_t>(res, rows, cols);
    raft::copy(res, device_int8.view(), host_int8_view);
    raft::resource::sync_stream(res);
    auto int8_index = cuvs::neighbors::vamana::build(res, params, raft::make_const_mdspan(device_int8.view()));

    // 4) Serialize header + graph + fp32 dataset in the format consumed by
    //    hnswlib::loadIndexFromVamanaGpuFormat. Layout:
    //      [metric:int][dim:int64][medoid:uint32][num_nodes:int64][graph_degree:int64]
    //      [graph mdspan]
    //      [has_dataset:bool][dataset_rows:int64][dataset_cols:int64][dataset mdspan]
    auto serialized_metric = static_cast<int>(int8_index.metric());
    raft::serialize_scalar(res, os, serialized_metric);

    int64_t serialized_dim = static_cast<int64_t>(int8_index.dim());
    raft::serialize_scalar(res, os, serialized_dim);

    auto medoid_id = int8_index.medoid();
    raft::serialize_scalar(res, os, medoid_id);

    auto graph_view = int8_index.graph();
    int64_t num_nodes = graph_view.extent(0);
    int64_t graph_degree = graph_view.extent(1);
    raft::serialize_scalar(res, os, num_nodes);
    raft::serialize_scalar(res, os, graph_degree);

    auto host_graph = raft::make_host_matrix<uint32_t, int64_t>(num_nodes, graph_degree);
    raft::copy(res, host_graph.view(), graph_view);
    raft::resource::sync_stream(res);
    raft::serialize_mdspan(res, os, host_graph.view());

    raft::serialize_scalar(res, os, true);
    raft::serialize_scalar(res, os, rows);
    raft::serialize_scalar(res, os, cols);

    auto fp32_host_view = raft::make_host_matrix_view<const float, int64_t>(fp32_data, rows, cols);
    raft::serialize_mdspan(res, os, fp32_host_view);
}

}  // namespace cuvs_knowhere
