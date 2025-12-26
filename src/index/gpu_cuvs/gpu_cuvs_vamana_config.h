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
#ifndef GPU_CUVS_VAMANA_CONFIG_H
#define GPU_CUVS_VAMANA_CONFIG_H

#include "common/cuvs/integration/cuvs_knowhere_config.hpp"
#include "common/cuvs/proto/cuvs_index_kind.hpp"
#include "index/ivf/ivf_config.h"
#include "knowhere/config.h"

namespace knowhere {

struct GpuCuvsVamanaConfig : public BaseConfig {
    CFG_INT graph_degree;
    CFG_INT visited_size;
    CFG_FLOAT vamana_iters;
    CFG_FLOAT alpha;
    CFG_FLOAT max_fraction;
    CFG_FLOAT batch_base;
    CFG_INT queue_size;
    CFG_BOOL adapt_for_cpu;
    CFG_INT ef;

    KNOHWERE_DECLARE_CONFIG(GpuCuvsVamanaConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(graph_degree).description("degree of knn graph").set_default(32).for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(visited_size).description("visited size").set_default(64).for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(vamana_iters).description("vamana iters").set_default(1.0).for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(alpha).description("alpha").set_default(1.2).for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(max_fraction).description("max fraction").set_default(0.06).for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(batch_base).description("batch base").set_default(2).for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(queue_size).description("queue size").set_default(127).for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(adapt_for_cpu)
            .description("train on GPU search on CPU")
            .set_default(false)
            .for_train()
            .for_deserialize();
        KNOWHERE_CONFIG_DECLARE_FIELD(ef)
            .description("hnsw ef")
            .allow_empty_without_default()
            .set_range(1, std::numeric_limits<CFG_INT::value_type>::max())
            .for_search();
    }

    Status
    CheckAndAdjust(PARAM_TYPE param_type, std::string* err_msg) override {
        if (param_type == PARAM_TYPE::TRAIN) {
            // cuVS Vamana only supports L2. COSINE is supported via L2 normalization + L2 distance.
            constexpr std::array<std::string_view, 2> legal_metric_list{"L2", "COSINE"};
            std::string metric = metric_type.value();
            if (std::find(legal_metric_list.begin(), legal_metric_list.end(), metric) == legal_metric_list.end()) {
                std::string msg =
                    "metric type " + metric + " not found or not supported, supported: [L2 COSINE]";
                return HandleError(err_msg, msg, Status::invalid_metric_type);
            }
        }
        return Status::success;
    }
};

[[nodiscard]] inline auto
to_cuvs_knowhere_config(GpuCuvsVamanaConfig const& cfg) {
    auto result = cuvs_knowhere::cuvs_knowhere_config{cuvs_proto::cuvs_index_kind::vamana};

    // Vamana build requires device memory - cuVS vamana implementation directly passes
    // dataset to GPU kernels, so host memory will cause cudaErrorIllegalAddress.
    result.cache_dataset_on_device = true;
    result.metric_type = cfg.metric_type.value();
    result.graph_degree = cfg.graph_degree;
    result.visited_size = cfg.visited_size;
    result.vamana_iters = cfg.vamana_iters;
    result.alpha = cfg.alpha;
    result.max_fraction = cfg.max_fraction;
    result.batch_base = cfg.batch_base;
    result.queue_size = cfg.queue_size;

    return result;
}

}  // namespace knowhere

#endif /*GPU_CUVS_VAMANA_CONFIG_H*/
