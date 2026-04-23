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

#include <cstring>
#include <sstream>
#include <vector>

#include "common/cuvs/integration/vamana_int8_from_fp32.hpp"
#include "common/cuvs/proto/cuvs_index_kind.hpp"
#include "gpu_cuvs.h"
#include "hnswlib/hnswalg.h"
#include "knowhere/dataset.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/index/index_node_thread_pool_wrapper.h"
#include "raft/core/device_resources.hpp"
#include "raft/util/cuda_rt_essentials.hpp"

namespace knowhere {

template <typename DataType>
class GpuCuvsVamanaHybridIndexNode : public GpuCuvsVamanaIndexNode<DataType> {
 public:
    using DistType = float;
    GpuCuvsVamanaHybridIndexNode(int32_t version, const Object& object)
        : GpuCuvsVamanaIndexNode<DataType>(version, object) {
    }

    Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        const GpuCuvsVamanaConfig& vamana_cfg = static_cast<const GpuCuvsVamanaConfig&>(*cfg);
        if (vamana_cfg.adapt_for_cpu.value())
            adapt_for_cpu = true;

        // Build graph on GPU with int8-quantized data but serialize the original fp32
        // dataset so that hnswlib CPU search still uses full precision. Only valid for
        // the fp32 Hybrid instance + adapt_for_cpu mode. The actual cuVS work runs here
        // in Train() so that Build-time metrics reflect reality; Serialize() below just
        // ships the cached bytes.
        if constexpr (std::is_same_v<DataType, float>) {
            if (adapt_for_cpu && vamana_cfg.build_with_int8.value_or(false)) {
                int8_build_mode_ = true;
                saved_rows_ = dataset->GetRows();
                saved_cols_ = dataset->GetDim();
                auto cuvs_config = to_cuvs_knowhere_config(vamana_cfg);
                // Copy fp32 input into a scratch buffer (normalization + quantization
                // both mutate it; we don't want to touch the caller's tensor).
                std::vector<float> fp32_scratch(reinterpret_cast<const float*>(dataset->GetTensor()),
                                                reinterpret_cast<const float*>(dataset->GetTensor()) +
                                                    static_cast<size_t>(saved_rows_) *
                                                        static_cast<size_t>(saved_cols_));
                std::stringbuf buf;
                try {
                    std::ostream os(&buf);
                    cuvs_knowhere::build_and_serialize_vamana_int8_fp32(os, cuvs_config, fp32_scratch.data(),
                                                                        saved_rows_, saved_cols_);
                    os.flush();
                } catch (const std::exception& e) {
                    LOG_KNOWHERE_ERROR_ << "int8 vamana build failed: " << e.what();
                    return Status::cuvs_inner_error;
                }
                int8_serialized_bytes_ = buf.str();  // keep until Serialize()
                return Status::success;
            }
        }

        return GpuCuvsVamanaIndexNode<DataType>::Train(dataset, cfg, use_knowhere_build_pool);
    }

    Status
    Serialize(BinarySet& binset) const override {
        if (int8_build_mode_) {
            if (int8_serialized_bytes_.empty()) {
                LOG_KNOWHERE_ERROR_ << "int8 build mode: serialized payload missing";
                return Status::empty_index;
            }
            std::shared_ptr<uint8_t[]> index_binary(new (std::nothrow) uint8_t[int8_serialized_bytes_.size()]);
            if (!index_binary) {
                return Status::malloc_error;
            }
            std::memcpy(index_binary.get(), int8_serialized_bytes_.data(), int8_serialized_bytes_.size());
            binset.Append(std::string(this->Type()), index_binary, int8_serialized_bytes_.size());
            return Status::success;
        }
        // For adapt_for_cpu, we still use normal GPU format serialization
        // The deserialization will convert GPU format to CPU-searchable format
        return GpuCuvsVamanaIndexNode<DataType>::Serialize(binset);
    }

    expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
           milvus::OpContext* op_context) const override {
        if (!adapt_for_cpu || hnsw_index_ == nullptr)
            return GpuCuvsVamanaIndexNode<DataType>::Search(dataset, std::move(cfg), bitset, op_context);
        auto nq = dataset->GetRows();
        auto xq = dataset->GetTensor();

        auto vamana_cfg = static_cast<const GpuCuvsVamanaConfig&>(*cfg);
        auto k = vamana_cfg.k.value();
        auto ef = vamana_cfg.ef.has_value() ? vamana_cfg.ef.value() : 100;

        auto p_id = std::make_unique<int64_t[]>(k * nq);
        auto p_dist = std::make_unique<DistType[]>(k * nq);

        hnswlib::SearchParam param{(size_t)ef};
        bool transform = (hnsw_index_->metric_type_ == hnswlib::Metric::INNER_PRODUCT ||
                          hnsw_index_->metric_type_ == hnswlib::Metric::COSINE);

        for (int i = 0; i < nq; ++i) {
            auto p_id_ptr = p_id.get();
            auto p_dist_ptr = p_dist.get();
            auto single_query = (const char*)xq + i * hnsw_index_->data_size_;
            auto rst = hnsw_index_->searchKnn(single_query, k, bitset, &param);
            size_t rst_size = rst.size();
            auto p_single_dis = p_dist_ptr + i * k;
            auto p_single_id = p_id_ptr + i * k;
            for (size_t idx = 0; idx < rst_size; ++idx) {
                const auto& [dist, id] = rst[idx];
                p_single_dis[idx] = transform ? (-dist) : dist;
                p_single_id[idx] = id;
            }
            for (size_t idx = rst_size; idx < (size_t)k; idx++) {
                p_single_dis[idx] = DistType(1.0 / 0.0);
                p_single_id[idx] = -1;
            }
        }

        auto res = GenResultDataSet(nq, k, p_id.release(), p_dist.release());

        return res;
    }

    int64_t
    Count() const override {
        if (int8_build_mode_ && hnsw_index_ == nullptr)
            return saved_rows_;
        if (!adapt_for_cpu)
            return GpuCuvsVamanaIndexNode<DataType>::Count();
        if (!hnsw_index_) {
            return 0;
        }
        return hnsw_index_->cur_element_count;
    }

    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config> cfg) override {
        const GpuCuvsVamanaConfig& vamana_cfg = static_cast<const GpuCuvsVamanaConfig&>(*cfg);
        if (vamana_cfg.adapt_for_cpu.value()) {
            adapt_for_cpu = true;
            if constexpr (std::is_same_v<DataType, std::int8_t>) {
                LOG_KNOWHERE_ERROR_ << "VAMANA+HNSW does not support INT8 data.";
                return Status::invalid_binary_set;
            } else {
                BinaryPtr binary = nullptr;
                hnswlib::SpaceInterface<float>* space = nullptr;
                hnsw_index_.reset(new (std::nothrow) hnswlib::HierarchicalNSW<DataType, float, hnswlib::None>(space));
                try {
                    if ((binary = binset.GetByName(std::string(this->Type()))) != nullptr) {
                        std::stringbuf buf;
                        buf.sputn((char*)binary->data.get(), binary->size);
                        std::istream stream(&buf);
                        // Use Vamana-specific GPU format loader
                        hnsw_index_->loadIndexFromVamanaGpuFormat(stream);
                    } else {
                        LOG_KNOWHERE_ERROR_ << "Invalid binary set.";
                        return Status::invalid_binary_set;
                    }
                    hnsw_index_->base_layer_only = true;

                    // export graph if enabled
                    auto base_cfg = static_cast<const knowhere::BaseConfig&>(vamana_cfg);
                    if (base_cfg.enable_export.has_value() && base_cfg.enable_export.value() &&
                        base_cfg.index_prefix.has_value()) {
                        hnsw_index_->exportGraph(base_cfg.index_prefix.value());
                    }
                } catch (std::exception& e) {
                    LOG_KNOWHERE_WARNING_ << "hnsw inner error: " << e.what();
                    return Status::hnsw_inner_error;
                }
                return Status::success;
            }
        }

        return GpuCuvsVamanaIndexNode<DataType>::Deserialize(binset, std::move(cfg));
    }

    Status
    DeserializeFromFile(const std::string& filename, std::shared_ptr<Config>) override {
        return Status::not_implemented;
    }

 private:
    bool adapt_for_cpu = false;
    bool int8_build_mode_ = false;
    int64_t saved_rows_ = 0;
    int64_t saved_cols_ = 0;
    // Built & serialized during Train() under int8 build mode, consumed by Serialize().
    std::string int8_serialized_bytes_;
    std::unique_ptr<hnswlib::HierarchicalNSW<DataType, float, hnswlib::None>> hnsw_index_ = nullptr;
};

KNOWHERE_REGISTER_GLOBAL_WITH_THREAD_POOL(GPU_CUVS_VAMANA, GpuCuvsVamanaHybridIndexNode, fp32,
                                          knowhere::feature::GPU_ANN_FLOAT_INDEX,
                                          []() {
                                              int count;
                                              RAFT_CUDA_TRY(cudaGetDeviceCount(&count));
                                              return count * cuda_concurrent_size_per_device;
                                          }()

);
KNOWHERE_REGISTER_GLOBAL_WITH_THREAD_POOL(GPU_VAMANA, GpuCuvsVamanaHybridIndexNode, fp32,
                                          knowhere::feature::GPU_ANN_FLOAT_INDEX, []() {
                                              int count;
                                              RAFT_CUDA_TRY(cudaGetDeviceCount(&count));
                                              return count * cuda_concurrent_size_per_device;
                                          }());
// INT8 does not support adapt_for_cpu yet
KNOWHERE_REGISTER_GLOBAL_WITH_THREAD_POOL(GPU_CUVS_VAMANA, GpuCuvsVamanaIndexNode, int8,
                                          knowhere::feature::GPU | knowhere::feature::INT8, []() {
                                              int count;
                                              RAFT_CUDA_TRY(cudaGetDeviceCount(&count));
                                              return count * cuda_concurrent_size_per_device;
                                          }());
KNOWHERE_REGISTER_GLOBAL_WITH_THREAD_POOL(GPU_VAMANA, GpuCuvsVamanaIndexNode, int8,
                                          knowhere::feature::GPU | knowhere::feature::INT8, []() {
                                              int count;
                                              RAFT_CUDA_TRY(cudaGetDeviceCount(&count));
                                              return count * cuda_concurrent_size_per_device;
                                          }());
}  // namespace knowhere
