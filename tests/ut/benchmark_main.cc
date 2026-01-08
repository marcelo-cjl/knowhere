// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/task.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/log.h"

// ============================================================================
// Dataset configuration
// ============================================================================
struct DatasetConfig {
    std::string name;
    std::string dir;
    std::string metric;

    std::string GetBasePath() const {
        return dir + "/" + name + ".fbin";
    }

    std::string GetQueryPath() const {
        return dir + "/" + name + "_query.fbin";
    }

    std::string GetTruthPath(int64_t topk, const std::string& filter_sig = "0.00") const {
        // First try exact match
        std::string exact_path = dir + "/" + name + "_query_" + metric + "_" + filter_sig + "_" + std::to_string(topk) + ".truth";
        if (std::filesystem::exists(exact_path)) {
            return exact_path;
        }

        // Search for truth file with truth_k >= topk (like vecTool)
        std::string truth_prefix = name + "_query_" + metric + "_" + filter_sig + "_";
        int optimal_truth_k = INT_MAX;
        std::string best_path = exact_path;

        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            if (!entry.is_regular_file()) continue;
            std::string filename = entry.path().filename().string();

            auto prefix_pos = filename.find(truth_prefix);
            if (prefix_pos == std::string::npos) continue;

            auto suffix_pos = filename.find(".truth");
            if (suffix_pos == std::string::npos) continue;

            try {
                int truth_k = std::stoi(filename.substr(prefix_pos + truth_prefix.length(),
                                                        suffix_pos - prefix_pos - truth_prefix.length()));
                if (truth_k >= topk && truth_k < optimal_truth_k) {
                    optimal_truth_k = truth_k;
                    best_path = entry.path().string();
                }
            } catch (...) {
                continue;
            }
        }
        return best_path;
    }

    bool Exists() const {
        return std::filesystem::exists(GetBasePath()) && std::filesystem::exists(GetQueryPath());
    }

    std::string GetIndexPath(const std::string& index_type, const std::string& param_suffix,
                             bool adapt_for_cpu = false) const {
        std::string suffix = adapt_for_cpu ? "_cpu" : "";
        return dir + "/" + name + "_" + index_type + param_suffix + suffix + ".index";
    }
};

// Generate parameter suffix for index filename
// Note: refine_k is search-only parameter, not included in index filename
std::string
GetIndexParamSuffix(const std::string& index_type, int nn_descent_niter = 20, int vamana_iters = 10,
                    const std::string& sq_type = "sq4u", const std::string& refine_type = "fp16") {
    if (index_type == knowhere::IndexEnum::INDEX_HNSWLIB) {
        return "_M32_ef360";  // M=32, efConstruction=360
    } else if (index_type == knowhere::IndexEnum::INDEX_HNSW_SQ) {
        // HNSW_SQ: sq_type + refine_type (refine_k is search-only, not in filename)
        return "_M32_" + sq_type + "_" + refine_type;
    } else if (index_type == knowhere::IndexEnum::INDEX_GPU_CAGRA) {
        return "_D64_iter" + std::to_string(nn_descent_niter);  // graph_degree=64, nn_descent_niter
    } else if (index_type == knowhere::IndexEnum::INDEX_GPU_VAMANA) {
        return "_D64_iter" + std::to_string(vamana_iters);  // graph_degree=64, vamana_iters
    } else if (index_type == knowhere::IndexEnum::INDEX_FAISS_IVFFLAT ||
               index_type == knowhere::IndexEnum::INDEX_FAISS_IVFFLAT_CC) {
        return "_nlist128";
    } else if (index_type == knowhere::IndexEnum::INDEX_FAISS_IVFSQ8) {
        return "_nlist128";
    } else if (index_type == knowhere::IndexEnum::INDEX_FAISS_IVFPQ) {
        return "_nlist128";
    }
    return "";
}

// All supported datasets
const std::map<std::string, DatasetConfig> ALL_DATASETS = {
    {"siftsmall", {"siftsmall", "/home/ubuntu/data/siftsmall", "L2"}},
    {"openaismall", {"openaismall", "/home/ubuntu/data/openaismall", "COSINE"}},
    {"gist", {"gist", "/home/ubuntu/data/gist", "L2"}},
    {"sift", {"sift", "/home/ubuntu/data/sift", "L2"}},
    {"glove25", {"glove25", "/home/ubuntu/data/glove25", "COSINE"}},
    {"glove50", {"glove50", "/home/ubuntu/data/glove50", "COSINE"}},
    {"glove100", {"glove100", "/home/ubuntu/data/glove100", "COSINE"}},
    {"glove200", {"glove200", "/home/ubuntu/data/glove200", "COSINE"}},
    {"cohere", {"cohere", "/home/ubuntu/data/cohere", "COSINE"}},
    {"openai", {"openai", "/home/ubuntu/data/openai", "COSINE"}},
    {"text2img", {"text2img", "/home/ubuntu/data/text2img", "IP"}},
    {"random_3072", {"random_3072", "/home/ubuntu/data/random_3072", "COSINE"}},
};

// ============================================================================
// Timer
// ============================================================================
class Timer {
 public:
    void Start() {
        start_ = std::chrono::steady_clock::now();
    }

    float Stop() {
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<float> elapsed = end - start_;
        return elapsed.count();
    }

 private:
    std::chrono::time_point<std::chrono::steady_clock> start_;
};

// ============================================================================
// Load fbin dataset
// ============================================================================
knowhere::DataSetPtr
LoadFbinDataset(const std::string& file_path) {
    std::ifstream reader(file_path, std::ios::binary);
    if (!reader) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    uint32_t nrows = 0, ncols = 0;
    reader.read(reinterpret_cast<char*>(&nrows), sizeof(uint32_t));
    reader.read(reinterpret_cast<char*>(&ncols), sizeof(uint32_t));

    float* data = new float[static_cast<size_t>(nrows) * ncols];
    reader.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(nrows) * ncols * sizeof(float));

    auto ds = knowhere::GenDataSet(nrows, ncols, data);
    ds->SetIsOwner(true);
    return ds;
}

// ============================================================================
// Split query dataset into batches
// ============================================================================
std::vector<knowhere::DataSetPtr>
SplitQueryDataset(const knowhere::DataSetPtr& query_ds, int64_t batch_size) {
    std::vector<knowhere::DataSetPtr> batches;
    int64_t nq = query_ds->GetRows();
    int64_t dim = query_ds->GetDim();
    const float* data = static_cast<const float*>(query_ds->GetTensor());

    if (batch_size <= 0 || batch_size >= nq) {
        batches.push_back(query_ds);
        return batches;
    }

    int64_t num_batches = (nq + batch_size - 1) / batch_size;
    for (int64_t i = 0; i < num_batches; ++i) {
        int64_t start = i * batch_size;
        int64_t end = std::min(start + batch_size, nq);
        int64_t size = end - start;

        float* batch_data = new float[size * dim];
        std::memcpy(batch_data, data + start * dim, size * dim * sizeof(float));

        auto batch_ds = knowhere::GenDataSet(size, dim, batch_data);
        batch_ds->SetIsOwner(true);
        batches.push_back(batch_ds);
    }
    return batches;
}

// ============================================================================
// Search result
// ============================================================================
class SearchResult {
 public:
    SearchResult() : nq_(0), topk_(0) {}

    void MergeFromDataSets(const std::vector<knowhere::DataSetPtr>& results) {
        if (results.empty()) return;

        int64_t total_nq = 0;
        int64_t topk = results[0]->GetDim();
        for (const auto& ds : results) {
            total_nq += ds->GetRows();
        }

        nq_ = total_nq;
        topk_ = topk;
        ids_.resize(static_cast<size_t>(nq_) * topk_);
        distances_.resize(static_cast<size_t>(nq_) * topk_);

        int64_t offset = 0;
        for (const auto& ds : results) {
            int64_t batch_nq = ds->GetRows();
            std::memcpy(ids_.data() + offset * topk_, ds->GetIds(), batch_nq * topk_ * sizeof(int64_t));
            std::memcpy(distances_.data() + offset * topk_, ds->GetDistance(), batch_nq * topk_ * sizeof(float));
            offset += batch_nq;
        }
    }

    void LoadFromFile(const std::string& file_path) {
        std::ifstream reader(file_path, std::ios::binary);
        if (!reader) {
            throw std::runtime_error("Failed to open file: " + file_path);
        }

        reader.read(reinterpret_cast<char*>(&nq_), sizeof(int32_t));
        reader.read(reinterpret_cast<char*>(&topk_), sizeof(int32_t));

        ids_.resize(static_cast<size_t>(nq_) * topk_);
        distances_.resize(static_cast<size_t>(nq_) * topk_);

        reader.read(reinterpret_cast<char*>(ids_.data()), ids_.size() * sizeof(int64_t));
        reader.read(reinterpret_cast<char*>(distances_.data()), distances_.size() * sizeof(float));
    }

    void SaveToFile(const std::string& file_path) const {
        std::ofstream writer(file_path, std::ios::binary);
        if (!writer) {
            throw std::runtime_error("Failed to create file: " + file_path);
        }

        writer.write(reinterpret_cast<const char*>(&nq_), sizeof(int32_t));
        writer.write(reinterpret_cast<const char*>(&topk_), sizeof(int32_t));
        writer.write(reinterpret_cast<const char*>(ids_.data()), ids_.size() * sizeof(int64_t));
        writer.write(reinterpret_cast<const char*>(distances_.data()), distances_.size() * sizeof(float));
    }

    int32_t GetNq() const { return nq_; }
    int32_t GetTopK() const { return topk_; }
    const int64_t* GetIds() const { return ids_.data(); }
    const float* GetDistances() const { return distances_.data(); }

 private:
    int32_t nq_;
    int32_t topk_;
    std::vector<int64_t> ids_;
    std::vector<float> distances_;
};

// ============================================================================
// Search statistics (like vecTool)
// ============================================================================
struct SearchStatistics {
    int64_t count = 0;
    float recall_sum = 0.0f, recall_min = 1.0f, recall_max = 0.0f;
    float dist_recall_sum = 0.0f, dist_recall_min = 1.0f, dist_recall_max = 0.0f;
    float ndcg_sum = 0.0f, ndcg_min = 1.0f, ndcg_max = 0.0f;
    float avg_disterr_sum = 0.0f, max_disterr = 0.0f;

    void Merge(float recall, float dist_recall, float ndcg, float avg_err, float max_err) {
        count++;
        recall_sum += recall;
        recall_min = std::min(recall_min, recall);
        recall_max = std::max(recall_max, recall);
        dist_recall_sum += dist_recall;
        dist_recall_min = std::min(dist_recall_min, dist_recall);
        dist_recall_max = std::max(dist_recall_max, dist_recall);
        ndcg_sum += ndcg;
        ndcg_min = std::min(ndcg_min, ndcg);
        ndcg_max = std::max(ndcg_max, ndcg);
        avg_disterr_sum += avg_err;
        max_disterr = std::max(max_disterr, max_err);
    }

    float AvgRecall() const { return count > 0 ? recall_sum / count : 0.0f; }
    float AvgDistRecall() const { return count > 0 ? dist_recall_sum / count : 0.0f; }
    float AvgNdcg() const { return count > 0 ? ndcg_sum / count : 0.0f; }
    float AvgDistErr() const { return count > 0 ? avg_disterr_sum / count : 0.0f; }
};

// ============================================================================
// Calculate detailed metrics (like vecTool)
// ============================================================================
SearchStatistics
CalculateMetrics(const SearchResult& ground_truth, const SearchResult& result, int64_t query_k) {
    SearchStatistics stats;
    auto nq = result.GetNq();
    auto res_k = result.GetTopK();
    auto gt_k = ground_truth.GetTopK();
    auto gt_ids = ground_truth.GetIds();
    auto res_ids = result.GetIds();
    auto gt_dists = ground_truth.GetDistances();
    auto res_dists = result.GetDistances();

    int64_t eval_k = std::min({query_k, static_cast<int64_t>(gt_k), static_cast<int64_t>(res_k)});

    for (int64_t i = 0; i < nq; ++i) {
        std::map<int64_t, int64_t> gt_index;  // id -> position
        int64_t valid_gt_k = 0;
        float ideal_dcg = 0.0f;

        for (int64_t j = 0; j < eval_k; ++j) {
            int64_t gt_id = gt_ids[i * gt_k + j];
            if (gt_id >= 0) {
                gt_index[gt_id] = j;
                ideal_dcg += 1.0f / std::log2(j + 2);
                valid_gt_k++;
            }
        }

        if (valid_gt_k == 0) {
            stats.Merge(1.0f, 1.0f, 1.0f, 0.0f, 0.0f);
            continue;
        }

        // Distance-based recall: check if result distance <= ground truth bound
        bool less_is_good = gt_dists[i * gt_k] < gt_dists[i * gt_k + valid_gt_k - 1];
        float gt_dist_bound = gt_dists[i * gt_k + valid_gt_k - 1];

        int64_t matches = 0, dist_matches = 0;
        float dcg = 0.0f, relative_error_sum = 0.0f, max_relative_error = 0.0f;

        for (int64_t j = 0; j < eval_k; ++j) {
            int64_t res_id = res_ids[i * res_k + j];
            float res_dist = res_dists[i * res_k + j];

            // Distance-based match
            bool dist_match = less_is_good ? (res_dist <= gt_dist_bound) : (res_dist >= gt_dist_bound);
            if (dist_match) dist_matches++;

            // ID-based match
            auto it = gt_index.find(res_id);
            if (it != gt_index.end()) {
                matches++;
                dcg += 1.0f / std::log2(it->second + 2);

                // Distance error
                float gt_dist = gt_dists[i * gt_k + it->second];
                if (std::abs(gt_dist) > 1e-9f) {
                    float rel_err = std::abs((gt_dist - res_dist) / gt_dist);
                    relative_error_sum += rel_err;
                    max_relative_error = std::max(max_relative_error, rel_err);
                }
            }
        }

        float recall = static_cast<float>(matches) / static_cast<float>(valid_gt_k);
        float dist_recall = static_cast<float>(dist_matches) / static_cast<float>(valid_gt_k);
        float ndcg = ideal_dcg > 0 ? dcg / ideal_dcg : 1.0f;
        float avg_err = matches > 0 ? relative_error_sum / matches : 0.0f;

        stats.Merge(recall, dist_recall, ndcg, avg_err, max_relative_error);
    }

    return stats;
}

// ============================================================================
// Get index configuration
// ============================================================================
knowhere::Json
GetIndexConfig(const std::string& index_type, int64_t dim, const std::string& metric, int64_t topk,
               bool adapt_for_cpu = false, int nn_descent_niter = 20, int vamana_iters = 1,
               const std::string& sq_type = "sq4u", float refine_k = 1.0f,
               const std::string& refine_type = "fp16", int ef = 100) {
    knowhere::Json json;
    json[knowhere::meta::DIM] = dim;
    json[knowhere::meta::METRIC_TYPE] = metric;
    json[knowhere::meta::TOPK] = topk;

    if (index_type == knowhere::IndexEnum::INDEX_HNSWLIB) {
        json[knowhere::indexparam::HNSW_M] = 32;  // level 0 neighbors = 2*M = 64
        json[knowhere::indexparam::EFCONSTRUCTION] = 360;
        json[knowhere::indexparam::EF] = ef;
    } else if (index_type == knowhere::IndexEnum::INDEX_HNSW_SQ) {
        // HNSW_SQ with SQ4U and refine (like HNSWSQ4U-16refine)
        json[knowhere::indexparam::HNSW_M] = 32;  // level 0 neighbors = 2*M = 64
        json[knowhere::indexparam::EFCONSTRUCTION] = 360;
        json[knowhere::indexparam::EF] = ef;
        json[knowhere::indexparam::SQ_TYPE] = sq_type;           // "sq4u" for 4-bit uniform
        json[knowhere::indexparam::HNSW_REFINE] = true;          // enable refine
        json[knowhere::indexparam::HNSW_REFINE_K] = refine_k;    // refine_k multiplier (e.g., 16)
        json[knowhere::indexparam::HNSW_REFINE_TYPE] = refine_type;  // refine with fp32
    } else if (index_type == knowhere::IndexEnum::INDEX_GPU_CAGRA) {
        json[knowhere::indexparam::GRAPH_DEGREE] = 64;
        json[knowhere::indexparam::INTERMEDIATE_GRAPH_DEGREE] = 128;
        json[knowhere::indexparam::NN_DESCENT_NITER] = nn_descent_niter;
        json[knowhere::indexparam::ITOPK_SIZE] = 128;
        if (adapt_for_cpu) {
            json[knowhere::indexparam::ADAPT_FOR_CPU] = true;
            json[knowhere::indexparam::EF] = ef;  // HNSW ef for CPU search
        }
    } else if (index_type == knowhere::IndexEnum::INDEX_GPU_VAMANA) {
        json[knowhere::indexparam::GRAPH_DEGREE] = 64;
        json[knowhere::indexparam::VISITED_SIZE] = 128;
        json[knowhere::indexparam::VAMANA_ITERS] = vamana_iters;
        if (adapt_for_cpu) {
            json[knowhere::indexparam::ADAPT_FOR_CPU] = true;
            json[knowhere::indexparam::EF] = ef;  // HNSW ef for CPU search
        }
    }
    return json;
}

// ============================================================================
// Concurrent search
// ============================================================================
template <typename IndexType>
SearchResult
ConcurrentSearch(IndexType& idx, const std::vector<knowhere::DataSetPtr>& query_batches,
                 const knowhere::Json& conf, int search_times) {
    size_t num_batches = query_batches.size();
    std::vector<knowhere::DataSetPtr> batch_results(num_batches);

    for (int t = 0; t < search_times; ++t) {
        std::vector<std::function<void()>> tasks;
        tasks.reserve(num_batches);

        for (size_t i = 0; i < num_batches; ++i) {
            tasks.emplace_back([&, i]() {
                auto res = idx.Search(query_batches[i], conf, nullptr);
                if (res.has_value()) {
                    batch_results[i] = res.value();
                }
            });
        }
        knowhere::ExecOverBuildThreadPool(tasks);
    }

    SearchResult merged;
    merged.MergeFromDataSets(batch_results);
    return merged;
}

// ============================================================================
// Mode: truth - Generate ground truth using brute force
// ============================================================================
int
RunTruthMode(const DatasetConfig& dataset, int64_t topk) {
    LOG_KNOWHERE_INFO_ << "=== Generating Ground Truth ===";
    LOG_KNOWHERE_INFO_ << "dataset=" << dataset.name << " | topk=" << topk;

    auto base_ds = LoadFbinDataset(dataset.GetBasePath());
    auto query_ds = LoadFbinDataset(dataset.GetQueryPath());

    LOG_KNOWHERE_INFO_ << "nb=" << base_ds->GetRows() << " | nq=" << query_ds->GetRows()
                       << " | dim=" << base_ds->GetDim();

    knowhere::Json json;
    json[knowhere::meta::DIM] = base_ds->GetDim();
    json[knowhere::meta::METRIC_TYPE] = dataset.metric;
    json[knowhere::meta::TOPK] = topk;

    Timer timer;
    timer.Start();
    auto result = knowhere::BruteForce::Search<knowhere::fp32>(base_ds, query_ds, json, nullptr);
    float elapsed = timer.Stop();

    if (!result.has_value()) {
        LOG_KNOWHERE_ERROR_ << "BruteForce search failed";
        return 1;
    }

    SearchResult ground_truth;
    std::vector<knowhere::DataSetPtr> results = {result.value()};
    ground_truth.MergeFromDataSets(results);

    std::string truth_path = dataset.GetTruthPath(topk);
    ground_truth.SaveToFile(truth_path);

    LOG_KNOWHERE_INFO_ << "=== Truth Complete ===";
    LOG_KNOWHERE_INFO_ << "saved=" << truth_path << " | time=" << std::fixed << std::setprecision(4) << elapsed << "s";
    return 0;
}

// ============================================================================
// Mode: build - Build and save index
// ============================================================================
int
RunBuildMode(const DatasetConfig& dataset, const std::string& index_type, int64_t topk, bool adapt_for_cpu,
             int nn_descent_niter = 20, int vamana_iters = 1, bool force_rebuild = false,
             const std::string& sq_type = "sq4u", float refine_k = 1.0f,
             const std::string& refine_type = "fp16", int ef = 100) {
    // Check if index already exists
    std::string param_suffix = GetIndexParamSuffix(index_type, nn_descent_niter, vamana_iters, sq_type, refine_type);
    std::string index_path = dataset.GetIndexPath(index_type, param_suffix, adapt_for_cpu);

    if (!force_rebuild && std::filesystem::exists(index_path)) {
        LOG_KNOWHERE_INFO_ << "Index already exists: " << index_path;
        LOG_KNOWHERE_INFO_ << "Use --rebuild to force rebuild";
        return 0;
    }

    LOG_KNOWHERE_INFO_ << "=== Building Index ===";
    LOG_KNOWHERE_INFO_ << "dataset=" << dataset.name << " | index=" << index_type
                       << (adapt_for_cpu ? " (adapt_for_cpu)" : "");
    if (index_type == knowhere::IndexEnum::INDEX_HNSW_SQ) {
        LOG_KNOWHERE_INFO_ << "sq_type=" << sq_type << " | refine_type=" << refine_type;
    }

    auto base_ds = LoadFbinDataset(dataset.GetBasePath());
    int64_t dim = base_ds->GetDim();
    int64_t nb = base_ds->GetRows();

    LOG_KNOWHERE_INFO_ << "nb=" << nb << " | dim=" << dim;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto conf = GetIndexConfig(index_type, dim, dataset.metric, topk, adapt_for_cpu, nn_descent_niter, vamana_iters,
                               sq_type, refine_k, refine_type, ef);
    auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_type, version).value();

    Timer timer;
    timer.Start();
    auto status = idx.Build(base_ds, conf);
    float build_time = timer.Stop();

    if (status != knowhere::Status::success) {
        LOG_KNOWHERE_ERROR_ << "Build failed with status: " << static_cast<int>(status);
        return 1;
    }

    // Save index
    knowhere::BinarySet binset;
    idx.Serialize(binset);

    // Try possible keys (GPU_XXX, GPU_CUVS_XXX, and _cpu variants for adapt_for_cpu)
    std::vector<std::string> possible_keys = {index_type, "GPU_CUVS_" + index_type.substr(4)};
    if (adapt_for_cpu) {
        possible_keys.push_back("GPU_CUVS_" + index_type.substr(4) + "_cpu");
    }
    auto binary = binset.GetByNames(possible_keys);
    if (binary != nullptr) {
        std::ofstream ofs(index_path, std::ios::binary);
        ofs.write(reinterpret_cast<const char*>(binary->data.get()), binary->size);
        ofs.close();
    }

    LOG_KNOWHERE_INFO_ << "=== Build Complete ===";
    LOG_KNOWHERE_INFO_ << "saved=" << index_path << " | size=" << (binary ? binary->size : 0)
                       << " | time=" << std::fixed << std::setprecision(4) << build_time << "s";
    return 0;
}

// ============================================================================
// Mode: search - Load/build index and run search benchmark
// ============================================================================
int
RunSearchMode(const DatasetConfig& dataset, const std::string& index_type, int64_t topk,
              int search_times, int64_t batch_size, bool adapt_for_cpu, bool force_rebuild,
              int nn_descent_niter = 20, int vamana_iters = 1,
              const std::string& sq_type = "sq4u", float refine_k = 1.0f,
              const std::string& refine_type = "fp16", int ef = 100) {
    // Check ground truth first
    std::string truth_path = dataset.GetTruthPath(topk);
    if (!std::filesystem::exists(truth_path)) {
        LOG_KNOWHERE_ERROR_ << "Ground truth not found: " << truth_path;
        LOG_KNOWHERE_ERROR_ << "Please run: --mode truth --dataset " << dataset.name << " --topk " << topk;
        return 1;
    }

    LOG_KNOWHERE_INFO_ << "=== Benchmark Config ===";
    LOG_KNOWHERE_INFO_ << "dataset=" << dataset.name << " | index=" << index_type
                       << (adapt_for_cpu ? " (adapt_for_cpu)" : "");
    if (index_type == knowhere::IndexEnum::INDEX_HNSW_SQ) {
        LOG_KNOWHERE_INFO_ << "sq_type=" << sq_type << " | refine_type=" << refine_type << " | refine_k=" << refine_k;
    }
    LOG_KNOWHERE_INFO_ << "topk=" << topk << " | times=" << search_times << " | batch=" << batch_size << " | ef=" << ef;

    auto base_ds = LoadFbinDataset(dataset.GetBasePath());
    auto query_ds = LoadFbinDataset(dataset.GetQueryPath());
    int64_t dim = base_ds->GetDim();
    int64_t nq = query_ds->GetRows();
    int64_t nb = base_ds->GetRows();

    LOG_KNOWHERE_INFO_ << "nb=" << nb << " | nq=" << nq << " | dim=" << dim;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto query_batches = SplitQueryDataset(query_ds, batch_size);
    auto conf = GetIndexConfig(index_type, dim, dataset.metric, topk, adapt_for_cpu, nn_descent_niter, vamana_iters,
                               sq_type, refine_k, refine_type, ef);
    auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_type, version).value();

    std::string param_suffix = GetIndexParamSuffix(index_type, nn_descent_niter, vamana_iters, sq_type, refine_type);
    std::string index_path = dataset.GetIndexPath(index_type, param_suffix, adapt_for_cpu);
    bool index_loaded = false;
    float build_time = 0.0f;
    Timer timer;

    // Try to load existing index
    if (!force_rebuild && std::filesystem::exists(index_path)) {
        LOG_KNOWHERE_INFO_ << "Loading index from: " << index_path;
        timer.Start();

        knowhere::BinarySet binset;
        std::ifstream ifs(index_path, std::ios::binary);
        ifs.seekg(0, std::ios::end);
        size_t file_size = ifs.tellg();
        ifs.seekg(0, std::ios::beg);

        auto data = new uint8_t[file_size];
        ifs.read(reinterpret_cast<char*>(data), file_size);
        ifs.close();

        // Add with all possible keys so Deserialize can find it
        auto shared_data = std::shared_ptr<uint8_t[]>(data);
        binset.Append(index_type, shared_data, file_size);
        // Also add with GPU_CUVS_XXX key for indexes that use that naming
        if (index_type.substr(0, 4) == "GPU_") {
            binset.Append("GPU_CUVS_" + index_type.substr(4), shared_data, file_size);
            // For adapt_for_cpu mode, the index is serialized with _cpu suffix
            if (adapt_for_cpu) {
                binset.Append("GPU_CUVS_" + index_type.substr(4) + "_cpu", shared_data, file_size);
            }
        }
        auto status = idx.Deserialize(binset, conf);
        build_time = timer.Stop();

        if (status == knowhere::Status::success) {
            index_loaded = true;
            LOG_KNOWHERE_INFO_ << "Index loaded in " << std::fixed << std::setprecision(4) << build_time << "s";
        } else {
            LOG_KNOWHERE_ERROR_ << "Failed to load index: " << index_path;
            return 1;
        }
    }

    // Build if not loaded
    if (!index_loaded) {
        timer.Start();
        auto status = idx.Build(base_ds, conf);
        build_time = timer.Stop();

        if (status != knowhere::Status::success) {
            LOG_KNOWHERE_ERROR_ << "Build failed with status: " << static_cast<int>(status);
            return 1;
        }

        // Serialize and save index
        LOG_KNOWHERE_INFO_ << "Saving index to: " << index_path;
        knowhere::BinarySet binset;
        idx.Serialize(binset);

        // Try possible keys (GPU_XXX, GPU_CUVS_XXX, and _cpu variants for adapt_for_cpu)
        std::vector<std::string> possible_keys = {index_type, "GPU_CUVS_" + index_type.substr(4)};
        if (adapt_for_cpu) {
            possible_keys.push_back("GPU_CUVS_" + index_type.substr(4) + "_cpu");
        }
        auto binary = binset.GetByNames(possible_keys);
        if (binary != nullptr) {
            std::ofstream ofs(index_path, std::ios::binary);
            ofs.write(reinterpret_cast<const char*>(binary->data.get()), binary->size);
            ofs.close();
            LOG_KNOWHERE_INFO_ << "Index saved (" << binary->size << " bytes)";
        }

        // For adapt_for_cpu mode, need to re-deserialize to initialize HNSW index
        if (adapt_for_cpu) {
            LOG_KNOWHERE_INFO_ << "Re-deserializing for adapt_for_cpu mode...";
            idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_type, version).value();
            status = idx.Deserialize(binset, conf);
            if (status != knowhere::Status::success) {
                LOG_KNOWHERE_ERROR_ << "Deserialize failed with status: " << static_cast<int>(status);
                return 1;
            }
        }
    }

    // Load ground truth
    SearchResult ground_truth;
    ground_truth.LoadFromFile(truth_path);

    // Warmup
    {
        std::vector<std::function<void()>> warmup_tasks;
        for (size_t i = 0; i < query_batches.size(); ++i) {
            warmup_tasks.emplace_back([&, i]() { idx.Search(query_batches[i], conf, nullptr); });
        }
        knowhere::ExecOverBuildThreadPool(warmup_tasks);
    }

    // Search
    timer.Start();
    auto search_result = ConcurrentSearch(idx, query_batches, conf, search_times);
    float search_time = timer.Stop();

    // Calculate metrics
    float qps = search_time > 0 ? static_cast<float>(nq * search_times) / search_time : 0.0f;
    auto stats = CalculateMetrics(ground_truth, search_result, topk);

    // Print result
    LOG_KNOWHERE_INFO_ << "=== Benchmark Result ===";
    LOG_KNOWHERE_INFO_ << "index=" << index_type << " | repeat=" << search_times << " | nq=" << nq
                       << " | topk=" << topk << " | batch=" << batch_size;
    LOG_KNOWHERE_INFO_ << "build_time=" << std::fixed << std::setprecision(4) << build_time << "s | "
                       << "search_time=" << std::fixed << std::setprecision(4) << search_time << "s | "
                       << "qps=" << std::fixed << std::setprecision(2) << qps;
    LOG_KNOWHERE_INFO_ << "avg_recall=" << std::fixed << std::setprecision(4) << stats.AvgRecall() << " | "
                       << "min_recall=" << std::fixed << std::setprecision(4) << stats.recall_min << " | "
                       << "dist_recall=" << std::fixed << std::setprecision(4) << stats.AvgDistRecall();
    LOG_KNOWHERE_INFO_ << "avg_ndcg=" << std::fixed << std::setprecision(4) << stats.AvgNdcg() << " | "
                       << "min_ndcg=" << std::fixed << std::setprecision(4) << stats.ndcg_min;
    LOG_KNOWHERE_INFO_ << "avg_disterr=" << std::fixed << std::setprecision(6) << stats.AvgDistErr() << " | "
                       << "max_disterr=" << std::fixed << std::setprecision(6) << stats.max_disterr;

    return 0;
}

// ============================================================================
// Print usage
// ============================================================================
void
PrintUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --mode <mode>       Mode: search, truth, or build (default: search)\n"
              << "  --dataset <name>    Dataset name (default: siftsmall)\n"
              << "  --index <type>      Index type (default: HNSW)\n"
              << "  --topk <n>          Top-K value (default: 100)\n"
              << "  --times <n>         Search iterations (default: 10)\n"
              << "  --batch <n>         Query batch size (default: 10)\n"
              << "  --ef <n>            efSearch parameter for HNSW (default: 100)\n"
              << "  --adapt_for_cpu     Build on GPU, search on CPU (for GPU_CAGRA/GPU_VAMANA)\n"
              << "  --nn_descent_niter <n>  NN descent iterations for CAGRA (default: 20)\n"
              << "  --vamana_iters <n>  Vamana build iterations (default: 1)\n"
              << "  --sq_type <type>    SQ type for HNSW_SQ: sq4u, sq6, sq8, fp16, bf16 (default: sq4u)\n"
              << "  --refine_k <n>      Refine K multiplier for HNSW_SQ (default: 1.0)\n"
              << "  --refine_type <t>   Refine type for HNSW_SQ: sq4u, sq6, sq8, fp16, bf16, fp32 (default: fp16)\n"
              << "  --rebuild           Force rebuild index even if exists\n"
              << "  --help              Show this help\n"
              << "\nSupported datasets:\n  ";
    for (const auto& [name, _] : ALL_DATASETS) {
        std::cout << name << " ";
    }
    std::cout << "\n\nSupported index types:\n"
              << "  HNSWLIB_DEPRECATED, HNSW_SQ, GPU_CAGRA, GPU_VAMANA\n"
              << "\nExample (HNSW_SQ with SQ4U + 16x refine):\n"
              << "  " << prog << " --index HNSW_SQ --sq_type sq4u --refine_k 16 --refine_type fp32\n";
}

// ============================================================================
// Main
// ============================================================================
int
main(int argc, char** argv) {
    // Default parameters
    std::string mode = "search";
    std::string dataset_name = "siftsmall";
    std::string index_type = "HNSWLIB_DEPRECATED";
    int64_t topk = 100;
    int search_times = 10;
    int64_t batch_size = 100;
    int ef = 100;
    bool adapt_for_cpu = false;
    bool force_rebuild = false;
    int nn_descent_niter = 20;
    int vamana_iters = 1;
    // HNSW_SQ specific parameters
    std::string sq_type = "sq4u";
    float refine_k = 1.0f;
    std::string refine_type = "fp16";

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            return 0;
        } else if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (arg == "--dataset" && i + 1 < argc) {
            dataset_name = argv[++i];
        } else if (arg == "--index" && i + 1 < argc) {
            index_type = argv[++i];
        } else if (arg == "--topk" && i + 1 < argc) {
            topk = std::stoll(argv[++i]);
        } else if (arg == "--times" && i + 1 < argc) {
            search_times = std::stoi(argv[++i]);
        } else if (arg == "--batch" && i + 1 < argc) {
            batch_size = std::stoll(argv[++i]);
        } else if (arg == "--ef" && i + 1 < argc) {
            ef = std::stoi(argv[++i]);
        } else if (arg == "--adapt_for_cpu") {
            adapt_for_cpu = true;
        } else if (arg == "--nn_descent_niter" && i + 1 < argc) {
            nn_descent_niter = std::stoi(argv[++i]);
        } else if (arg == "--vamana_iters" && i + 1 < argc) {
            vamana_iters = std::stoi(argv[++i]);
        } else if (arg == "--sq_type" && i + 1 < argc) {
            sq_type = argv[++i];
        } else if (arg == "--refine_k" && i + 1 < argc) {
            refine_k = std::stof(argv[++i]);
        } else if (arg == "--refine_type" && i + 1 < argc) {
            refine_type = argv[++i];
        } else if (arg == "--rebuild") {
            force_rebuild = true;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            PrintUsage(argv[0]);
            return 1;
        }
    }

    // Find dataset
    auto it = ALL_DATASETS.find(dataset_name);
    if (it == ALL_DATASETS.end()) {
        std::cerr << "Dataset not found: " << dataset_name << "\n";
        return 1;
    }
    const auto& dataset = it->second;

    if (!dataset.Exists()) {
        std::cerr << "Dataset files not found: " << dataset.dir << "\n";
        return 1;
    }

    // Dispatch to mode-specific functions
    if (mode == "truth") {
        return RunTruthMode(dataset, topk);
    } else if (mode == "build") {
        return RunBuildMode(dataset, index_type, topk, adapt_for_cpu, nn_descent_niter, vamana_iters, force_rebuild,
                            sq_type, refine_k, refine_type, ef);
    } else {
        return RunSearchMode(dataset, index_type, topk, search_times, batch_size, adapt_for_cpu, force_rebuild,
                             nn_descent_niter, vamana_iters, sq_type, refine_k, refine_type, ef);
    }
}
