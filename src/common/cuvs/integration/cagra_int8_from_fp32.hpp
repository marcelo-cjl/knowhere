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
#ifndef CAGRA_INT8_FROM_FP32_HPP
#define CAGRA_INT8_FROM_FP32_HPP

#include <cstdint>
#include <ostream>

#include "common/cuvs/integration/cuvs_knowhere_config.hpp"

namespace cuvs_knowhere {

// Build a cuVS CAGRA graph with int8-quantized data derived from fp32 input,
// then wrap the graph into an fp32 CAGRA index and write the hnswlib format
// expected by hnswlib::HierarchicalNSW::loadIndexFromGpuFormat (fp32 dataset).
// The final trailing bool=false matches cuvs_knowhere_index::serialize_to_hnswlib.
//
// fp32_data: mutable pointer — for COSINE metric the function normalizes in place.
void
build_and_serialize_cagra_int8_fp32(std::ostream& os, const cuvs_knowhere_config& config, float* fp32_data,
                                    int64_t rows, int64_t cols);

}  // namespace cuvs_knowhere

#endif  // CAGRA_INT8_FROM_FP32_HPP
