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
#ifndef VAMANA_INT8_FROM_FP32_HPP
#define VAMANA_INT8_FROM_FP32_HPP

#include <cstdint>
#include <ostream>

#include "common/cuvs/integration/cuvs_knowhere_config.hpp"

namespace cuvs_knowhere {

// Build a cuVS vamana graph using int8-quantized data derived from the fp32
// input, then write the serialization format expected by
// hnswlib::HierarchicalNSW::loadIndexFromVamanaGpuFormat. The serialized
// dataset portion is the original fp32 data (normalized if metric is COSINE),
// so CPU search via hnswlib uses full fp32 precision.
//
// fp32_data: mutable pointer — for COSINE metric the function normalizes in place.
void
build_and_serialize_vamana_int8_fp32(std::ostream& os, const cuvs_knowhere_config& config, float* fp32_data,
                                     int64_t rows, int64_t cols);

}  // namespace cuvs_knowhere

#endif  // VAMANA_INT8_FROM_FP32_HPP
