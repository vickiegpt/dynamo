// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Module to integration with PyTorch and libtorch
//!
//! Primarily to enable the use of the same CUDA context, stream and events created by
//! PyTorch and use them within Dynamo.
