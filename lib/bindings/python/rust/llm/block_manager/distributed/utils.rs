// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub fn get_kvbm_leader_port() -> u16 {
    // Default if nothing valid is set
    const DEFAULT_PORT: u16 = 5555;

    match std::env::var("DYN_KVBM_LEADER_PORT") {
        Ok(raw) if !raw.trim().is_empty() => {
            // Try parse to u16
            match raw.trim().parse::<u16>() {
                Ok(port) if port > 0 => port,
                Ok(_) | Err(_) => {
                    tracing::error!(
                        "Invalid DYN_KVBM_LEADER_PORT `{}` – falling back to {}",
                        raw,
                        DEFAULT_PORT
                    );
                    DEFAULT_PORT
                }
            }
        }
        _ => DEFAULT_PORT,
    }
}
