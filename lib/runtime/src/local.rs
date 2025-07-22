// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The [`LocalRuntime`] module
//!

use super::{
    engine::AnyAsyncEngine, entity::descriptor::EndpointDescriptor, LocalRuntime, Runtime,
};

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

impl LocalRuntime {
    /// Create a new [LocalRuntime] with the given [Runtime].
    pub fn new(runtime: Runtime) -> anyhow::Result<LocalRuntime> {
        Ok(LocalRuntime {
            runtime,
            endpoint_engines: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Get an [AnyAsyncEngine] for an [EndpointDescriptor] if it exists, otherwise return None.
    pub fn get_engine(&self, descriptor: &EndpointDescriptor) -> Option<Arc<dyn AnyAsyncEngine>> {
        let engines = self.endpoint_engines.lock().unwrap();
        engines.get(descriptor).cloned()
    }

    /// Register an [AnyAsyncEngine] to an [EndpointDescriptor].  If an engine is already
    /// registered for the descriptor an error is returned.
    pub(crate) fn register_engine(
        &self,
        descriptor: EndpointDescriptor,
        engine: Arc<dyn AnyAsyncEngine>,
    ) -> anyhow::Result<()> {
        let mut engines = self.endpoint_engines.lock().unwrap();

        if engines.contains_key(&descriptor) {
            return Err(anyhow::anyhow!("Engine already registered for descriptor"));
        }

        engines.insert(descriptor, engine);

        Ok(())
    }
}
