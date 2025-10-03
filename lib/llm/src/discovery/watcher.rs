// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use tokio::sync::mpsc::Sender;

use anyhow::Context as _;
use tokio::sync::{Notify, mpsc::Receiver};

use dynamo_runtime::{
    DistributedRuntime,
    pipeline::{
        ManyOut, Operator, RouterMode, SegmentSource, ServiceBackend, SingleIn, Source,
        network::egress::push_router::PushRouter,
    },
    protocols::{EndpointId, annotated::Annotated},
    transports::etcd::{KeyValue, WatchEvent},
};

use crate::{
    backend::Backend,
    entrypoint,
    kv_router::KvRouterConfig,
    model_card::{self, ModelDeploymentCard},
    model_type::{ModelInput, ModelType},
    preprocessor::{OpenAIPreprocessor, PreprocessedEmbeddingRequest, prompt::PromptFormatter},
    protocols::{
        common::llm_backend::EmbeddingsEngineOutput,
        openai::{
            chat_completions::{
                NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
            },
            completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
            embeddings::{NvCreateEmbeddingRequest, NvCreateEmbeddingResponse},
        },
        tensor::{NvCreateTensorRequest, NvCreateTensorResponse},
    },
};

use super::ModelManager;
use crate::namespace::is_global_namespace;

#[derive(Debug, Clone)]
pub enum ModelUpdate {
    Added(ModelDeploymentCard),
    Removed(ModelDeploymentCard),
}

pub struct ModelWatcher {
    manager: Arc<ModelManager>,
    drt: DistributedRuntime,
    router_mode: RouterMode,
    notify_on_model: Notify,
    model_update_tx: Option<Sender<ModelUpdate>>,
    kv_router_config: Option<KvRouterConfig>,
    busy_threshold: Option<f64>,
}

const ALL_MODEL_TYPES: &[ModelType] = &[
    ModelType::Chat,
    ModelType::Completions,
    ModelType::Embedding,
    ModelType::TensorBased,
];

impl ModelWatcher {
    pub fn new(
        runtime: DistributedRuntime,
        model_manager: Arc<ModelManager>,
        router_mode: RouterMode,
        kv_router_config: Option<KvRouterConfig>,
        busy_threshold: Option<f64>,
    ) -> ModelWatcher {
        Self {
            manager: model_manager,
            drt: runtime,
            router_mode,
            notify_on_model: Notify::new(),
            model_update_tx: None,
            kv_router_config,
            busy_threshold,
        }
    }

    pub fn set_notify_on_model_update(&mut self, tx: Sender<ModelUpdate>) {
        self.model_update_tx = Some(tx);
    }

    /// Wait until we have at least one chat completions model and return it's name.
    pub async fn wait_for_chat_model(&self) -> String {
        // Loop in case it gets added and immediately deleted
        loop {
            if let Some(model_name) = self.manager.list_chat_completions_models().first() {
                return model_name.to_owned();
            }
            self.notify_on_model.notified().await
        }
    }

    /// Common watch logic with optional namespace filtering
    pub async fn watch(&self, mut events_rx: Receiver<WatchEvent>, target_namespace: Option<&str>) {
        let global_namespace = target_namespace.is_none_or(is_global_namespace);

        while let Some(event) = events_rx.recv().await {
            match event {
                WatchEvent::Put(kv) => {
                    let mut card = match serde_json::from_slice::<ModelDeploymentCard>(kv.value()) {
                        Ok(card) => card,
                        Err(err) => {
                            match kv.value_str() {
                                Ok(value) => {
                                    tracing::error!(%err, value, "Invalid JSON in model card")
                                }
                                Err(value_str_err) => {
                                    tracing::error!(original_error = %err, %value_str_err, "Invalid UTF-8 string in model card, expected JSON")
                                }
                            }
                            continue;
                        }
                    };
                    let key = match kv.key_str() {
                        Ok(k) => k,
                        Err(err) => {
                            tracing::error!(%err, ?kv, "Invalid UTF-8 string in model card key, skipping");
                            continue;
                        }
                    };
                    let endpoint_id = match etcd_key_to_endpoint_id(key) {
                        Ok(eid) => eid,
                        Err(err) => {
                            tracing::error!(%key, model_name = card.name(), %err, "Failed extracting EndpointId from key. Ignoring instance.");
                            continue;
                        }
                    };

                    // Filter by namespace if target_namespace is specified
                    if !global_namespace
                        && let Some(target_ns) = target_namespace
                        && endpoint_id.namespace != target_ns
                    {
                        tracing::debug!(
                            model_namespace = endpoint_id.namespace,
                            target_namespace = target_ns,
                            model_name = card.name(),
                            "Skipping model from different namespace"
                        );
                        continue;
                    }

                    match self.handle_put(key, &endpoint_id, &mut card).await {
                        Ok(()) => {
                            tracing::info!(
                                model_name = card.name(),
                                namespace = endpoint_id.namespace,
                                "added model"
                            );
                            self.notify_on_model.notify_waiters();
                        }
                        Err(err) => {
                            tracing::error!(
                                error = format!("{err:#}"),
                                "error adding model {} from namespace {}",
                                card.name(),
                                endpoint_id.namespace,
                            );
                        }
                    }
                }
                WatchEvent::Delete(kv) => match self
                    .handle_delete(&kv, target_namespace, global_namespace)
                    .await
                {
                    Ok(Some(model_name)) => {
                        tracing::info!(model_name, "removed model");
                    }
                    Ok(None) => {
                        // There are other instances running this model, nothing to do
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "error removing model");
                    }
                },
            }
        }
    }

    /// If the last instance running this model has gone delete it.
    /// Returns the name of the model we just deleted, if any.
    async fn handle_delete(
        &self,
        kv: &KeyValue,
        target_namespace: Option<&str>,
        is_global_namespace: bool,
    ) -> anyhow::Result<Option<String>> {
        let key = kv.key_str()?;
        let card = match self.manager.remove_model_card(key) {
            Some(card) => card,
            None => {
                anyhow::bail!("Missing ModelDeploymentCard for {key}");
            }
        };
        let model_name = card.name().to_string();
        let active_instances = self
            .cards_for_model(&model_name, target_namespace, is_global_namespace)
            .await
            .with_context(|| model_name.clone())?;
        if !active_instances.is_empty() {
            tracing::debug!(
                model_name,
                target_namespace = ?target_namespace,
                active_instance_count = active_instances.len(),
                "Model has other active instances, not removing"
            );
            return Ok(None);
        }

        // Ignore the errors because model could be either type
        let chat_model_remove_err = self.manager.remove_chat_completions_model(&model_name);
        let completions_model_remove_err = self.manager.remove_completions_model(&model_name);
        let embeddings_model_remove_err = self.manager.remove_embeddings_model(&model_name);
        let tensor_model_remove_err = self.manager.remove_tensor_model(&model_name);

        let mut chat_model_removed = false;
        let mut completions_model_removed = false;
        let mut embeddings_model_removed = false;
        let mut tensor_model_removed = false;

        if chat_model_remove_err.is_ok() && self.manager.list_chat_completions_models().is_empty() {
            chat_model_removed = true;
        }
        if completions_model_remove_err.is_ok() && self.manager.list_completions_models().is_empty()
        {
            completions_model_removed = true;
        }
        if embeddings_model_remove_err.is_ok() && self.manager.list_embeddings_models().is_empty() {
            embeddings_model_removed = true;
        }
        if tensor_model_remove_err.is_ok() && self.manager.list_tensor_models().is_empty() {
            tensor_model_removed = true;
        }

        if !chat_model_removed
            && !completions_model_removed
            && !embeddings_model_removed
            && !tensor_model_removed
        {
            tracing::debug!(
                "No updates to send for model {}: chat_model_removed: {}, completions_model_removed: {}, embeddings_model_removed: {}, tensor_model_removed: {}",
                model_name,
                chat_model_removed,
                completions_model_removed,
                embeddings_model_removed,
                tensor_model_removed
            );
        } else {
            for model_type in ALL_MODEL_TYPES {
                if ((chat_model_removed && *model_type == ModelType::Chat)
                    || (completions_model_removed && *model_type == ModelType::Completions)
                    || (embeddings_model_removed && *model_type == ModelType::Embedding)
                    || (tensor_model_removed && *model_type == ModelType::TensorBased))
                    && let Some(tx) = &self.model_update_tx
                {
                    tx.send(ModelUpdate::Removed(card.clone())).await.ok();
                }
            }
        }

        Ok(Some(model_name))
    }

    // Handles a PUT event from etcd, this usually means adding a new model to the list of served
    // models.
    async fn handle_put(
        &self,
        key: &str,
        endpoint_id: &EndpointId,
        card: &mut ModelDeploymentCard,
    ) -> anyhow::Result<()> {
        card.move_from_nats(self.drt.nats_client()).await?;
        let component = self
            .drt
            .namespace(&endpoint_id.namespace)?
            .component(&endpoint_id.component)?;
        let client = component.endpoint(&endpoint_id.name).client().await?;
        tracing::debug!(model_name = card.name(), "adding model");
        self.manager.save_model_card(key, card.clone());

        if self.manager.has_model_any(card.name()) {
            tracing::trace!(
                model_name = card.name(),
                namespace = endpoint_id.namespace,
                "New endpoint for existing model"
            );
            self.notify_on_model.notify_waiters();
            return Ok(());
        }

        if let Some(tx) = &self.model_update_tx {
            tx.send(ModelUpdate::Added(card.clone())).await.ok();
        }

        if card.model_input == ModelInput::Tokens
            && (card.model_type.supports_chat() || card.model_type.supports_completions())
        {
            // Case 1: Tokens + (Chat OR Completions OR Both)
            // A model that expects pre-processed requests meaning it's up to us whether we
            // handle Chat or Completions requests, so handle whatever the model supports.

            let kv_chooser = if self.router_mode == RouterMode::KV {
                Some(
                    self.manager
                        .kv_chooser_for(
                            card.name(),
                            &component,
                            card.kv_cache_block_size,
                            self.kv_router_config,
                        )
                        .await?,
                )
            } else {
                None
            };

            // This is expensive, we are loading ~10MiB JSON, so only do it once
            let tokenizer_hf = card.tokenizer_hf().context("tokenizer_hf")?;

            // Add chat engine only if the model supports chat
            if card.model_type.supports_chat() {
                let chat_engine = entrypoint::build_routed_pipeline::<
                    NvCreateChatCompletionRequest,
                    NvCreateChatCompletionStreamResponse,
                >(
                    card,
                    &client,
                    self.router_mode,
                    self.busy_threshold,
                    kv_chooser.clone(),
                    tokenizer_hf.clone(),
                )
                .await
                .context("build_routed_pipeline")?;
                self.manager
                    .add_chat_completions_model(card.name(), chat_engine)
                    .context("add_chat_completions_model")?;
                tracing::info!("Chat completions is ready");
            }

            // Add completions engine only if the model supports completions
            if card.model_type.supports_completions() {
                let formatter = PromptFormatter::no_op();
                let PromptFormatter::OAI(formatter) = formatter;
                let preprocessor = OpenAIPreprocessor::new_with_parts(
                    card.clone(),
                    formatter,
                    tokenizer_hf.clone(),
                )
                .context("OpenAIPreprocessor::new_with_parts")?;
                let completions_engine = entrypoint::build_routed_pipeline_with_preprocessor::<
                    NvCreateCompletionRequest,
                    NvCreateCompletionResponse,
                >(
                    card,
                    &client,
                    self.router_mode,
                    self.busy_threshold,
                    kv_chooser,
                    preprocessor,
                    tokenizer_hf,
                )
                .await
                .context("build_routed_pipeline_with_preprocessor")?;
                self.manager
                    .add_completions_model(card.name(), completions_engine)
                    .context("add_completions_model")?;
                tracing::info!("Completions is ready");
            }
        } else if card.model_input == ModelInput::Text && card.model_type.supports_chat() {
            // Case 3: Text + Chat
            let push_router = PushRouter::<
                NvCreateChatCompletionRequest,
                Annotated<NvCreateChatCompletionStreamResponse>,
            >::from_client_with_threshold(
                client, self.router_mode, self.busy_threshold
            )
            .await?;
            let engine = Arc::new(push_router);
            self.manager
                .add_chat_completions_model(card.name(), engine)?;
        } else if card.model_input == ModelInput::Text && card.model_type.supports_completions() {
            // Case 2: Text + Completions
            let push_router = PushRouter::<
                NvCreateCompletionRequest,
                Annotated<NvCreateCompletionResponse>,
            >::from_client_with_threshold(
                client, self.router_mode, self.busy_threshold
            )
            .await?;
            let engine = Arc::new(push_router);
            self.manager.add_completions_model(card.name(), engine)?;
        } else if card.model_input == ModelInput::Tokens && card.model_type.supports_embedding() {
            // Case 4: Tokens + Embeddings

            // Create preprocessing pipeline similar to Backend
            let frontend = SegmentSource::<
                SingleIn<NvCreateEmbeddingRequest>,
                ManyOut<Annotated<NvCreateEmbeddingResponse>>,
            >::new();

            let preprocessor = OpenAIPreprocessor::new(card.clone())?.into_operator();
            let backend = Backend::from_mdc(card).into_operator();

            let router = PushRouter::<
                PreprocessedEmbeddingRequest,
                Annotated<EmbeddingsEngineOutput>,
            >::from_client_with_threshold(
                client, self.router_mode, self.busy_threshold
            )
            .await?;

            // Note: Embeddings don't need KV routing complexity
            let service_backend = ServiceBackend::from_engine(Arc::new(router));

            // Link the pipeline: frontend -> preprocessor -> backend -> service_backend -> backend -> preprocessor -> frontend
            let embedding_engine = frontend
                .link(preprocessor.forward_edge())?
                .link(backend.forward_edge())?
                .link(service_backend)?
                .link(backend.backward_edge())?
                .link(preprocessor.backward_edge())?
                .link(frontend)?;

            self.manager
                .add_embeddings_model(card.name(), embedding_engine)?;
        } else if card.model_input == ModelInput::Tensor && card.model_type.supports_tensor() {
            // Case 5: Tensor + Tensor (non-LLM)
            let push_router = PushRouter::<
                NvCreateTensorRequest,
                Annotated<NvCreateTensorResponse>,
            >::from_client_with_threshold(
                client, self.router_mode, self.busy_threshold
            )
            .await?;
            let engine = Arc::new(push_router);
            self.manager.add_tensor_model(card.name(), engine)?;
        } else {
            // Reject unsupported combinations
            anyhow::bail!(
                "Unsupported model configuration: {} with {} input. Supported combinations: \
                Tokens+(Chat|Completions), Text+Chat, Text+Completions, Tokens+Embeddings, Tensor+TensorBased",
                card.model_type,
                card.model_input.as_str()
            );
        }

        Ok(())
    }

    /// All the registered ModelDeploymentCard, one per instance
    pub async fn all_cards(&self) -> anyhow::Result<Vec<ModelDeploymentCard>> {
        let Some(etcd_client) = self.drt.etcd_client() else {
            anyhow::bail!("all_cards: Missing etcd client");
        };
        let kvs = etcd_client.kv_get_prefix(model_card::ROOT_PATH).await?;
        let mut cards = Vec::with_capacity(kvs.len());
        for kv in kvs {
            let card = match serde_json::from_slice::<ModelDeploymentCard>(kv.value()) {
                Ok(card) => card,
                Err(err) => {
                    match kv.value_str() {
                        Ok(value) => {
                            tracing::error!(%err, value, "Invalid JSON in model card")
                        }
                        Err(value_str_err) => {
                            tracing::error!(original_error = %err, %value_str_err, "Invalid UTF-8 string in model card, expected JSON")
                        }
                    }
                    continue;
                }
            };
            cards.push(card);
        }
        Ok(cards)
    }

    pub async fn cards_for_model(
        &self,
        model_name: &str,
        _target_namespace: Option<&str>,
        _is_global_namespace: bool,
    ) -> anyhow::Result<Vec<ModelDeploymentCard>> {
        let mut all = self.all_cards().await?;
        all.retain(|card| {
            card.name() == model_name
            // TODO Restore namespace matching
            /*
            let matches_name = card.name() == model_name;
            let matches_namespace = match (is_global_namespace, target_namespace) {
                (true, _) => true,
                (false, None) => true,
                (false, Some(target_ns)) => entry.endpoint_id.namespace == target_ns,
            };
            matches_name && matches_namespace
            */
        });
        Ok(all)
    }
}

/// The ModelDeploymentCard is published in etcd with a key like "v1/mdc/dynamo/backend/generate/694d9981145a61ad".
/// Extract the EndpointId from that.
fn etcd_key_to_endpoint_id(s: &str) -> anyhow::Result<EndpointId> {
    let parts: Vec<&str> = s.split('/').collect();
    let start_idx = if !parts.is_empty() && parts[0] == "v1" {
        1
    } else {
        0
    };

    // Need at least prefix model_card::ROOT_PATH + 3 parts: namespace, component, name
    if parts.len() <= start_idx + 3 {
        anyhow::bail!("Invalid format: not enough path segments in {s}");
    }

    if parts.get(start_idx) != Some(&model_card::ROOT_PATH) {
        anyhow::bail!("Invalid format: expected model card ROOT_PATH segment in {s}");
    }

    Ok(EndpointId {
        namespace: parts[start_idx + 1].to_string(),
        component: parts[start_idx + 2].to_string(),
        name: parts[start_idx + 3].to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_etcd_key_to_endpoint_id() {
        let input = format!(
            "v1/{}/dynamo/backend/generate/694d9981145a61ad",
            model_card::ROOT_PATH
        );
        let result = etcd_key_to_endpoint_id(&input).unwrap();
        assert_eq!(result.namespace, "dynamo");
        assert_eq!(result.component, "backend");
        assert_eq!(result.name, "generate");

        let input = format!(
            "{}/dynamo/backend/generate/694d9981145a61ad",
            model_card::ROOT_PATH
        );
        let result = etcd_key_to_endpoint_id(&input).unwrap();
        assert_eq!(result.namespace, "dynamo");
        assert_eq!(result.component, "backend");
        assert_eq!(result.name, "generate");
    }
}
