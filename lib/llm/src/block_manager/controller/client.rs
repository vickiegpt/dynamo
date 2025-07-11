use super::*;

use anyhow::Result;
use dynamo_runtime::{component::Component, pipeline::PushRouter, protocols::annotated::Annotated};
use futures::StreamExt;
use serde::de::DeserializeOwned;
use serde_json::Value;

pub struct ControlClient {
    client: PushRouter<ControlMessage, Annotated<Value>>,
    instance_id: i64,
}

impl ControlClient {
    pub async fn new(component: Component, instance_id: i64) -> Result<Self> {
        let client = component.endpoint("controller").client().await?;
        client.wait_for_instances().await?;
        let client =
            PushRouter::<ControlMessage, Annotated<Value>>::from_client(client, Default::default())
                .await?;

        Ok(Self {
            client,
            instance_id,
        })
    }

    pub async fn status(&self, cache_level: CacheLevel) -> Result<PoolStatus> {
        self.execute::<PoolStatus>(ControlMessage::Status(cache_level))
            .await
    }

    pub async fn reset_pool(&self, cache_level: CacheLevel) -> Result<()> {
        self.execute::<()>(ControlMessage::Reset(ResetRequest {
            cache_level,
            sequence_hashes: None,
        }))
        .await
    }

    // pub async fn reset_all_pools(&self) -> Result<()> {}

    // pub async fn reset_pool(&self, cache_level: CacheLevel) -> Result<()> {}

    // pub async fn reset_blocks(&self, sequence_hashes: Vec<SequenceHash>) -> Result<()> {}

    // pub async fn reset_blocks_by_cache_level(
    //     &self,
    //     cache_levels: Vec<CacheLevel>,
    //     sequence_hashes: Vec<SequenceHash>,
    // ) -> Result<()> {
    // }

    async fn execute<T: DeserializeOwned>(&self, message: ControlMessage) -> Result<T> {
        let mut stream = self.client.direct(message.into(), self.instance_id).await?;
        let resp = stream
            .next()
            .await
            .ok_or(anyhow::anyhow!("No response from controller"))?;
        let data = resp
            .into_result()?
            .ok_or(anyhow::anyhow!("No response from controller"))?;
        let result: T = serde_json::from_value(data)?;
        let next = stream.next().await;
        assert!(next.is_none());
        Ok(result)
    }
}
