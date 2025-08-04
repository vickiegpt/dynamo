// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Module to record the data passed into the leader portion of the vLLM connector API.
//!
//! This data will be captured and used to replay the events for unit and integration tests.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action {
    // create slot
    GetNumNewMatchedTokens(GetNumNewMatchedTokensInput, GetNumNewMatchedTokensOutput),
    UpdateStateAfterAlloc(UpdateStateAfterAllocInput),

    // make scheudler output serde
    // ConnectorMetadata should already be serde Serialize + Deserialize
    BuildConnectorMeta(SchedulerOutput, ConnectorMetadata),
    RequestFinished(RequestFinishedInput, RequestFinishedOutput),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetNumNewMatchedTokensInput {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetNumNewMatchedTokensOutput {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateStateAfterAllocInput {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestFinishedInput {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestFinishedOutput {}

pub trait LeaderInterface {
    fn create_slot(&self, input: CreateSlotInput) -> CreateSlotOutput;
    fn get_num_new_matched_tokens(
        &self,
        input: GetNumNewMatchedTokensInput,
    ) -> GetNumNewMatchedTokensOutput;
    fn update_state_after_alloc(&self, input: UpdateStateAfterAllocInput);
    fn build_connector_meta(&self, input: SchedulerOutput) -> ConnectorMetadata;
    fn request_finished(&self, input: RequestFinishedInput) -> RequestFinishedOutput;
}

// thoughts
// - create a trait for the leader interface
// - have the leader impl - the rust only bits that separate the bindigns and the impl implement the trait
// - have the recorder implment the trait taking an impl of the trait as a parameter

pub struct RecorderClient {
    tx: mpsc::Sender<Action>,
    leader: Box<dyn LeaderInterface>,
    // has the rx end of the mpsc channel
    // - serializes the bits to a jsonl file
    // background_task handle: CriticalTaskHandle
}

impl LeaderInterface for RecorderClient {
    // capture input and output wrapping the actual leader call
    fn get_num_new_matched_tokens(
        &self,
        input: GetNumNewMatchedTokensInput,
    ) -> GetNumNewMatchedTokensOutput {
        let input_copy = input.clone();
        let output = self.leader.get_num_new_matched_tokens(input);
        self.tx
            .send(Action::GetNumNewMatchedTokens(input_copy, output))
            .await
            .unwrap();
        output
    }
}

// dynamically wrap the leader with the recorder if an env if flipped.
// they are both valid Box<dyn LeaderInterface>, so the python bindings will not know the difference.
