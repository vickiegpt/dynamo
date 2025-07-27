use super::protocol::*;
use super::*;

use tokio::sync::mpsc;

pub enum SchedulingDecision {
    Execute,
    Cancel,
}

/// A client for the scheduler. Onetime use. Capture a clone per task.
#[derive(Clone)]
pub struct SchedulerClient {
    scheduler_tx: mpsc::Sender<TransferScheduleRequest>,
}

impl SchedulerClient {
    pub fn new(scheduler_tx: mpsc::Sender<TransferScheduleRequest>) -> Self {
        Self { scheduler_tx }
    }

    pub async fn schedule_transfer(self, request: LeaderTransferRequest) -> SchedulingDecision {
        let (response_tx, response_rx) = oneshot::channel();
        let tx = self.scheduler_tx.clone();

        let request = TransferScheduleRequest {
            leader_request: request,
            response_tx,
        };

        if tx.send(request).await.is_err() {
            tracing::warn!("connection to scheduler dropped; cancelling transfer");
            return SchedulingDecision::Cancel;
        }

        match response_rx.await {
            Ok(response) => response,
            Err(_) => {
                tracing::warn!("connection to scheduler dropped; cancelling transfer");
                SchedulingDecision::Cancel
            }
        }
    }
}

pub struct Scheduler {}

pub struct Request {
    pub leader_request: Option<LeaderTransferRequest>,
    pub worker_request: Option<WorkerTransferRequest>,
}
