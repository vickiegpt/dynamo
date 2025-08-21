use dynamo_runtime::{metrics::MetricsRegistry, component::Namespace};
use prometheus::IntCounter;


#[derive(Clone, Debug)]
pub struct KvbmMetrics {
    pub offload_requests: IntCounter,
}

impl KvbmMetrics {
    pub fn new(ns: Namespace) -> Self {
        let offload_requests = ns.create_intcounter("kvbm_connector_slot_manager_component", "The component for the kvbm connector slot manager", &[]).unwrap();
        Self { offload_requests }
    }
}
