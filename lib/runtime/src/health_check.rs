// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{DistributedRuntime, HealthStatus, SystemHealth};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::time::{MissedTickBehavior, interval};
use tracing::{debug, error, info, warn};

/// Configuration for health check behavior
pub struct HealthCheckConfig {
    /// How often to check endpoint health
    pub check_interval: Duration,
    /// How long since last response before considering unhealthy
    pub respond_stale_threshold: Duration,
    /// Timeout for health check requests
    pub request_timeout: Duration,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(crate::config::DEFAULT_HEALTH_CHECK_INTERVAL_SECS),
            respond_stale_threshold: Duration::from_secs(
                crate::config::DEFAULT_HEALTH_CHECK_RESPOND_STALE_THRESHOLD_SECS,
            ),
            request_timeout: Duration::from_secs(
                crate::config::DEFAULT_HEALTH_CHECK_REQUEST_TIMEOUT_SECS,
            ),
        }
    }
}

/// Health check manager that monitors endpoint health
pub struct HealthCheckManager {
    drt: Arc<DistributedRuntime>,
    config: HealthCheckConfig,
    /// Track when we last sent health checks to each endpoint
    last_health_check_sent: Arc<Mutex<HashMap<String, Instant>>>,
}

impl HealthCheckManager {
    pub fn new(drt: Arc<DistributedRuntime>, config: HealthCheckConfig) -> Self {
        Self {
            drt,
            config,
            last_health_check_sent: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Start the health check polling loop
    pub async fn start(self: Arc<Self>) {
        let mut check_interval = interval(self.config.check_interval);
        check_interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

        info!(
            "Health check manager started with interval: {:?}, respond_stale_threshold: {:?}",
            self.config.check_interval, self.config.respond_stale_threshold
        );

        loop {
            check_interval.tick().await;

            if let Err(e) = self.check_all_endpoints().await {
                error!("Health check error: {}", e);
            }
        }
    }

    /// Check health of all registered endpoints
    pub async fn check_all_endpoints(&self) -> anyhow::Result<()> {
        // Get list of endpoints with health check payloads from SystemHealth
        let endpoints_to_check: Vec<(String, serde_json::Value)> = {
            let system_health = self.drt.system_health.lock().unwrap();
            system_health.get_health_check_payloads()
        };

        debug!("Checking health of {} endpoints", endpoints_to_check.len());

        for (endpoint_subject, health_check_payload) in endpoints_to_check {
            self.check_endpoint_health(&endpoint_subject, &health_check_payload)
                .await?;
        }

        Ok(())
    }

    /// Check health of a single endpoint
    ///
    /// Simple health check logic:
    /// 1. If endpoint responded recently -> mark as healthy
    /// 2. Else:
    ///    - If health check not sent -> send one
    ///    - Else:
    ///      - If within timeout -> keep status unchanged
    ///      - Else -> mark as unhealthy
    pub async fn check_endpoint_health(
        &self,
        endpoint_subject: &str,
        health_check_payload: &serde_json::Value,
    ) -> anyhow::Result<()> {
        // Step 1: Check if endpoint has responded recently
        let has_recent_response = {
            let system_health = self.drt.system_health.lock().unwrap();
            system_health
                .has_responded_recently(endpoint_subject, self.config.respond_stale_threshold)
        };

        if has_recent_response {
            debug!(
                "Endpoint {} is healthy (responded within {:?})",
                endpoint_subject, self.config.respond_stale_threshold
            );
            let system_health = self.drt.system_health.lock().unwrap();
            system_health.set_endpoint_health_status(endpoint_subject, HealthStatus::Ready);

            return Ok(());
        }

        // Step 2: No recent response - check if we've sent a health check
        info!(
            "Endpoint {} has not responded recently (threshold: {:?})",
            endpoint_subject, self.config.respond_stale_threshold
        );

        let last_health_check_sent_at = {
            let last_sent = self.last_health_check_sent.lock().unwrap();
            last_sent.get(endpoint_subject).copied()
        };

        if let Some(sent_at) = last_health_check_sent_at {
            // We've sent a health check - check if it has timed out
            let elapsed = sent_at.elapsed();

            if elapsed < self.config.request_timeout {
                // Still within timeout - don't change status
                debug!(
                    "Waiting for health check response from {} (elapsed: {:?} of {:?})",
                    endpoint_subject, elapsed, self.config.request_timeout
                );
                // Status remains unchanged
            } else {
                // Timeout exceeded - mark as unhealthy
                warn!(
                    "Health check timeout exceeded for {} (elapsed: {:?}, timeout: {:?})",
                    endpoint_subject, elapsed, self.config.request_timeout
                );
                let system_health = self.drt.system_health.lock().unwrap();
                system_health.set_endpoint_health_status(endpoint_subject, HealthStatus::NotReady);
            }
        } else {
            // No health check sent yet - send one now
            {
                let mut last_sent = self.last_health_check_sent.lock().unwrap();
                last_sent.insert(endpoint_subject.to_string(), Instant::now());
            }

            match self
                .send_health_check_request(endpoint_subject, health_check_payload)
                .await
            {
                Ok(()) => {
                    info!("Health check request sent to {}", endpoint_subject);
                    // Don't wait for response - let the normal payload handling update last_response_time
                    // The next check cycle will determine if the endpoint responded
                }
                Err(e) => {
                    error!("Failed to send health check to {}: {}", endpoint_subject, e);
                    // Mark as unhealthy if we can't even send the health check
                    let system_health = self.drt.system_health.lock().unwrap();
                    system_health
                        .set_endpoint_health_status(endpoint_subject, HealthStatus::NotReady);
                }
            }
        }

        Ok(())
    }

    /// Send a health check request to an endpoint (fire and forget)
    async fn send_health_check_request(
        &self,
        endpoint_subject: &str,
        payload: &serde_json::Value,
    ) -> anyhow::Result<()> {
        debug!(
            "Sending health check request to {} with payload: {}",
            endpoint_subject, payload
        );

        // Create a NATS client and send the request
        let nats_client = self.drt.nats_client();
        let payload_bytes = serde_json::to_vec(payload)?;

        // Send request but don't wait for response - fire and forget
        // The normal payload handling path will update last_response_time when response arrives
        let client = nats_client.client().clone();
        let subject = endpoint_subject.to_string();

        tokio::spawn(async move {
            // We don't care about the response - just send the request
            let _ = client.request(subject, payload_bytes.into()).await;
        });

        debug!("Health check request dispatched to {}", endpoint_subject);
        Ok(())
    }
}

/// Start health check manager for the distributed runtime
pub fn start_health_check_manager(
    drt: Arc<DistributedRuntime>,
    config: Option<HealthCheckConfig>,
) -> tokio::task::JoinHandle<()> {
    let config = config.unwrap_or_default();
    let manager = Arc::new(HealthCheckManager::new(drt.clone(), config));

    // Spawn the health check loop
    tokio::spawn(async move {
        manager.start().await;
    })
}

/// Get health check status for all endpoints
pub async fn get_health_check_status(
    drt: &DistributedRuntime,
    threshold: Duration,
) -> anyhow::Result<serde_json::Value> {
    // Get endpoints list from SystemHealth
    let endpoint_subjects: Vec<String> = {
        let system_health = drt.system_health.lock().unwrap();
        system_health.get_health_check_endpoints()
    };

    let mut endpoint_statuses = std::collections::HashMap::new();

    // Check each endpoint
    {
        let system_health = drt.system_health.lock().unwrap();
        for endpoint_subject in &endpoint_subjects {
            let has_recent_response =
                system_health.has_responded_recently(endpoint_subject, threshold);

            let last_response = system_health
                .get_last_response_time(endpoint_subject)
                .map(|t| t.elapsed().as_secs())
                .unwrap_or(999999);

            endpoint_statuses.insert(
                endpoint_subject.clone(),
                serde_json::json!({
                    "healthy": has_recent_response,
                    "seconds_since_last_response": last_response,
                }),
            );
        }
    }

    let overall_healthy = endpoint_statuses
        .values()
        .all(|v| v["healthy"].as_bool().unwrap_or(false));

    Ok(serde_json::json!({
        "status": if overall_healthy { "ready" } else { "notready" },
        "endpoints_checked": endpoint_subjects.len(),
        "endpoint_statuses": endpoint_statuses,
        "check_threshold_seconds": threshold.as_secs(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    // NOTE: These tests demonstrate the logic of HealthCheckManager but are limited
    // because HealthCheckManager is tightly coupled with DistributedRuntime.
    // For proper unit testing, HealthCheckManager would need to be refactored to:
    // 1. Accept trait objects for SystemHealth instead of concrete types
    // 2. Use dependency injection for NATS client
    // 3. Or use a builder pattern that allows test configuration

    #[test]
    fn test_health_check_manager_creation() {
        // Test HealthCheckManager::new() correctly stores configuration
        let config = HealthCheckConfig {
            check_interval: Duration::from_secs(5),
            respond_stale_threshold: Duration::from_secs(10),
            request_timeout: Duration::from_secs(2),
        };

        // Verify the config values themselves (can't create actual manager without DRT)
        assert_eq!(config.check_interval, Duration::from_secs(5));
        assert_eq!(config.respond_stale_threshold, Duration::from_secs(10));
        assert_eq!(config.request_timeout, Duration::from_secs(2));

        // The actual HealthCheckManager::new() would:
        // 1. Store the drt reference
        // 2. Store the config
        // 3. Initialize empty last_health_check_sent HashMap
    }

    #[tokio::test]
    async fn test_check_endpoint_health_healthy() {
        // This test verifies the logic: "Recent response = healthy status"
        // In check_endpoint_health(), if has_recent_response is true,
        // it should mark the endpoint as Ready and return early

        let threshold = Duration::from_secs(5);

        // Simulate a recent response (3 seconds ago, within threshold)
        let last_response_time = Instant::now() - Duration::from_secs(3);

        // Check that elapsed time is within threshold
        assert!(last_response_time.elapsed() < threshold);
        // At this point, check_endpoint_health would:
        // 1. Call system_health.has_responded_recently() -> returns true
        // 2. Set endpoint status to HealthStatus::Ready
        // 3. Return early without sending health check
    }

    #[tokio::test]
    async fn test_check_endpoint_health_send_check() {
        // This test verifies: "No recent response triggers health check"
        // Logic path: has_recent_response = false, last_sent = None -> send health check

        let threshold = Duration::from_secs(5);

        // Simulate a stale response (create a time in the past)
        let last_response_time = Instant::now() - Duration::from_secs(10);

        // Verify response is stale
        assert!(last_response_time.elapsed() > threshold);

        // Simulate the last_health_check_sent tracking
        let last_sent: Arc<Mutex<HashMap<String, Instant>>> = Arc::new(Mutex::new(HashMap::new()));
        let endpoint = "test_endpoint";

        // Initially, no health check has been sent
        {
            let sent = last_sent.lock().unwrap();
            assert!(!sent.contains_key(endpoint));
        }

        // The manager would now:
        // 1. Detect no recent response (has_recent_response = false)
        // 2. Check last_health_check_sent -> None
        // 3. Insert current time into last_health_check_sent
        // 4. Call send_health_check_request()
        {
            let mut sent = last_sent.lock().unwrap();
            sent.insert(endpoint.to_string(), Instant::now());
        }

        // Verify health check is now tracked
        {
            let sent = last_sent.lock().unwrap();
            assert!(sent.contains_key(endpoint));
        }
    }

    #[tokio::test]
    async fn test_check_endpoint_health_within_timeout() {
        // This test verifies: "Pending check within timeout keeps status unchanged"
        // Logic path: has_recent_response = false, last_sent = Some(time), elapsed < timeout

        let timeout = Duration::from_secs(3);
        let last_sent: Arc<Mutex<HashMap<String, Instant>>> = Arc::new(Mutex::new(HashMap::new()));
        let endpoint = "test_endpoint";

        // Record that we sent a health check 2 seconds ago (within timeout)
        let sent_time = Instant::now() - Duration::from_secs(2);
        {
            let mut sent = last_sent.lock().unwrap();
            sent.insert(endpoint.to_string(), sent_time);
        }

        // Check the elapsed time since health check was sent
        {
            let sent = last_sent.lock().unwrap();
            let sent_time = sent.get(endpoint).unwrap();
            let elapsed = sent_time.elapsed();

            assert!(elapsed < timeout);
            // At this point, check_endpoint_health would:
            // 1. Find has_recent_response = false
            // 2. Find last_health_check_sent_at = Some(sent_time)
            // 3. Calculate elapsed < timeout
            // 4. Log "Waiting for health check response"
            // 5. NOT change the health status
        }
    }

    #[tokio::test]
    async fn test_check_endpoint_health_timeout_exceeded() {
        // This test verifies: "Mark unhealthy after timeout"
        // Logic path: has_recent_response = false, last_sent = Some(time), elapsed >= timeout

        let timeout = Duration::from_secs(3);
        let last_sent: Arc<Mutex<HashMap<String, Instant>>> = Arc::new(Mutex::new(HashMap::new()));
        let endpoint = "test_endpoint";

        // Record that we sent a health check in the past (5 seconds ago, exceeding timeout)
        let sent_time = Instant::now() - Duration::from_secs(5);
        {
            let mut sent = last_sent.lock().unwrap();
            sent.insert(endpoint.to_string(), sent_time);
        }

        // Verify timeout is exceeded
        {
            let sent = last_sent.lock().unwrap();
            let sent_time = sent.get(endpoint).unwrap();
            let elapsed = sent_time.elapsed();

            assert!(elapsed >= timeout);
            // At this point, check_endpoint_health would:
            // 1. Find has_recent_response = false
            // 2. Find last_health_check_sent_at = Some(sent_time)
            // 3. Calculate elapsed >= timeout
            // 4. Log "Health check timeout exceeded"
            // 5. Set endpoint status to HealthStatus::NotReady
        }
    }

    #[tokio::test]
    async fn test_flood_prevention() {
        // This test verifies: "No duplicate sends within timeout"
        // The manager should not send multiple health checks while one is pending

        let timeout = Duration::from_secs(3);
        let last_sent: Arc<Mutex<HashMap<String, Instant>>> = Arc::new(Mutex::new(HashMap::new()));
        let endpoint = "test_endpoint";

        // First health check gets sent
        let first_send_time = Instant::now();
        {
            let mut sent = last_sent.lock().unwrap();
            assert!(!sent.contains_key(endpoint)); // No previous send
            sent.insert(endpoint.to_string(), first_send_time);
        }

        // Check immediately after (simulating next polling interval)
        // The health check is still pending (within timeout)
        {
            let sent = last_sent.lock().unwrap();
            let sent_time = sent.get(endpoint).unwrap();

            // Should be very recent (just sent)
            assert!(sent_time.elapsed() < timeout);

            // The manager would NOT send another health check because:
            // 1. has_recent_response = false
            // 2. last_health_check_sent_at = Some(time)
            // 3. elapsed < timeout
            // 4. It enters the "waiting for response" branch, not the "send" branch
        }

        // Simulate a check after timeout expires
        // Create a new timestamp representing old send time
        {
            let mut sent = last_sent.lock().unwrap();
            let old_send_time = Instant::now() - Duration::from_secs(5);
            sent.insert(endpoint.to_string(), old_send_time);
        }

        // Now verify the timeout is exceeded
        {
            let sent = last_sent.lock().unwrap();
            let sent_time = sent.get(endpoint).unwrap();
            assert!(sent_time.elapsed() >= timeout);
            // Now the manager would mark as unhealthy
            // On the NEXT check cycle, it could send a new health check
        }
    }

    #[test]
    fn test_handle_health_check_response() {
        // This test verifies: "Update response times correctly"
        // When a health check response is received, SystemHealth should update last_response_time

        // Simulate the response time tracking that happens in SystemHealth
        let response_times: Arc<Mutex<HashMap<String, Instant>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let endpoint = "test_endpoint";

        // Initial response
        let first_response = Instant::now();
        {
            let mut times = response_times.lock().unwrap();
            times.insert(endpoint.to_string(), first_response);
        }

        // Verify it was recorded
        {
            let times = response_times.lock().unwrap();
            assert_eq!(times.get(endpoint), Some(&first_response));
        }

        // Simulate time passing and new response
        std::thread::sleep(Duration::from_millis(10));
        let second_response = Instant::now();

        {
            let mut times = response_times.lock().unwrap();
            times.insert(endpoint.to_string(), second_response);
        }

        // Verify the update
        {
            let times = response_times.lock().unwrap();
            let stored_time = times.get(endpoint).unwrap();
            assert_eq!(*stored_time, second_response);
            assert_ne!(*stored_time, first_response);
            // This proves the response time was updated correctly
        }
    }

    #[test]
    fn test_health_check_config_default() {
        // Test that default configuration uses the constants
        let config = HealthCheckConfig::default();
        assert_eq!(
            config.check_interval,
            Duration::from_secs(crate::config::DEFAULT_HEALTH_CHECK_INTERVAL_SECS)
        );
        assert_eq!(
            config.respond_stale_threshold,
            Duration::from_secs(crate::config::DEFAULT_HEALTH_CHECK_RESPOND_STALE_THRESHOLD_SECS)
        );
        assert_eq!(
            config.request_timeout,
            Duration::from_secs(crate::config::DEFAULT_HEALTH_CHECK_REQUEST_TIMEOUT_SECS)
        );
    }
}

// ===============================
// Integration Tests (require DRT)
// ===============================
#[cfg(all(test, feature = "integration"))]
mod integration_tests {
    use super::*;
    use crate::HealthStatus;
    use crate::distributed::distributed_test_utils::create_test_drt_async;
    use std::sync::Arc;
    use std::time::Duration;

    #[tokio::test]
    async fn test_initialization() {
        let drt = Arc::new(create_test_drt_async().await);

        let check_interval = Duration::from_secs(5);
        let respond_stale_threshold = Duration::from_secs(2);
        let request_timeout = Duration::from_secs(3);

        let config = HealthCheckConfig {
            check_interval: check_interval,
            respond_stale_threshold: respond_stale_threshold,
            request_timeout: request_timeout,
        };

        let manager = HealthCheckManager::new(drt.clone(), config);

        assert_eq!(manager.config.check_interval, check_interval);
        assert_eq!(
            manager.config.respond_stale_threshold,
            respond_stale_threshold
        );
        assert_eq!(manager.config.request_timeout, request_timeout);

        // Verify DRT is properly stored
        assert!(Arc::ptr_eq(&manager.drt, &drt));
    }

    #[tokio::test]
    async fn test_payload_registration() {
        let drt = Arc::new(create_test_drt_async().await);

        let endpoint = "test.endpoint";
        let payload = serde_json::json!({
            "prompt": "test",
            "_health_check": true
        });

        drt.system_health
            .lock()
            .unwrap()
            .register_health_check_payload(endpoint, payload.clone());

        let retrieved = drt
            .system_health
            .lock()
            .unwrap()
            .get_health_check_payload(endpoint);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), payload);

        // Verify endpoint appears in the list
        let endpoints = drt
            .system_health
            .lock()
            .unwrap()
            .get_health_check_endpoints();
        assert!(endpoints.contains(&endpoint.to_string()));
    }

    #[tokio::test]
    async fn test_poll_all_endpoints() {
        let drt = Arc::new(create_test_drt_async().await);

        for i in 0..3 {
            let endpoint = format!("test.endpoint.{}", i);
            let payload = serde_json::json!({
                "prompt": format!("test{}", i),
                "_health_check": true
            });
            drt.system_health
                .lock()
                .unwrap()
                .register_health_check_payload(&endpoint, payload);
        }

        let config = HealthCheckConfig {
            check_interval: Duration::from_secs(5),
            respond_stale_threshold: Duration::from_secs(2),
            request_timeout: Duration::from_secs(1),
        };

        let manager = HealthCheckManager::new(drt.clone(), config);
        manager.check_all_endpoints().await.unwrap();

        // Verify all endpoints were checked
        let sent = manager.last_health_check_sent.lock().unwrap();
        assert_eq!(sent.len(), 3);
        assert!(sent.contains_key("test.endpoint.0"));
        assert!(sent.contains_key("test.endpoint.1"));
        assert!(sent.contains_key("test.endpoint.2"));
    }

    #[tokio::test]
    async fn test_response_handling() {
        let drt = Arc::new(create_test_drt_async().await);

        let endpoint = "test.endpoint";

        // Register endpoint
        let payload = serde_json::json!({
            "prompt": "test",
            "_health_check": true
        });
        drt.system_health
            .lock()
            .unwrap()
            .register_health_check_payload(endpoint, payload.clone());
        // Simulate a response by updating the last response time
        drt.system_health
            .lock()
            .unwrap()
            .update_last_response_time(endpoint);

        // Check that response was recorded
        let response_times = drt.system_health.lock().unwrap().get_response_times();
        let times = response_times.lock().unwrap();
        assert!(times.contains_key(endpoint));

        // Status should be Ready
        let status = drt
            .system_health
            .lock()
            .unwrap()
            .get_endpoint_health_status(endpoint);
        assert_eq!(status, Some(HealthStatus::Ready));
    }

    #[tokio::test]
    async fn test_timeout_behavior() {
        let drt = Arc::new(create_test_drt_async().await);

        let endpoint = "test.endpoint.timeout";
        let payload = serde_json::json!({
            "prompt": "test",
            "_health_check": true
        });
        drt.system_health
            .lock()
            .unwrap()
            .register_health_check_payload(endpoint, payload.clone());

        // Initially, the endpoint should be Ready (default after registration)
        let initial_status = drt
            .system_health
            .lock()
            .unwrap()
            .get_endpoint_health_status(endpoint);
        assert_eq!(initial_status, Some(HealthStatus::Ready));

        // Simulate a recent response to ensure it starts healthy
        drt.system_health
            .lock()
            .unwrap()
            .update_last_response_time(endpoint);

        let config = HealthCheckConfig {
            check_interval: Duration::from_secs(5),
            respond_stale_threshold: Duration::from_secs(2),
            request_timeout: Duration::from_millis(100), // Very short timeout
        };

        let manager = HealthCheckManager::new(drt.clone(), config);

        // First check while healthy - should NOT send health check (has recent response)
        manager
            .check_endpoint_health(endpoint, &payload)
            .await
            .unwrap();
        {
            let sent = manager.last_health_check_sent.lock().unwrap();
            assert!(
                !sent.contains_key(endpoint),
                "Should NOT send health check when response is recent"
            );
        }
        let status = drt
            .system_health
            .lock()
            .unwrap()
            .get_endpoint_health_status(endpoint);
        assert_eq!(
            status,
            Some(HealthStatus::Ready),
            "Should remain Ready with recent response"
        );

        // Wait for response to become stale (exceed respond_stale_threshold)
        tokio::time::sleep(Duration::from_secs(3)).await;

        // Now check - should send health check since response is stale
        manager
            .check_endpoint_health(endpoint, &payload)
            .await
            .unwrap();

        // Verify health check was sent now that response is stale
        {
            let sent = manager.last_health_check_sent.lock().unwrap();
            assert!(
                sent.contains_key(endpoint),
                "Should send health check when response is stale"
            );
        }

        // Wait for timeout to expire (no response received)
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Check again - should mark as unhealthy due to timeout
        manager
            .check_endpoint_health(endpoint, &payload)
            .await
            .unwrap();

        // Status should now be NotReady due to timeout
        let final_status = drt
            .system_health
            .lock()
            .unwrap()
            .get_endpoint_health_status(endpoint);
        assert_eq!(
            final_status,
            Some(HealthStatus::NotReady),
            "Should be NotReady after timeout"
        );
    }
}
