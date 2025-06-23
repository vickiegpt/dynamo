// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    async_trait, Arc, AsyncEngine, AsyncEngineContextProvider, AsyncEngineInflightGuards, Data,
};

use prometheus::{IntCounter, IntGauge};

#[derive(Clone)]
pub struct EngineInstance<
    ReqT: Data,
    RespT: Data + AsyncEngineContextProvider + AsyncEngineInflightGuards,
    ErrT: Data,
> {
    name: String,
    engine: Arc<dyn AsyncEngine<ReqT, RespT, ErrT>>,

    /// Number of successful generate calls
    ///
    /// Note: This only tracks the result of the generate call, which might return a future
    /// or a stream; in those cases, the success of the future or stream is not tracked.
    generate_success: IntCounter,

    /// Number of failed generate calls
    ///
    /// Note: This only tracks the result of the generate call, which might return a future
    /// or a stream; in those cases, the error of the future or stream is not tracked.
    generate_error: IntCounter,

    /// Number of inflight responses
    inflight_gauge: IntGauge,
}

impl<
        ReqT: Data,
        RespT: Data + AsyncEngineContextProvider + AsyncEngineInflightGuards,
        ErrT: Data,
    > EngineInstance<ReqT, RespT, ErrT>
{
    pub fn new(name: String, engine: Arc<dyn AsyncEngine<ReqT, RespT, ErrT>>) -> Self {
        let sanitized_name = name.replace("-", "_");
        let generate_success = IntCounter::new(
            format!("{}_generate_success", sanitized_name),
            "generate_success",
        )
        .unwrap();
        let generate_error = IntCounter::new(
            format!("{}_generate_error", sanitized_name),
            "generate_error",
        )
        .unwrap();
        let inflight_gauge = IntGauge::new(
            format!("{}_inflight_gauge", sanitized_name),
            "inflight_gauge",
        )
        .unwrap();

        Self {
            name,
            engine,
            generate_success,
            generate_error,
            inflight_gauge,
        }
    }

    pub fn into_async_engine(self) -> Arc<dyn AsyncEngine<ReqT, RespT, ErrT>> {
        Arc::new(self)
    }
}

impl<ReqT, RespT, ErrT> std::fmt::Debug for EngineInstance<ReqT, RespT, ErrT>
where
    ReqT: Data,
    RespT: Data + AsyncEngineContextProvider + AsyncEngineInflightGuards,
    ErrT: Data,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EngineInstance<name: {}>", self.name)
    }
}

#[async_trait]
impl<
        ReqT: Data,
        RespT: Data + AsyncEngineContextProvider + AsyncEngineInflightGuards,
        ErrT: Data,
    > AsyncEngine<ReqT, RespT, ErrT> for EngineInstance<ReqT, RespT, ErrT>
{
    async fn generate(&self, request: ReqT) -> Result<RespT, ErrT> {
        let result = self.engine.generate(request).await;
        match result {
            Ok(mut resp) => {
                self.generate_success.inc();
                // Only create and increment the guard if the response supports it
                if resp.supports_inflight_guards() {
                    let guard = InflightGuard::with_value(self.inflight_gauge.clone(), 1);
                    // We know this should succeed since supports_inflight_guards() returned true
                    let _ = resp.try_add_inflight_guard(Box::new(guard));
                }
                Ok(resp)
            }
            Err(e) => {
                self.generate_error.inc();
                Err(e)
            }
        }
    }
}

/// A guard that increments the inflight gauge when created and decrements it when dropped.
/// This is used to track the number of inflight requests.
pub struct InflightGuard {
    gauge: IntGauge,
    value: i64,
}

impl InflightGuard {
    fn with_value(gauge: IntGauge, value: i64) -> Self {
        gauge.add(value);
        Self { gauge, value }
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.gauge.sub(self.value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{
        AsyncEngineContext, AsyncEngineContextProvider, AsyncEngineInflightGuards,
        InflightGuardNotSupported,
    };
    use std::{
        any::Any,
        collections::HashMap,
        sync::{
            atomic::{AtomicBool, AtomicU64, Ordering},
            Mutex,
        },
        time::Duration,
    };
    use tokio::time::sleep;

    /// Mock request type for testing
    #[derive(Debug, Clone, PartialEq)]
    struct TestRequest {
        id: u64,
        data: String,
    }

    /// Mock response type that implements required traits
    #[derive(Debug, Clone)]
    struct TestResponse {
        id: u64,
        result: String,
        guards: Arc<Mutex<Vec<Box<dyn Any + Send + Sync>>>>,
        context: Arc<MockAsyncEngineContext>,
    }

    impl TestResponse {
        fn new(id: u64, result: String) -> Self {
            Self {
                id,
                result,
                guards: Arc::new(Mutex::new(Vec::new())),
                context: Arc::new(MockAsyncEngineContext::new(id.to_string())),
            }
        }
    }

    impl AsyncEngineContextProvider for TestResponse {
        fn context(&self) -> Arc<dyn AsyncEngineContext> {
            self.context.clone()
        }
    }

    impl AsyncEngineInflightGuards for TestResponse {
        fn try_add_inflight_guard(&mut self, guard: Box<dyn Any + Send + Sync>) -> bool {
            self.guards.lock().unwrap().push(guard);
            true
        }

        fn supports_inflight_guards(&self) -> bool {
            true
        }
    }

    /// Mock error type for testing
    #[derive(Debug, Clone, PartialEq, thiserror::Error)]
    enum TestError {
        #[error("Processing failed: {message}")]
        ProcessingFailed { message: String },
        #[error("Network error: {code}")]
        NetworkError { code: u32 },
        #[error("Timeout occurred")]
        Timeout,
    }

    /// Mock AsyncEngineContext for testing
    #[derive(Debug)]
    struct MockAsyncEngineContext {
        id: String,
        stopped: AtomicBool,
        killed: AtomicBool,
    }

    impl MockAsyncEngineContext {
        fn new(id: String) -> Self {
            Self {
                id,
                stopped: AtomicBool::new(false),
                killed: AtomicBool::new(false),
            }
        }
    }

    #[async_trait]
    impl AsyncEngineContext for MockAsyncEngineContext {
        fn id(&self) -> &str {
            &self.id
        }

        fn is_stopped(&self) -> bool {
            self.stopped.load(Ordering::Relaxed)
        }

        fn is_killed(&self) -> bool {
            self.killed.load(Ordering::Relaxed)
        }

        async fn stopped(&self) {
            while !self.is_stopped() {
                sleep(Duration::from_millis(1)).await;
            }
        }

        async fn killed(&self) {
            while !self.is_killed() {
                sleep(Duration::from_millis(1)).await;
            }
        }

        fn stop_generating(&self) {
            self.stopped.store(true, Ordering::Relaxed);
        }

        fn stop(&self) {
            self.stop_generating();
        }

        fn kill(&self) {
            self.stopped.store(true, Ordering::Relaxed);
            self.killed.store(true, Ordering::Relaxed);
        }
    }

    /// Mock AsyncEngine implementation for testing
    #[derive(Debug)]
    struct MockAsyncEngine {
        name: String,
        success_responses: Arc<Mutex<HashMap<u64, TestResponse>>>,
        error_responses: Arc<Mutex<HashMap<u64, TestError>>>,
        call_count: Arc<AtomicU64>,
        delay: Option<Duration>,
    }

    impl MockAsyncEngine {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                success_responses: Arc::new(Mutex::new(HashMap::new())),
                error_responses: Arc::new(Mutex::new(HashMap::new())),
                call_count: Arc::new(AtomicU64::new(0)),
                delay: None,
            }
        }

        fn with_delay(mut self, delay: Duration) -> Self {
            self.delay = Some(delay);
            self
        }

        fn add_success_response(&self, id: u64, response: TestResponse) {
            self.success_responses.lock().unwrap().insert(id, response);
        }

        fn add_error_response(&self, id: u64, error: TestError) {
            self.error_responses.lock().unwrap().insert(id, error);
        }

        fn call_count(&self) -> u64 {
            self.call_count.load(Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl AsyncEngine<TestRequest, TestResponse, TestError> for MockAsyncEngine {
        async fn generate(&self, request: TestRequest) -> Result<TestResponse, TestError> {
            self.call_count.fetch_add(1, Ordering::Relaxed);

            if let Some(delay) = self.delay {
                sleep(delay).await;
            }

            // Check for predefined error responses first
            if let Some(error) = self.error_responses.lock().unwrap().get(&request.id) {
                return Err(error.clone());
            }

            // Check for predefined success responses
            if let Some(response) = self.success_responses.lock().unwrap().get(&request.id) {
                return Ok(response.clone());
            }

            // Default success response
            Ok(TestResponse::new(
                request.id,
                format!("Processed: {}", request.data),
            ))
        }
    }

    /// Helper function to create a test engine instance
    fn create_test_engine_instance(
        name: &str,
    ) -> EngineInstance<TestRequest, TestResponse, TestError> {
        let mock_engine = Arc::new(MockAsyncEngine::new(name));
        EngineInstance::new(name.to_string(), mock_engine)
    }

    /// Helper function to create a test engine instance with custom mock engine
    fn create_test_engine_instance_with_mock(
        name: &str,
        mock_engine: MockAsyncEngine,
    ) -> EngineInstance<TestRequest, TestResponse, TestError> {
        EngineInstance::new(name.to_string(), Arc::new(mock_engine))
    }

    /// Tests basic EngineInstance creation and Debug trait implementation.
    ///
    /// Validates that:
    /// - EngineInstance can be created with a name and mock engine
    /// - Debug formatting shows the expected format: "EngineInstance<name: {name}>"
    ///
    /// This is a simple smoke test ensuring the basic constructor and debug formatting work.
    #[tokio::test]
    async fn test_engine_instance_creation() {
        let instance = create_test_engine_instance("test-engine");

        // Verify the instance is created with correct name
        assert_eq!(
            format!("{:?}", instance),
            "EngineInstance<name: test-engine>"
        );
    }

    /// Tests successful request processing with metrics and inflight guard tracking.
    ///
    /// Validates that:
    /// - Successful requests increment the success counter
    /// - Error counter remains at 0
    /// - Inflight guards are properly added to successful responses
    /// - Guard increments the inflight gauge correctly
    ///
    /// This test ensures the happy path works correctly with proper RAII behavior.
    #[tokio::test]
    async fn test_successful_generate_call() {
        let instance = create_test_engine_instance("success-engine");
        let request = TestRequest {
            id: 1,
            data: "test data".to_string(),
        };

        let result = instance.generate(request.clone()).await;

        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.id, 1);
        assert_eq!(response.result, "Processed: test data");

        // Verify metrics were updated
        assert_eq!(instance.generate_success.get(), 1);
        assert_eq!(instance.generate_error.get(), 0);
        assert_eq!(instance.inflight_gauge.get(), 1);

        // Drop response to cleanup guard
        drop(response);
        assert_eq!(instance.inflight_gauge.get(), 0);
    }

    /// Tests error handling with metrics tracking and guard behavior.
    ///
    /// Validates that:
    /// - Failed requests increment the error counter
    /// - Success counter remains at 0
    /// - Correct error is returned from the underlying engine
    /// - No inflight guards are created for failed requests
    ///
    /// This test ensures error paths are handled correctly without leaking resources.
    #[tokio::test]
    async fn test_failed_generate_call() {
        let mock_engine = MockAsyncEngine::new("error-engine");
        let error = TestError::ProcessingFailed {
            message: "Test error".to_string(),
        };
        mock_engine.add_error_response(1, error.clone());

        let instance = create_test_engine_instance_with_mock("error-engine", mock_engine);
        let request = TestRequest {
            id: 1,
            data: "test data".to_string(),
        };

        let result = instance.generate(request).await;

        assert!(result.is_err());
        let returned_error = result.unwrap_err();
        assert_eq!(returned_error, error);

        // Verify metrics were updated
        assert_eq!(instance.generate_success.get(), 0);
        assert_eq!(instance.generate_error.get(), 1);
        assert_eq!(instance.inflight_gauge.get(), 0);
    }

    /// Tests proper cleanup of inflight guards when responses are dropped.
    ///
    /// Uses a barrier-based approach to ensure deterministic testing:
    /// 1. Spawns a task that captures a response with an active guard
    /// 2. Uses a barrier to synchronize when the guard is active
    /// 3. Validates the inflight gauge shows the active guard
    /// 4. Waits for the task to complete (dropping the response and guard)
    /// 5. Verifies the gauge returns to 0 after cleanup
    ///
    /// This test validates the RAII Drop implementation works correctly in async contexts.
    #[tokio::test]
    async fn test_inflight_guard_cleanup_on_drop() {
        use tokio::sync::Barrier;
        use tokio::time::{sleep, Duration};

        let instance = Arc::new(create_test_engine_instance("cleanup-test-engine"));
        let barrier = Arc::new(Barrier::new(2)); // Main task + spawned task

        let instance_clone = instance.clone();
        let barrier_clone = barrier.clone();

        let handle = tokio::spawn(async move {
            let request = TestRequest {
                id: 1,
                data: "test data".to_string(),
            };

            let result = instance_clone.generate(request).await;
            assert!(result.is_ok());

            let response = result.unwrap();

            // Signal that we have the response and guard is active
            barrier_clone.wait().await;

            // Return response to keep it alive until task completes
            response
        });

        // Wait for the spawned task to capture response
        barrier.wait().await;

        // Verify gauge is incremented while response is alive
        assert_eq!(instance.inflight_gauge.get(), 1);

        // Wait for task to complete (this drops the response and guard)
        let _response = handle.await.unwrap();

        // Explicitly drop the response
        drop(_response);

        // Give time for the Drop implementation to run
        sleep(Duration::from_millis(10)).await;

        // Now gauge should be decremented back to 0
        assert_eq!(instance.inflight_gauge.get(), 0);
    }

    /// Tests mixed success and error scenarios with proper inflight tracking.
    ///
    /// Uses barrier synchronization to test concurrent mixed requests:
    /// - Sets up predefined errors for specific request IDs (2 and 4)
    /// - Spawns 6 concurrent requests with barrier coordination
    /// - Validates that successes and errors are properly categorized
    /// - Ensures only successful responses have inflight guards
    /// - Verifies proper cleanup when successful responses are dropped
    ///
    /// This test validates behavior under realistic mixed-result scenarios.
    /// Uses barrier synchronization to test concurrent mixed requests:
    /// - Sets up predefined errors for specific request IDs (2 and 4)
    /// - Spawns 6 concurrent requests with barrier coordination
    /// - Validates that successes and errors are properly categorized
    /// - Ensures only successful responses have inflight guards
    /// - Verifies proper cleanup when successful responses are dropped
    ///
    /// This test validates behavior under realistic mixed-result scenarios.
    #[tokio::test]
    async fn test_mixed_success_and_error_requests() {
        use tokio::sync::Barrier;

        let mock_engine = MockAsyncEngine::new("mixed-engine");

        // Set up some requests to fail
        mock_engine.add_error_response(2, TestError::Timeout);
        mock_engine.add_error_response(4, TestError::NetworkError { code: 404 });

        let instance = Arc::new(create_test_engine_instance_with_mock(
            "mixed-engine",
            mock_engine,
        ));

        // Barrier to synchronize after all requests are captured
        let barrier = Arc::new(Barrier::new(7)); // 6 tasks + 1 main task
        let mut handles = Vec::new();
        let mut success_responses = Vec::new();

        // Spawn requests (some will succeed, some will fail)
        for i in 0..6 {
            let instance_clone = instance.clone();
            let barrier_clone = barrier.clone();

            let handle = tokio::spawn(async move {
                let request = TestRequest {
                    id: i,
                    data: format!("mixed data {}", i),
                };
                let result = instance_clone.generate(request).await;

                // Wait for all tasks to complete their requests
                barrier_clone.wait().await;

                (i, result)
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete their requests
        barrier.wait().await;

        // Now collect results and validate
        let mut success_count = 0;
        let mut error_count = 0;

        for handle in handles {
            let (id, result) = handle.await.unwrap();
            match result {
                Ok(response) => {
                    success_count += 1;
                    success_responses.push(response);
                    // IDs 2 and 4 should fail, others should succeed
                    assert!(id != 2 && id != 4, "Request {} should have failed", id);
                }
                Err(_) => {
                    error_count += 1;
                    // Only IDs 2 and 4 should fail
                    assert!(id == 2 || id == 4, "Request {} should have succeeded", id);
                }
            }
        }

        assert_eq!(success_count, 4);
        assert_eq!(error_count, 2);

        // Verify metrics
        assert_eq!(instance.generate_success.get(), 4);
        assert_eq!(instance.generate_error.get(), 2);

        // Verify inflight gauge shows only successful responses (with guards)
        assert_eq!(instance.inflight_gauge.get(), 4);

        // Drop successful responses to cleanup guards
        drop(success_responses);

        // Give a moment for cleanup
        tokio::task::yield_now().await;

        // Now gauge should be back to 0
        assert_eq!(instance.inflight_gauge.get(), 0);
    }

    /// Tests metrics isolation between separate EngineInstance objects.
    ///
    /// Uses barrier synchronization to test concurrent operations on different instances:
    /// - Creates two separate EngineInstance objects
    /// - Runs concurrent requests on both instances
    /// - Validates that each instance maintains its own metrics
    /// - Ensures inflight guards are properly isolated per instance
    /// - Verifies cleanup works independently for each instance
    ///
    /// This test ensures that multiple engine instances don't interfere with each other.
    #[tokio::test]
    async fn test_metrics_isolation_between_instances() {
        use tokio::sync::Barrier;

        let instance1 = Arc::new(create_test_engine_instance("engine-1"));
        let instance2 = Arc::new(create_test_engine_instance("engine-2"));

        let barrier = Arc::new(Barrier::new(3)); // 2 tasks + 1 main task

        let request = TestRequest {
            id: 1,
            data: "test".to_string(),
        };

        // Spawn concurrent requests on both instances
        let instance1_clone = instance1.clone();
        let barrier1 = barrier.clone();
        let request1 = request.clone();
        let handle1 = tokio::spawn(async move {
            let result = instance1_clone.generate(request1).await;
            barrier1.wait().await;
            result.unwrap()
        });

        let instance2_clone = instance2.clone();
        let barrier2 = barrier.clone();
        let request2 = request.clone();
        let handle2 = tokio::spawn(async move {
            let result = instance2_clone.generate(request2).await;
            barrier2.wait().await;
            result.unwrap()
        });

        // Wait for both tasks to complete their requests
        barrier.wait().await;

        // Verify metrics are isolated while responses are alive
        assert_eq!(instance1.generate_success.get(), 1);
        assert_eq!(instance1.generate_error.get(), 0);
        assert_eq!(instance1.inflight_gauge.get(), 1);

        assert_eq!(instance2.generate_success.get(), 1);
        assert_eq!(instance2.generate_error.get(), 0);
        assert_eq!(instance2.inflight_gauge.get(), 1);

        // Collect responses
        let response1 = handle1.await.unwrap();
        let response2 = handle2.await.unwrap();

        // Verify both instances still have their guards active
        assert_eq!(instance1.inflight_gauge.get(), 1);
        assert_eq!(instance2.inflight_gauge.get(), 1);

        // Drop responses to cleanup guards
        drop(response1);
        drop(response2);

        // Give a moment for cleanup
        tokio::task::yield_now().await;

        // Both gauges should be back to 0
        assert_eq!(instance1.inflight_gauge.get(), 0);
        assert_eq!(instance2.inflight_gauge.get(), 0);
    }

    /// Tests thread safety of EngineInstance cloning and concurrent usage.
    ///
    /// Uses barrier synchronization to test Arc-based cloning:
    /// - Creates an EngineInstance and clones it
    /// - Runs concurrent requests on both the original and cloned instances
    /// - Validates that both instances share the same underlying metrics (same Arc)
    /// - Ensures proper guard tracking across cloned instances
    /// - Verifies cleanup works correctly for shared instances
    ///
    /// This test validates that EngineInstance is properly thread-safe when cloned.
    #[tokio::test]
    async fn test_engine_instance_clone_safety() {
        use tokio::sync::Barrier;

        let instance = Arc::new(create_test_engine_instance("clone-test-engine"));
        let cloned_instance = instance.clone();

        let barrier = Arc::new(Barrier::new(3)); // 2 tasks + 1 main task

        let request = TestRequest {
            id: 1,
            data: "clone test".to_string(),
        };

        // Test that both original and cloned instances work concurrently
        let instance1 = instance.clone();
        let barrier1 = barrier.clone();
        let request1 = request.clone();
        let handle1 = tokio::spawn(async move {
            let result = instance1.generate(request1).await;
            barrier1.wait().await;
            result.unwrap()
        });

        let instance2 = cloned_instance.clone();
        let barrier2 = barrier.clone();
        let request2 = request.clone();
        let handle2 = tokio::spawn(async move {
            let result = instance2.generate(request2).await;
            barrier2.wait().await;
            result.unwrap()
        });

        // Wait for both tasks to complete
        barrier.wait().await;

        // Both should share the same metrics (since they're the same instance)
        assert_eq!(instance.generate_success.get(), 2);
        assert_eq!(instance.generate_error.get(), 0);
        assert_eq!(instance.inflight_gauge.get(), 2);

        // Collect responses
        let response1 = handle1.await.unwrap();
        let response2 = handle2.await.unwrap();

        assert!(response1.result.contains("clone test"));
        assert!(response2.result.contains("clone test"));

        // Drop responses to cleanup guards
        drop(response1);
        drop(response2);

        // Give a moment for cleanup
        tokio::task::yield_now().await;

        // Gauge should be back to 0
        assert_eq!(instance.inflight_gauge.get(), 0);
    }

    /// Tests graceful handling of inflight guard addition failures.
    ///
    /// Creates a custom response type that doesn't support inflight guards:
    /// - Implements AsyncEngineInflightGuards to return errors
    /// - Creates a mock engine that returns the failing response type
    /// - Validates that EngineInstance handles guard failures gracefully
    /// - Ensures that requests still succeed even when guards can't be added
    /// - Verifies that metrics are updated correctly despite guard failures
    /// - Confirms that inflight gauge remains at 0 when guards can't be added
    ///
    /// This test ensures robustness when working with responses that don't support guards.
    /// Tests graceful handling of inflight guard addition failures.
    ///
    /// Creates a custom response type that doesn't support inflight guards:
    /// - Implements AsyncEngineInflightGuards to return errors
    /// - Creates a mock engine that returns the failing response type
    /// - Validates that EngineInstance handles guard failures gracefully
    /// - Ensures that requests still succeed even when guards can't be added
    /// - Verifies that metrics are updated correctly despite guard failures
    /// - Confirms that inflight gauge remains at 0 when guards can't be added
    ///
    /// This test ensures robustness when working with responses that don't support guards.
    #[tokio::test]
    async fn test_guard_addition_failure() {
        // Test what happens when guard addition fails
        #[derive(Debug)]
        struct FailingResponse {
            context: Arc<MockAsyncEngineContext>,
        }

        impl AsyncEngineContextProvider for FailingResponse {
            fn context(&self) -> Arc<dyn AsyncEngineContext> {
                self.context.clone()
            }
        }

        impl AsyncEngineInflightGuards for FailingResponse {
            fn try_add_inflight_guard(&mut self, _guard: Box<dyn Any + Send + Sync>) -> bool {
                false
            }

            fn supports_inflight_guards(&self) -> bool {
                false
            }
        }

        // Create a mock engine that returns a FailingResponse
        struct FailingMockEngine;

        #[async_trait]
        impl AsyncEngine<TestRequest, FailingResponse, TestError> for FailingMockEngine {
            async fn generate(&self, _request: TestRequest) -> Result<FailingResponse, TestError> {
                Ok(FailingResponse {
                    context: Arc::new(MockAsyncEngineContext::new("failing-test".to_string())),
                })
            }
        }

        let instance = EngineInstance::new(
            "failing-guard-engine".to_string(),
            Arc::new(FailingMockEngine),
        );

        let request = TestRequest {
            id: 1,
            data: "test".to_string(),
        };

        // Initial gauge should be 0
        assert_eq!(instance.inflight_gauge.get(), 0);

        let result = instance.generate(request).await;

        // Request should still succeed even if guard addition fails
        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(!response.supports_inflight_guards());

        // Gauge should remain 0 since guard addition failed
        assert_eq!(instance.inflight_gauge.get(), 0);

        // Metrics should still be updated
        assert_eq!(instance.generate_success.get(), 1);
        assert_eq!(instance.generate_error.get(), 0);
    }

    /// Tests concurrent creation and cleanup of many inflight guards.
    ///
    /// Uses barrier synchronization to test high-concurrency guard operations:
    /// - Creates 50 guards concurrently across separate tasks
    /// - Uses a barrier to ensure all guards are created before validation
    /// - Validates that all guards are active simultaneously (gauge = 50)
    /// - Collects all guards to trigger their Drop implementations
    /// - Adds explicit timing to ensure Drop implementations complete
    /// - Verifies that all guards are properly cleaned up (gauge = 0)
    ///
    /// This test validates that the InflightGuard RAII mechanism works correctly
    /// under high concurrency without race conditions or resource leaks.
    /// Tests concurrent creation and cleanup of many inflight guards.
    ///
    /// Uses barrier synchronization to test high-concurrency guard operations:
    /// - Creates 50 guards concurrently across separate tasks
    /// - Uses a barrier to ensure all guards are created before validation
    /// - Validates that all guards are active simultaneously (gauge = 50)
    /// - Collects all guards to trigger their Drop implementations
    /// - Adds explicit timing to ensure Drop implementations complete
    /// - Verifies that all guards are properly cleaned up (gauge = 0)
    ///
    /// This test validates that the InflightGuard RAII mechanism works correctly
    /// under high concurrency without race conditions or resource leaks.
    #[tokio::test]
    async fn test_concurrent_guard_operations() {
        use tokio::sync::Barrier;
        use tokio::time::{sleep, Duration};

        let gauge = IntGauge::new("concurrent_test_gauge", "Concurrent test gauge").unwrap();
        let num_guards = 50;
        let barrier = Arc::new(Barrier::new(num_guards + 1)); // +1 for main task

        let mut handles = Vec::new();

        // Create many guards concurrently
        for _i in 0..num_guards {
            let gauge_clone = gauge.clone();
            let barrier_clone = barrier.clone();

            let handle = tokio::spawn(async move {
                let guard = InflightGuard::with_value(gauge_clone, 1);

                // Wait for all guards to be created
                barrier_clone.wait().await;

                // Keep guard alive until task completes
                guard
            });
            handles.push(handle);
        }

        // Wait for all guards to be created
        barrier.wait().await;

        // All guards should be active
        assert_eq!(gauge.get(), num_guards as i64);

        // Collect all guards (this will drop them)
        let _guards: Vec<_> = futures::future::try_join_all(handles).await.unwrap();

        // Explicitly drop the guards vector to ensure cleanup
        drop(_guards);

        // Give more time for the Drop implementations to run
        sleep(Duration::from_millis(10)).await;

        // All guards should be cleaned up
        assert_eq!(gauge.get(), 0);
    }

    /// Tests comprehensive error handling across all error types.
    ///
    /// Validates error handling for each defined error variant:
    /// - ProcessingFailed: Tests error with custom message
    /// - NetworkError: Tests error with status code
    /// - Timeout: Tests simple timeout error
    ///
    /// For each error type, validates that:
    /// - The correct error is returned from the engine
    /// - Error metrics are incremented appropriately
    /// - No inflight guards are created for failed requests
    /// - Success metrics remain at 0
    ///
    /// This test ensures comprehensive error coverage and proper resource management.
    /// Tests comprehensive error handling across all error types.
    ///
    /// Validates error handling for each defined error variant:
    /// - ProcessingFailed: Tests error with custom message
    /// - NetworkError: Tests error with status code
    /// - Timeout: Tests simple timeout error
    ///
    /// For each error type, validates that:
    /// - The correct error is returned from the engine
    /// - Error metrics are incremented appropriately
    /// - No inflight guards are created for failed requests
    /// - Success metrics remain at 0
    ///
    /// This test ensures comprehensive error coverage and proper resource management.
    #[tokio::test]
    async fn test_error_types_comprehensive() {
        let mock_engine = MockAsyncEngine::new("comprehensive-error-engine");

        // Add different error types
        mock_engine.add_error_response(
            1,
            TestError::ProcessingFailed {
                message: "Processing error".to_string(),
            },
        );
        mock_engine.add_error_response(2, TestError::NetworkError { code: 500 });
        mock_engine.add_error_response(3, TestError::Timeout);

        let instance =
            create_test_engine_instance_with_mock("comprehensive-error-engine", mock_engine);

        // Test each error type
        for (id, expected_error) in [
            (
                1,
                TestError::ProcessingFailed {
                    message: "Processing error".to_string(),
                },
            ),
            (2, TestError::NetworkError { code: 500 }),
            (3, TestError::Timeout),
        ] {
            let request = TestRequest {
                id,
                data: format!("error test {}", id),
            };

            let result = instance.generate(request).await;
            assert!(result.is_err());
            assert_eq!(result.unwrap_err(), expected_error);
        }

        // Verify all errors were counted and no guards were created
        assert_eq!(instance.generate_success.get(), 0);
        assert_eq!(instance.generate_error.get(), 3);
        assert_eq!(instance.inflight_gauge.get(), 0);
    }

    /// Tests high-concurrency request processing with proper synchronization.
    ///
    /// Uses a two-barrier approach for deterministic concurrent testing:
    /// 1. First barrier: Ensures all tasks have captured their responses
    /// 2. Validates inflight counter while all responses are alive
    /// 3. Second barrier: Allows all tasks to complete and drop responses
    /// 4. Validates cleanup after all responses are dropped
    ///
    /// Spawns 10 concurrent requests and validates:
    /// - All requests succeed with correct response data
    /// - Metrics are properly updated (10 successes, 0 errors)
    /// - Inflight gauge correctly tracks active responses (10 while alive)
    /// - Proper cleanup when responses are dropped (gauge returns to 0)
    ///
    /// This test validates the system's behavior under realistic concurrent load.
    /// Tests high-concurrency request processing with proper synchronization.
    ///
    /// Uses a two-barrier approach for deterministic concurrent testing:
    /// 1. First barrier: Ensures all tasks have captured their responses
    /// 2. Validates inflight counter while all responses are alive
    /// 3. Second barrier: Allows all tasks to complete and drop responses
    /// 4. Validates cleanup after all responses are dropped
    ///
    /// Spawns 10 concurrent requests and validates:
    /// - All requests succeed with correct response data
    /// - Metrics are properly updated (10 successes, 0 errors)
    /// - Inflight gauge correctly tracks active responses (10 while alive)
    /// - Proper cleanup when responses are dropped (gauge returns to 0)
    ///
    /// This test validates the system's behavior under realistic concurrent load.
    #[tokio::test]
    async fn test_multiple_concurrent_requests() {
        use tokio::sync::Barrier;

        let instance = Arc::new(create_test_engine_instance("concurrent-engine"));
        let num_tasks = 10;

        // Barrier to ensure all tasks have captured their responses before we check inflight counter
        let start_barrier = Arc::new(Barrier::new(num_tasks + 1)); // +1 for main task
                                                                   // Barrier to hold tasks until we've validated the inflight counter
        let continue_barrier = Arc::new(Barrier::new(num_tasks + 1)); // +1 for main task

        let mut handles = Vec::new();

        // Spawn multiple concurrent requests
        for i in 0..num_tasks {
            let instance_clone = instance.clone();
            let start_barrier_clone = start_barrier.clone();
            let continue_barrier_clone = continue_barrier.clone();

            let handle = tokio::spawn(async move {
                let request = TestRequest {
                    id: i as u64,
                    data: format!("concurrent data {}", i),
                };

                // Generate the request and capture the response
                let result = instance_clone.generate(request).await;

                // Validate response is ok
                assert!(result.is_ok());
                let response = result.unwrap();
                assert_eq!(response.id, i as u64);

                // Signal that this task has captured its response
                start_barrier_clone.wait().await;

                // Wait for main task to validate inflight counter
                continue_barrier_clone.wait().await;

                // Return the response so it stays alive until the task completes
                response
            });
            handles.push(handle);
        }

        // Wait for all tasks to capture their responses
        start_barrier.wait().await;

        // Now all tasks have their responses captured - check inflight counter
        assert_eq!(instance.generate_success.get(), num_tasks as u64);
        assert_eq!(instance.generate_error.get(), 0);
        assert_eq!(instance.inflight_gauge.get(), num_tasks as i64);

        // Allow all tasks to complete
        continue_barrier.wait().await;

        // Wait for all tasks to finish (this will drop their responses)
        let results: Vec<_> = futures::future::try_join_all(handles).await.unwrap();

        // Verify all responses are valid
        assert_eq!(results.len(), num_tasks);
        for (i, response) in results.iter().enumerate() {
            assert_eq!(response.id, i as u64);
        }

        // Drop all responses to cleanup guards
        drop(results);

        // Give a moment for cleanup
        tokio::task::yield_now().await;

        // Now gauge should be back to 0
        assert_eq!(instance.inflight_gauge.get(), 0);
    }

    /// Tests the Debug trait implementation for EngineInstance.
    ///
    /// Validates that:
    /// - Debug formatting produces the expected string format
    /// - The engine name is correctly included in the debug output
    /// - Format follows the pattern: "EngineInstance<name: {name}>"
    ///
    /// This is a simple but important test ensuring debugging output is useful.
    /// Tests the Debug trait implementation for EngineInstance.
    ///
    /// Validates that:
    /// - Debug formatting produces the expected string format
    /// - The engine name is correctly included in the debug output
    /// - Format follows the pattern: "EngineInstance<name: {name}>"
    ///
    /// This is a simple but important test ensuring debugging output is useful.
    #[tokio::test]
    async fn test_engine_instance_debug_format() {
        let instance = create_test_engine_instance("debug-test-engine");
        let debug_str = format!("{:?}", instance);
        assert_eq!(debug_str, "EngineInstance<name: debug-test-engine>");
    }

    /// Tests InflightGuard behavior with custom increment values.
    ///
    /// Validates that:
    /// - Guards can be created with custom values (not just 1)
    /// - The gauge is incremented by the custom value when guard is created
    /// - The gauge is decremented by the same value when guard is dropped
    /// - Proper cleanup occurs with non-standard values
    ///
    /// This test ensures the RAII mechanism works correctly with arbitrary values.
    /// Tests InflightGuard behavior with custom increment values.
    ///
    /// Validates that:
    /// - Guards can be created with custom values (not just 1)
    /// - The gauge is incremented by the custom value when guard is created
    /// - The gauge is decremented by the same value when guard is dropped
    /// - Proper cleanup occurs with non-standard values
    ///
    /// This test ensures the RAII mechanism works correctly with arbitrary values.
    #[tokio::test]
    async fn test_inflight_guard_with_custom_value() {
        let gauge = IntGauge::new("test_gauge", "Test gauge").unwrap();

        // Test guard with custom value
        {
            let _guard = InflightGuard::with_value(gauge.clone(), 5);
            assert_eq!(gauge.get(), 5);
        }

        // After guard is dropped, gauge should be decremented
        assert_eq!(gauge.get(), 0);
    }

    /// Tests multiple InflightGuards with different values on the same gauge.
    ///
    /// Validates that:
    /// - Multiple guards can be active simultaneously on the same gauge
    /// - Each guard contributes its individual value to the total
    /// - Guards can be dropped individually with correct value decrements
    /// - Final cleanup brings the gauge back to 0
    ///
    /// Tests with values 3, 2, and 1 to ensure:
    /// - Total gauge value is 6 when all are active
    /// - Dropping guards individually decrements correctly (6→3→1→0)
    ///
    /// This test validates that multiple guards work correctly together.
    /// Tests multiple InflightGuards with different values on the same gauge.
    ///
    /// Validates that:
    /// - Multiple guards can be active simultaneously on the same gauge
    /// - Each guard contributes its individual value to the total
    /// - Guards can be dropped individually with correct value decrements
    /// - Final cleanup brings the gauge back to 0
    ///
    /// Tests with values 3, 2, and 1 to ensure:
    /// - Total gauge value is 6 when all are active
    /// - Dropping guards individually decrements correctly (6→3→1→0)
    ///
    /// This test validates that multiple guards work correctly together.
    #[tokio::test]
    async fn test_inflight_guard_multiple_guards() {
        let gauge = IntGauge::new("multi_test_gauge", "Multi test gauge").unwrap();

        // Create multiple guards
        let guard1 = InflightGuard::with_value(gauge.clone(), 3);
        let guard2 = InflightGuard::with_value(gauge.clone(), 2);
        let guard3 = InflightGuard::with_value(gauge.clone(), 1);

        // Gauge should show sum of all guards
        assert_eq!(gauge.get(), 6);

        // Drop guards one by one
        drop(guard1);
        assert_eq!(gauge.get(), 3);

        drop(guard2);
        assert_eq!(gauge.get(), 1);

        drop(guard3);
        assert_eq!(gauge.get(), 0);
    }

    /// Tests the AsyncEngineInflightGuards trait implementation on TestResponse.
    ///
    /// Validates that:
    /// - TestResponse correctly reports that it supports inflight guards
    /// - Guards can be successfully added to the response
    /// - Added guards are stored in the response's guard collection
    /// - Guards can be downcasted back to their original types
    ///
    /// This test ensures the mock response type properly implements the guard
    /// interface needed for testing the EngineInstance behavior.
    /// Tests the AsyncEngineInflightGuards trait implementation on TestResponse.
    ///
    /// Validates that:
    /// - TestResponse correctly reports that it supports inflight guards
    /// - Guards can be successfully added to the response
    /// - Added guards are stored in the response's guard collection
    /// - Guards can be downcasted back to their original types
    ///
    /// This test ensures the mock response type properly implements the guard
    /// interface needed for testing the EngineInstance behavior.
    #[tokio::test]
    async fn test_response_guard_functionality() {
        let mut response = TestResponse::new(1, "test".to_string());

        // Verify initial state
        assert!(response.supports_inflight_guards());
        assert_eq!(response.guards.lock().unwrap().len(), 0);

        // Add a guard
        let guard = Box::new(42u32) as Box<dyn Any + Send + Sync>;
        let result = response.try_add_inflight_guard(guard);

        assert!(result);
        assert_eq!(response.guards.lock().unwrap().len(), 1);

        // Verify we can downcast the guard back
        let guards = response.guards.lock().unwrap();
        let guard_ref = guards[0].downcast_ref::<u32>().unwrap();
        assert_eq!(*guard_ref, 42);
    }

    /// Stress test with 100 concurrent requests using barrier synchronization.
    ///
    /// Uses the same two-barrier approach as test_multiple_concurrent_requests
    /// but with a much higher load (100 concurrent requests) to validate:
    /// - System stability under high concurrent load
    /// - Proper synchronization with many tasks
    /// - Correct metrics tracking at scale (100 successes, 0 errors)
    /// - Accurate inflight gauge tracking (100 while active, 0 after cleanup)
    /// - Memory management with many concurrent guards
    ///
    /// This test serves as a stress test to identify potential race conditions,
    /// memory leaks, or performance issues that might not appear with smaller loads.
    #[tokio::test]
    async fn test_engine_instance_stress_test() {
        use tokio::sync::Barrier;

        let instance = Arc::new(create_test_engine_instance("stress-test-engine"));
        let num_tasks = 100;

        // Barriers for synchronization
        let start_barrier = Arc::new(Barrier::new(num_tasks + 1)); // +1 for main task
        let continue_barrier = Arc::new(Barrier::new(num_tasks + 1)); // +1 for main task

        let mut handles = Vec::new();

        // Generate many concurrent requests with random data
        for i in 0..num_tasks {
            let instance_clone = instance.clone();
            let start_barrier_clone = start_barrier.clone();
            let continue_barrier_clone = continue_barrier.clone();

            let handle = tokio::spawn(async move {
                let request = TestRequest {
                    id: i as u64,
                    data: format!("stress_test_data_{}", i),
                };

                // Generate the request and capture the response
                let result = instance_clone.generate(request).await;

                // Validate response is ok
                assert!(result.is_ok());
                let response = result.unwrap();

                // Signal that this task has captured its response
                start_barrier_clone.wait().await;

                // Wait for main task to validate inflight counter
                continue_barrier_clone.wait().await;

                // Return the response so it stays alive until the task completes
                response
            });
            handles.push(handle);
        }

        // Wait for all tasks to capture their responses
        start_barrier.wait().await;

        // Now all tasks have their responses captured - check inflight counter
        assert_eq!(instance.generate_success.get(), num_tasks as u64);
        assert_eq!(instance.generate_error.get(), 0);
        assert_eq!(instance.inflight_gauge.get(), num_tasks as i64);

        // Allow all tasks to complete
        continue_barrier.wait().await;

        // Wait for all tasks to finish (this will drop their responses)
        let results = futures::future::try_join_all(handles).await.unwrap();

        // All should succeed
        assert_eq!(results.len(), num_tasks);
        for result in &results {
            assert_eq!(
                result.result,
                format!("Processed: stress_test_data_{}", result.id)
            );
        }

        // Drop all responses to cleanup guards
        drop(results);

        // Give a moment for cleanup
        tokio::task::yield_now().await;

        // Now gauge should be back to 0
        assert_eq!(instance.inflight_gauge.get(), 0);
    }

    /// Tests timing behavior with delayed mock engine responses.
    ///
    /// Validates that:
    /// - EngineInstance properly handles engines with artificial delays
    /// - Timing measurements work correctly (elapsed >= expected delay)
    /// - Metrics are updated correctly even with delayed responses
    /// - The delay doesn't interfere with guard or metrics functionality
    ///
    /// Uses a 100ms delay to test timing without making tests too slow.
    /// This test ensures EngineInstance works correctly with slow underlying engines.
    #[tokio::test]
    async fn test_engine_with_delay_timing() {
        use std::time::Instant;

        let delay = Duration::from_millis(100);
        let mock_engine = MockAsyncEngine::new("delay-engine").with_delay(delay);
        let instance = create_test_engine_instance_with_mock("delay-engine", mock_engine);

        let request = TestRequest {
            id: 1,
            data: "delayed request".to_string(),
        };

        let start = Instant::now();
        let result = instance.generate(request).await;
        let elapsed = start.elapsed();

        // Should succeed and take at least the delay time
        assert!(result.is_ok());
        assert!(elapsed >= delay);

        // Metrics should be updated
        assert_eq!(instance.generate_success.get(), 1);
        assert_eq!(instance.generate_error.get(), 0);
    }

    /// Tests InflightGuard behavior with edge case values.
    ///
    /// Validates guard behavior with unusual values:
    /// - Zero value: Guard with 0 increment (gauge should remain unchanged)
    /// - Negative value: Guard with -5 (gauge should decrease, then return to 0)
    /// - Maximum value: Guard with i64::MAX (tests extreme values)
    ///
    /// Each test validates:
    /// - Gauge is modified by the expected amount when guard is created
    /// - Gauge returns to original value when guard is dropped
    /// - No overflow or underflow issues with extreme values
    ///
    /// This test ensures the RAII mechanism is robust with edge case values.
    #[tokio::test]
    async fn test_guard_value_edge_cases() {
        let gauge = IntGauge::new("edge_case_gauge", "Edge case gauge").unwrap();

        // Test with zero value
        {
            let _guard = InflightGuard::with_value(gauge.clone(), 0);
            assert_eq!(gauge.get(), 0);
        }
        assert_eq!(gauge.get(), 0);

        // Test with negative value
        {
            let _guard = InflightGuard::with_value(gauge.clone(), -5);
            assert_eq!(gauge.get(), -5);
        }
        assert_eq!(gauge.get(), 0);

        // Test with large value
        {
            let _guard = InflightGuard::with_value(gauge.clone(), i64::MAX);
            assert_eq!(gauge.get(), i64::MAX);
        }
        assert_eq!(gauge.get(), 0);
    }

    /// Tests multiple sequential requests to validate call counting behavior.
    ///
    /// Validates that:
    /// - Multiple sequential requests all succeed
    /// - Success metrics are properly incremented for each request
    /// - No error metrics are recorded for successful requests
    /// - The underlying mock engine is called the expected number of times
    ///
    /// Makes 5 sequential requests to test:
    /// - Sequential request processing works correctly
    /// - Metrics accumulate properly across multiple calls
    /// - No interference between sequential requests
    ///
    /// Note: Cannot directly access MockAsyncEngine.call_count() through the trait
    /// object, so validates behavior indirectly through metrics.
    #[tokio::test]
    async fn test_mock_engine_call_counting() {
        let mock_engine = MockAsyncEngine::new("call-counter-engine");
        let instance = create_test_engine_instance_with_mock("call-counter-engine", mock_engine);

        // Get reference to underlying mock engine to check call count
        let _mock_ref = &instance.engine;

        // Initially no calls
        // Note: We can't directly access the mock engine's call_count method through the trait object
        // So we'll verify behavior indirectly through multiple requests

        for i in 0..5 {
            let request = TestRequest {
                id: i,
                data: format!("call {}", i),
            };

            let result = instance.generate(request).await;
            assert!(result.is_ok());
        }

        // Verify all calls succeeded
        assert_eq!(instance.generate_success.get(), 5);
        assert_eq!(instance.generate_error.get(), 0);
    }

    /// Tests AsyncEngineContext functionality through response context providers.
    ///
    /// Validates that:
    /// - Responses properly implement AsyncEngineContextProvider
    /// - Context IDs are correctly set (should match request ID)
    /// - Context control methods work correctly:
    ///   - Initial state: not stopped, not killed
    ///   - stop_generating() sets stopped state
    ///   - kill() sets both stopped and killed states
    ///
    /// This test ensures the context provider interface works correctly,
    /// which is important for request lifecycle management and cancellation.
    #[tokio::test]
    async fn test_context_provider_functionality() {
        let instance = create_test_engine_instance("context-test-engine");
        let request = TestRequest {
            id: 42,
            data: "context test".to_string(),
        };

        let result = instance.generate(request).await;
        assert!(result.is_ok());

        let response = result.unwrap();
        let context = response.context();

        // Verify context functionality
        assert_eq!(context.id(), "42");
        assert!(!context.is_stopped());
        assert!(!context.is_killed());

        // Test context control methods
        context.stop_generating();
        assert!(context.is_stopped());

        context.kill();
        assert!(context.is_killed());
    }
}
