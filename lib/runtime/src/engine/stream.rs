use super::*;

/// Adapter for a [`DataStream`] to a [`ResponseStream`].
///
/// A common pattern is to consume the [`ResponseStream`] with standard stream combinators
/// which produces a [`DataStream`] stream, then form a [`ResponseStream`] by propagating the
/// original [`AsyncEngineContext`].
pub struct ResponseStream<R: Data> {
    stream: DataStream<R>,
    ctx: Arc<dyn AsyncEngineContext>,
    inflight_guards: Option<Vec<Box<dyn Any + Send + Sync>>>,
}

impl<R: Data> ResponseStream<R> {
    pub fn new(stream: DataStream<R>, ctx: Arc<dyn AsyncEngineContext>) -> Pin<Box<Self>> {
        Box::pin(Self {
            stream,
            ctx,
            inflight_guards: None,
        })
    }
}

impl<R: Data> Stream for ResponseStream<R> {
    type Item = R;

    #[inline]
    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        Pin::new(&mut self.stream).poll_next(cx)
    }
}

impl<R: Data> AsyncEngineStream<R> for ResponseStream<R> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl<R: Data> AsyncEngineInflightGuards for ResponseStream<R> {
    fn try_add_inflight_guard(&mut self, guard: Box<dyn Any + Send + Sync>) -> bool {
        match self.inflight_guards {
            None => {
                self.inflight_guards = Some(vec![guard]);
                true
            }
            Some(ref mut guards) => {
                guards.push(guard);
                true
            }
        }
    }

    fn supports_inflight_guards(&self) -> bool {
        true
    }
}

impl<R: Data> AsyncEngineContextProvider for ResponseStream<R> {
    fn context(&self) -> Arc<dyn AsyncEngineContext> {
        self.ctx.clone()
    }
}

impl<R: Data> Debug for ResponseStream<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResponseStream")
            // todo: add debug for stream - possibly propagate some information about what
            // engine created the stream
            // .field("stream", &self.stream)
            .field("ctx", &self.ctx)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{instance::InflightGuard, test_utils::*};
    use futures::StreamExt;
    use prometheus::{IntGauge, Registry};
    use std::sync::Arc;
    use tokio::sync::Barrier;

    /// Tests basic ResponseStream functionality without guards.
    ///
    /// Validates that:
    /// - ResponseStream can be created from a DataStream and context
    /// - Stream items can be consumed correctly
    /// - Context provider functionality works
    /// - Debug formatting works
    #[tokio::test]
    async fn test_response_stream_basic_functionality() {
        // Create test data
        let items = vec!["item1", "item2", "item3"];
        let data_stream: DataStream<&str> = Box::pin(futures::stream::iter(items.clone()));
        let context = Arc::new(MockAsyncEngineContext::new("test-stream-1".to_string()));

        // Create ResponseStream
        let mut response_stream = ResponseStream::new(data_stream, context.clone());

        // Test context provider
        assert_eq!(response_stream.context().id(), "test-stream-1");

        // Test stream consumption
        let mut collected_items = Vec::new();
        while let Some(item) = response_stream.next().await {
            collected_items.push(item);
        }

        assert_eq!(collected_items, items);

        // Test debug formatting
        let debug_str = format!("{:?}", response_stream);
        assert!(debug_str.contains("ResponseStream"));
        assert!(debug_str.contains("test-stream-1"));
    }

    /// Tests ResponseStream inflight guard functionality.
    ///
    /// Validates that:
    /// - Guards can be added to ResponseStream
    /// - Multiple guards can be stored
    /// - Guards are properly held during stream lifetime
    /// - Guards support inflight_guards trait correctly
    #[tokio::test]
    async fn test_response_stream_inflight_guards() {
        let registry = Registry::new();
        let gauge = IntGauge::new("test_response_stream_gauge", "Test gauge").unwrap();
        registry.register(Box::new(gauge.clone())).unwrap();

        // Create test stream
        let items = vec!["guard_item1", "guard_item2"];
        let data_stream: DataStream<&str> = Box::pin(futures::stream::iter(items));
        let context = Arc::new(MockAsyncEngineContext::new("guard-test".to_string()));
        let mut response_stream = ResponseStream::new(data_stream, context);

        // Initially gauge should be 0
        assert_eq!(gauge.get(), 0);

        // Test guard support
        assert!(response_stream.supports_inflight_guards());

        // Add first guard
        let guard1 = InflightGuard::with_value(gauge.clone(), 1);
        let success = response_stream.try_add_inflight_guard(Box::new(guard1));
        assert!(success);
        assert_eq!(gauge.get(), 1);

        // Add second guard with different value
        let guard2 = InflightGuard::with_value(gauge.clone(), 2);
        let success = response_stream.try_add_inflight_guard(Box::new(guard2));
        assert!(success);
        assert_eq!(gauge.get(), 3); // 1 + 2

        // Stream should still be consumable with guards
        let first_item = response_stream.next().await;
        assert_eq!(first_item, Some("guard_item1"));
        assert_eq!(gauge.get(), 3); // Guards still active

        let second_item = response_stream.next().await;
        assert_eq!(second_item, Some("guard_item2"));
        assert_eq!(gauge.get(), 3); // Guards still active

        // Stream exhausted but guards still active
        let no_more_items = response_stream.next().await;
        assert_eq!(no_more_items, None);
        assert_eq!(gauge.get(), 3); // Guards still active

        // Drop stream to cleanup guards
        drop(response_stream);

        // Give time for cleanup
        tokio::task::yield_now().await;

        // Guards should be cleaned up
        assert_eq!(gauge.get(), 0);
    }

    /// Tests streaming AsyncEngine with ResponseStream and inflight guard lifecycle.
    ///
    /// This test validates that inflight guards are properly managed throughout
    /// a streaming response lifecycle:
    /// 1. Guards are added when the stream is created
    /// 2. Guards remain active while the stream is being consumed
    /// 3. Guards are properly cleaned up when the stream is fully consumed
    /// 4. Guards are cleaned up even if the stream is dropped early
    ///
    /// Uses a mock streaming engine that returns a ResponseStream with multiple items,
    /// testing both full consumption and early termination scenarios.
    #[tokio::test]
    async fn test_streaming_engine_inflight_guards() {
        use crate::engine::instance::EngineInstance;

        // Create a streaming engine that returns ResponseStream
        struct StreamingMockEngine {
            gauge: IntGauge,
        }

        #[async_trait]
        impl AsyncEngine<TestRequest, EngineStream<String>, TestError> for StreamingMockEngine {
            async fn generate(
                &self,
                request: TestRequest,
            ) -> Result<EngineStream<String>, TestError> {
                // Create a stream that yields multiple items
                let items = vec![
                    format!("Item 1 for {}", request.data),
                    format!("Item 2 for {}", request.data),
                    format!("Item 3 for {}", request.data),
                    format!("Final item for {}", request.data),
                ];

                let stream = futures::stream::iter(items);
                let data_stream: DataStream<String> = Box::pin(stream);

                let context = Arc::new(MockAsyncEngineContext::new(request.id.to_string()));
                let mut response_stream = ResponseStream::new(data_stream, context);

                // Add an inflight guard to the stream
                let guard = InflightGuard::with_value(self.gauge.clone(), 1);
                let guard_added = response_stream.try_add_inflight_guard(Box::new(guard));
                assert!(
                    guard_added,
                    "Failed to add inflight guard to ResponseStream"
                );

                Ok(response_stream)
            }
        }

        let registry = Registry::new();
        let gauge = IntGauge::new("streaming_test_gauge", "Test gauge for streaming").unwrap();
        registry.register(Box::new(gauge.clone())).unwrap();

        let streaming_engine = Arc::new(StreamingMockEngine {
            gauge: gauge.clone(),
        });

        let instance = EngineInstance::new("streaming-test-engine".to_string(), streaming_engine);

        // Test 1: Full stream consumption
        {
            assert_eq!(gauge.get(), 0);

            let request = TestRequest {
                id: 1,
                data: "stream test".to_string(),
            };

            let result = instance.generate(request).await;
            assert!(result.is_ok());

            let mut stream = result.unwrap();

            // Verify guard was added (gauge should be incremented)
            assert_eq!(gauge.get(), 1);

            // Consume all items from the stream
            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item);
                // Guard should remain active during consumption
                assert_eq!(gauge.get(), 1);
            }

            // Verify we got all expected items
            assert_eq!(items.len(), 4);
            assert!(items[0].contains("Item 1 for stream test"));
            assert!(items[3].contains("Final item for stream test"));

            // Stream is now fully consumed but still alive
            assert_eq!(gauge.get(), 1);

            // Drop the stream
            drop(stream);

            // Give time for cleanup
            tokio::task::yield_now().await;

            // Guard should be cleaned up
            assert_eq!(gauge.get(), 0);
        }

        // Test 2: Early stream termination
        {
            assert_eq!(gauge.get(), 0);

            let request = TestRequest {
                id: 2,
                data: "early termination test".to_string(),
            };

            let result = instance.generate(request).await;
            assert!(result.is_ok());

            let mut stream = result.unwrap();

            // Verify guard was added
            assert_eq!(gauge.get(), 1);

            // Consume only first item
            let first_item = stream.next().await;
            assert!(first_item.is_some());
            assert!(first_item
                .unwrap()
                .contains("Item 1 for early termination test"));

            // Guard should still be active
            assert_eq!(gauge.get(), 1);

            // Drop stream early (before full consumption)
            drop(stream);

            // Give time for cleanup
            tokio::task::yield_now().await;

            // Guard should still be cleaned up properly
            assert_eq!(gauge.get(), 0);
        }

        // Test 3: Concurrent streams with barrier synchronization
        {
            assert_eq!(gauge.get(), 0);

            let barrier = Arc::new(Barrier::new(4)); // 3 tasks + 1 main task
            let mut handles = Vec::new();

            for i in 0..3 {
                let instance_clone = Arc::new(EngineInstance::new(
                    format!("concurrent-stream-{}", i),
                    Arc::new(StreamingMockEngine {
                        gauge: gauge.clone(),
                    }),
                ));
                let barrier_clone = barrier.clone();

                let handle = tokio::spawn(async move {
                    let request = TestRequest {
                        id: i + 10,
                        data: format!("concurrent stream {}", i),
                    };

                    let result = instance_clone.generate(request).await;
                    assert!(result.is_ok());

                    let mut stream = result.unwrap();

                    // Signal that stream is created and guard is active
                    barrier_clone.wait().await;

                    // Consume first item to keep stream active
                    let first_item = stream.next().await;
                    assert!(first_item.is_some());

                    // Return stream to keep it alive until task completes
                    stream
                });
                handles.push(handle);
            }

            // Wait for all streams to be created
            barrier.wait().await;

            // All 3 concurrent streams should have active guards
            assert_eq!(gauge.get(), 3);

            // Wait for all tasks to complete (this drops the streams)
            let streams = futures::future::try_join_all(handles).await.unwrap();

            // Streams are still alive
            assert_eq!(gauge.get(), 3);

            // Drop all streams
            drop(streams);

            // Give time for cleanup
            tokio::task::yield_now().await;

            // All guards should be cleaned up
            assert_eq!(gauge.get(), 0);
        }
    }

    /// Tests ResponseStream context control functionality.
    ///
    /// Validates that:
    /// - Context methods work correctly through ResponseStream
    /// - stop_generating() and kill() affect context state
    /// - Context state can be queried correctly
    #[tokio::test]
    async fn test_response_stream_context_control() {
        let items = vec!["ctx1", "ctx2"];
        let data_stream: DataStream<&str> = Box::pin(futures::stream::iter(items));
        let context = Arc::new(MockAsyncEngineContext::new(
            "context-control-test".to_string(),
        ));
        let response_stream = ResponseStream::new(data_stream, context.clone());

        let stream_context = response_stream.context();

        // Initial state
        assert_eq!(stream_context.id(), "context-control-test");
        assert!(!stream_context.is_stopped());
        assert!(!stream_context.is_killed());

        // Test stop
        stream_context.stop_generating();
        assert!(stream_context.is_stopped());
        assert!(!stream_context.is_killed());

        // Test kill
        stream_context.kill();
        assert!(stream_context.is_stopped());
        assert!(stream_context.is_killed());
    }
}
