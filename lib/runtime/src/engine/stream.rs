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

impl<R: Data> AsyncEngineStream<R> for ResponseStream<R> {}

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
