use opentelemetry::global;
use opentelemetry::trace::{TracerProvider, TraceContextExt};
use opentelemetry_sdk::trace::TracerProvider as SdkTracerProvider;
use tracing::{info, instrument};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use tracing_opentelemetry::OpenTelemetrySpanExt;

#[instrument]
fn do_something() {
    info!("doing something interesting");
    for i in 0..3 {
        do_nested_work(i);
    }
}

#[instrument]
fn do_nested_work(id: i32) {
    // Get the current tracing span and extract OpenTelemetry context from it
    let current_span = tracing::Span::current();
    let otel_context = current_span.context();
    let span = otel_context.span();
    let span_context = span.span_context();
    
    info!(
        work_id = id,
        trace_id = %span_context.trace_id(),
        span_id = %span_context.span_id(),
        "performing nested work"
    );
    
    std::thread::sleep(std::time::Duration::from_millis(100));
}

fn main() {
    // Create a basic OpenTelemetry tracer (no export for simplicity)
    let provider = SdkTracerProvider::builder().build();
    let tracer = provider.tracer("rust-tracing-playground");
    global::set_tracer_provider(provider);

    // Initialize tracing subscriber with both console output and OpenTelemetry
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_opentelemetry::layer().with_tracer(tracer))
        .with(tracing_subscriber::filter::LevelFilter::INFO)
        .init();

    // Now create a root span AFTER the subscriber is initialized
    let root_span = tracing::info_span!("application");
    let _enter = root_span.enter();

    info!("starting the application");
    
    // This will create spans that work with OpenTelemetry
    do_something();
    
    // Now we're still in the root span, so we'll get the same trace ID
    let current_span = tracing::Span::current();
    let otel_context = current_span.context();
    let span = otel_context.span();
    let span_context = span.span_context();
    
    info!(
        trace_id = %span_context.trace_id(),
        span_id = %span_context.span_id(),
        "application shutting down"
    );
    info!(
        trace_id = %span_context.trace_id(),
        span_id = %span_context.span_id(),
        "another message"
    );
    
    // Ensure all spans are processed before shutdown
    global::shutdown_tracer_provider();
}