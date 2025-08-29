use opentelemetry::global;
use opentelemetry::trace::{TraceContextExt};
use opentelemetry_otlp::WithExportConfig;
use std::time::Duration;
use tracing::{info, instrument};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use tracing_opentelemetry::OpenTelemetrySpanExt;

#[instrument]
async fn simulate_work(task_name: &str, duration_ms: u64) {
    info!(task = task_name, "Starting task");
    tokio::time::sleep(Duration::from_millis(duration_ms)).await;
    
    // Simulate some nested work
    for i in 0..3 {
        simulate_nested_task(i, duration_ms / 3).await;
    }
    
    info!(task = task_name, "Task completed");
}

#[instrument]
async fn simulate_nested_task(id: u32, duration_ms: u64) {
    let current_span = tracing::Span::current();
    let otel_context = current_span.context();
    let span = otel_context.span();
    let span_context = span.span_context();
    
    info!(
        task_id = id,
        trace_id = %span_context.trace_id(),
        span_id = %span_context.span_id(),
        "Processing subtask"
    );
    
    tokio::time::sleep(Duration::from_millis(duration_ms)).await;
    
    if id == 1 {
        // Simulate an error condition occasionally
        tracing::warn!(
            task_id = id, 
            trace_id = %span_context.trace_id(),
            span_id = %span_context.span_id(),
            "Simulated warning in subtask");
    }
}

#[instrument]
async fn process_request(request_id: u32) {
    info!(request_id = request_id, "Processing new request");
    
    // Simulate different types of work
    simulate_work("database_query", 200).await;
    simulate_work("external_api_call", 300).await;
    simulate_work("data_processing", 150).await;
    
    info!(request_id = request_id, "Request processing completed");
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create OTLP exporter that sends to Tempo
    let otlp_exporter = opentelemetry_otlp::new_exporter()
        .tonic()
        .with_endpoint("http://localhost:4317");

    let tracer = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(otlp_exporter)
        .with_trace_config(
            opentelemetry_sdk::trace::config()
                .with_resource(opentelemetry_sdk::Resource::new(vec![
                    opentelemetry::KeyValue::new("service.name", "rust-tracing-playground"),
                    opentelemetry::KeyValue::new("service.version", "1.0.0"),
                ]))
        )
        .install_batch(opentelemetry_sdk::runtime::Tokio)?;

    // Initialize tracing subscriber
    tracing_subscriber::registry()
        .with(tracing_opentelemetry::layer().with_tracer(tracer))
        .with(tracing_subscriber::filter::LevelFilter::INFO)
        .with(
            tracing_subscriber::fmt::layer()
                .with_ansi(false)  // Disable ANSI colors and formatting
                .with_target(false) // Remove target (crate name) from output
        )
        .init();

    info!("🚀 Starting Rust Tracing Playground with OTLP export to Tempo");

    // Generate 2 separate requests with distinct trace IDs
    // Each process_request call will create its own root trace
    {
        let request_span = tracing::info_span!("request_1");
        let _enter = request_span.enter();
        info!("Processing first request");
        process_request(1).await;
    }
    
    // Wait between requests to clearly separate them
    tokio::time::sleep(Duration::from_secs(1)).await;
    
    {
        let request_span = tracing::info_span!("request_2");
        let _enter = request_span.enter();
        info!("Processing second request");
        process_request(2).await;
    }

    info!("🏁 Both requests completed, shutting down");
    
    // Ensure all traces are exported before shutdown
    global::shutdown_tracer_provider();
    
    // Give some time for final export
    tokio::time::sleep(Duration::from_secs(2)).await;
    
    Ok(())
}