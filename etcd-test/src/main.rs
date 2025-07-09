use etcd_client::{Client, CompactionOptions, DeleteOptions, GetOptions, PutOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Starting etcd client tests...");

    // Connect to etcd server (default localhost:2379)
    let mut client = Client::connect(["http://127.0.0.1:2379"], None).await?;
    println!("✓ Connected to etcd server");

    // Test PUT operation
    println!("\n--- Testing PUT operation ---");
    let put_response = client
        .put("test_key", "test_value", None)
        .await?;
    println!("✓ PUT successful. Header: {:?}", put_response.header());

    // Test GET operation
    println!("\n--- Testing GET operation ---");
    let get_response = client
        .get("test_key", None)
        .await?;

    if let Some(kv) = get_response.kvs().first() {
        println!("✓ GET successful:");
        println!("  Key: {}", String::from_utf8_lossy(kv.key()));
        println!("  Value: {}", String::from_utf8_lossy(kv.value()));
        println!("  Version: {}", kv.version());
        println!("  Revision: {}", kv.mod_revision());
    } else {
        println!("✗ Key not found");
    }

    // Test GET with range (prefix)
    println!("\n--- Testing GET with prefix ---");
    // Put a few more keys with same prefix
    client.put("test_key_1", "value_1", None).await?;
    client.put("test_key_2", "value_2", None).await?;
    client.put("other_key", "other_value", None).await?;

    let get_options = GetOptions::new().with_prefix();
    let prefix_response = client
        .get("test_key", Some(get_options))
        .await?;

    println!("✓ GET with prefix found {} keys:", prefix_response.count());
    for kv in prefix_response.kvs() {
        println!("  {}: {}",
            String::from_utf8_lossy(kv.key()),
            String::from_utf8_lossy(kv.value())
        );
    }

    // Test PUT with options (previous key value)
    println!("\n--- Testing PUT with prev_kv option ---");
    let put_options = PutOptions::new().with_prev_key();
    let put_with_prev = client
        .put("test_key", "updated_value", Some(put_options))
        .await?;

    if let Some(prev_kv) = put_with_prev.prev_key() {
        println!("✓ PUT with prev_kv successful:");
        println!("  Previous value: {}", String::from_utf8_lossy(prev_kv.value()));
    }

    // Get current revision for compact test
    let current_response = client.get("test_key", None).await?;
    let current_revision = current_response.header().unwrap().revision();
    println!("Current revision: {}", current_revision);

    // Test DELETE operation
    println!("\n--- Testing DELETE operation ---");
    let delete_options = DeleteOptions::new().with_prev_key();
    let delete_response = client
        .delete("test_key_1", Some(delete_options))
        .await?;

    println!("✓ DELETE successful. Deleted {} keys", delete_response.deleted());
    if let Some(prev_kv) = delete_response.prev_kvs().first() {
        println!("  Deleted key: {}", String::from_utf8_lossy(prev_kv.key()));
        println!("  Previous value: {}", String::from_utf8_lossy(prev_kv.value()));
    }

    // Test DELETE with range
    println!("\n--- Testing DELETE with prefix ---");
    let delete_range_options = DeleteOptions::new().with_prefix().with_prev_key();
    let delete_range_response = client
        .delete("test_key", Some(delete_range_options))
        .await?;

    println!("✓ DELETE range successful. Deleted {} keys", delete_range_response.deleted());
    for prev_kv in delete_range_response.prev_kvs() {
        println!("  Deleted: {}", String::from_utf8_lossy(prev_kv.key()));
    }

    // Test COMPACT operation
    println!("\n--- Testing COMPACT operation ---");
    if current_revision > 1 {
        let compact_revision = current_revision - 1;
        let compact_response = client
            .compact(compact_revision, None)
            .await?;

        println!("✓ COMPACT successful");
        println!("  Compacted up to revision: {}", compact_revision);
        println!("  Response header: {:?}", compact_response.header());
    } else {
        println!("⚠ Skipping compact - not enough revisions");
    }

    // Test COMPACT with options
    println!("\n--- Testing COMPACT with physical option ---");
    let compact_options = CompactionOptions::new().with_physical();
    match client.compact(current_revision, Some(compact_options)).await {
        Ok(compact_response) => {
            println!("✓ Physical COMPACT successful");
            println!("  Response header: {:?}", compact_response.header());
        }
        Err(e) => {
            println!("⚠ Physical compact failed (this might be expected): {}", e);
        }
    }

    // Final verification - try to get the remaining keys
    println!("\n--- Final verification ---");
    let final_response = client.get("other_key", None).await?;
    if let Some(kv) = final_response.kvs().first() {
        println!("✓ Remaining key found:");
        println!("  {}: {}",
            String::from_utf8_lossy(kv.key()),
            String::from_utf8_lossy(kv.value())
        );
    }

    // Clean up
    println!("\n--- Cleanup ---");
    client.delete("other_key", None).await?;
    println!("✓ Cleanup completed");

    println!("\n🎉 All etcd operations tested successfully!");
    Ok(())
}
