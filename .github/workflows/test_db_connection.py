#!/usr/bin/env python3
"""
Simplified script to test database connection for GitHub Actions runners
"""
import requests
import sys
from datetime import datetime

def get_external_ip():
    """Get the external IP of the current runner"""
    try:
        response = requests.get("https://ifconfig.me", timeout=10)
        return response.text.strip()
    except Exception as e:
        return f"Unknown ({e})"

def test_database_connection():
    """Test actual database connection with a simple POST"""
    try:
        test_data = {
            "test": "runner_connection_check",
            "_id": f"test_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "runner_test": True,
            "workflow": "connection-test"
        }
        
        response = requests.post(
            "http://10.127.9.225/dataflow2/swdl-triton-ops-pipelines/posting",
            json=test_data,
            headers={
                "Content-Type": "application/json",
                "Accept-Charset": "UTF-8"
            },
            timeout=10
        )
        
        return {
            "success": response.status_code in [200, 201],
            "status_code": response.status_code,
            "response": response.text[:100] if response.text else "No response body"
        }
    except requests.exceptions.ConnectTimeout:
        return {"success": False, "error": "Connection timeout - likely firewall blocked"}
    except requests.exceptions.ConnectionError as e:
        return {"success": False, "error": f"Connection error: {str(e)[:150]}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)[:150]}"}

def main():
    print("🔍 GitHub Actions Runner Database Connection Test")
    print("=" * 60)
    
    # Get basic info
    external_ip = get_external_ip()
    print(f"🌐 Runner IP: {external_ip}")
    
    # Determine network type
    if external_ip.startswith(("10.", "192.168.", "172.")):
        network_type = "Internal/Private Network"
        expected_access = "✅ Expected to work"
    else:
        network_type = "External/Public Network"  
        expected_access = "❌ Likely blocked by firewall"
    
    print(f"🏠 Network Type: {network_type}")
    print(f"🔐 Database Access: {expected_access}")
    print()
    
    # Test database connection
    print("🗄️  Testing Database Connection...")
    db_result = test_database_connection()
    
    if db_result["success"]:
        print("✅ SUCCESS: Database connection works!")
        print(f"   Status Code: {db_result['status_code']}")
        print(f"   Response: {db_result['response']}")
        print("\n💡 RESULT: Metrics upload will work from this runner")
        print("🎯 RECOMMENDATION: Use this runner type for metrics collection")
    else:
        print("❌ FAILED: Database connection blocked")
        print(f"   Error: {db_result['error']}")
        print("\n💡 RESULT: Metrics upload will NOT work from this runner")
        print("🎯 RECOMMENDATIONS:")
        print("   1. Change to: runs-on: self-hosted")
        print("   2. Or use internal runners (gpu-l40-runners)")
        print("   3. Or request IP allowlisting for GitHub Actions")
    
    print("=" * 60)
    
    # Return exit code for workflow decisions
    return 0 if db_result["success"] else 1

if __name__ == "__main__":
    sys.exit(main())
