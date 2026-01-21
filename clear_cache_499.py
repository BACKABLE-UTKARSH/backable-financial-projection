import requests
import sys
import os

# Fix Windows encoding issues
if os.name == 'nt':
    sys.stdout.reconfigure(encoding='utf-8')

# Configuration
CLIENT_ID = "499"
API_URL = "http://localhost:8005"

def clear_cache_via_api():
    """Clear cache for client 499 using the API endpoint"""
    try:
        print(f"🗑️ Clearing cache for client {CLIENT_ID}...")

        # Call the clear-cache endpoint
        response = requests.post(
            f"{API_URL}/clear-cache",
            params={"client_id": CLIENT_ID}
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result['message']}")
            print(f"📊 Cleared {result['cleared_count']} cached projections")
            return True
        else:
            print(f"❌ Error: HTTP {response.status_code}")
            print(f"Details: {response.text}")
            return False

    except requests.ConnectionError:
        print(f"❌ Connection Error: Cannot connect to {API_URL}")
        print("Make sure your backend is running on port 8005")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

def test_cache_status():
    """Test if cache is cleared by trying to fetch projection"""
    try:
        print(f"\n🔍 Testing cache status for client {CLIENT_ID}...")

        # Try to fetch projection
        response = requests.post(
            f"{API_URL}/predict",
            params={"client_id": CLIENT_ID}
        )

        if response.status_code == 200:
            result = response.json()
            if result.get('from_cache'):
                print("⚠️ Data is still cached (might be re-cached already)")
            else:
                print("✅ Data is freshly generated (cache was cleared)")
            return True
        else:
            print(f"❌ Error fetching projection: HTTP {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Error testing cache: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("CACHE CLEANER FOR CLIENT 499")
    print("=" * 50)

    # Clear the cache
    if clear_cache_via_api():
        print("\n✨ Cache cleared successfully!")

        # Optional: Test if cache is really cleared
        print("\nWould you like to test if cache is cleared? (y/n): ", end="")
        if input().lower() == 'y':
            test_cache_status()
    else:
        print("\n❌ Failed to clear cache")
        print("\nTroubleshooting:")
        print("1. Make sure your backend is running: python BACKABLE NEW INFRASTRUCTURE FINANCIAL PROJECTION.py")
        print("2. Check if the backend is accessible at http://localhost:8005")
        print("3. Try accessing http://localhost:8005/health in your browser")