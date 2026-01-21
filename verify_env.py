"""
Environment Variables Verification Script
Checks that all required environment variables are properly loaded
"""

import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

def check_env_var(var_name, is_optional=False):
    """Check if environment variable is set"""
    value = os.getenv(var_name)
    if value:
        # Mask sensitive values
        if len(value) > 20:
            display_value = f"{value[:10]}...{value[-5:]}"
        else:
            display_value = f"{value[:5]}..."
        print(f"✅ {var_name}: {display_value}")
        return True
    else:
        if is_optional:
            print(f"⚠️  {var_name}: Not set (optional)")
        else:
            print(f"❌ {var_name}: NOT SET (required)")
        return False

print("=" * 80)
print("🔍 ENVIRONMENT VARIABLES VERIFICATION")
print("=" * 80)

# Track results
required_vars = []
optional_vars = []

# Vertex AI Configuration (Primary Method)
print("\n📍 VERTEX AI CONFIGURATION (Primary Method)")
print("-" * 80)
required_vars.append(check_env_var("VERTEX_PROJECT_ID"))
required_vars.append(check_env_var("VERTEX_LOCATION"))
required_vars.append(check_env_var("GOOGLE_APPLICATION_CREDENTIALS_JSON"))

# Verify Vertex AI JSON is valid
vertex_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if vertex_json:
    try:
        creds_dict = json.loads(vertex_json)
        print(f"   ↳ Project ID in JSON: {creds_dict.get('project_id', 'NOT FOUND')}")
        print(f"   ↳ Service Account: {creds_dict.get('client_email', 'NOT FOUND')}")
    except json.JSONDecodeError:
        print(f"   ❌ Invalid JSON format in GOOGLE_APPLICATION_CREDENTIALS_JSON")

# Gemini API Keys (Fallback Method)
print("\n📍 GEMINI API KEYS (Fallback Method)")
print("-" * 80)
api_keys_loaded = 0
for i in range(1, 11):
    if check_env_var(f"GEMINI_API_KEY_{i}", is_optional=True):
        api_keys_loaded += 1

print(f"\n   📊 Total API Keys Loaded: {api_keys_loaded}/10")
if api_keys_loaded > 0:
    print(f"   ✅ Fallback method available")
else:
    print(f"   ⚠️  No API keys - Vertex AI only mode")

# Azure Storage
print("\n📍 AZURE STORAGE CONFIGURATION")
print("-" * 80)
required_vars.append(check_env_var("AZURE_STORAGE_CONNECTION_STRING"))

# Onboarding Database
print("\n📍 ONBOARDING DATABASE (Google Infrastructure)")
print("-" * 80)
optional_vars.append(check_env_var("ONBOARDING_DB_HOST", is_optional=True))
optional_vars.append(check_env_var("ONBOARDING_DB_NAME", is_optional=True))
optional_vars.append(check_env_var("ONBOARDING_DB_USER", is_optional=True))
required_vars.append(check_env_var("ONBOARDING_DB_PASSWORD"))
optional_vars.append(check_env_var("ONBOARDING_DB_PORT", is_optional=True))

# Finance Database
print("\n📍 FINANCE DATABASE")
print("-" * 80)
optional_vars.append(check_env_var("FINANCE_DB_HOST", is_optional=True))
optional_vars.append(check_env_var("FINANCE_DB_NAME", is_optional=True))
optional_vars.append(check_env_var("FINANCE_DB_USER", is_optional=True))
required_vars.append(check_env_var("FINANCE_DB_PASSWORD"))
optional_vars.append(check_env_var("FINANCE_DB_PORT", is_optional=True))

# Projections Database
print("\n📍 PROJECTIONS DATABASE")
print("-" * 80)
optional_vars.append(check_env_var("PROJECTIONS_DB_HOST", is_optional=True))
optional_vars.append(check_env_var("PROJECTIONS_DB_NAME", is_optional=True))
optional_vars.append(check_env_var("PROJECTIONS_DB_USER", is_optional=True))
required_vars.append(check_env_var("PROJECTIONS_DB_PASSWORD"))
optional_vars.append(check_env_var("PROJECTIONS_DB_PORT", is_optional=True))

# Summary
print("\n" + "=" * 80)
print("📊 SUMMARY")
print("=" * 80)

required_count = sum(required_vars)
required_total = len(required_vars)
optional_count = sum(optional_vars)
optional_total = len(optional_vars)

print(f"✅ Required Variables: {required_count}/{required_total}")
print(f"ℹ️  Optional Variables: {optional_count}/{optional_total}")
print(f"🔑 Gemini API Keys: {api_keys_loaded}/10")

print("\n" + "=" * 80)
if required_count == required_total:
    print("🎉 ALL REQUIRED VARIABLES SET - Ready for deployment!")
else:
    print("⚠️  MISSING REQUIRED VARIABLES - Please check your .env file")
    print("\n📝 Next Steps:")
    print("   1. Copy .env.example to .env")
    print("   2. Fill in all required credentials")
    print("   3. Run this script again to verify")
print("=" * 80)

# Test database connection (optional)
test_db = input("\n🧪 Test database connections? (y/n): ").lower().strip()
if test_db == 'y':
    import psycopg2

    databases = [
        ("Onboarding", os.getenv("ONBOARDING_DB_HOST"), os.getenv("ONBOARDING_DB_NAME"),
         os.getenv("ONBOARDING_DB_USER"), os.getenv("ONBOARDING_DB_PASSWORD"),
         os.getenv("ONBOARDING_DB_PORT", "5432")),
        ("Finance", os.getenv("FINANCE_DB_HOST"), os.getenv("FINANCE_DB_NAME"),
         os.getenv("FINANCE_DB_USER"), os.getenv("FINANCE_DB_PASSWORD"),
         os.getenv("FINANCE_DB_PORT", "5432")),
        ("Projections", os.getenv("PROJECTIONS_DB_HOST"), os.getenv("PROJECTIONS_DB_NAME"),
         os.getenv("PROJECTIONS_DB_USER"), os.getenv("PROJECTIONS_DB_PASSWORD"),
         os.getenv("PROJECTIONS_DB_PORT", "5432"))
    ]

    print("\n" + "=" * 80)
    print("🧪 DATABASE CONNECTION TESTS")
    print("=" * 80)

    for name, host, dbname, user, password, port in databases:
        try:
            if not all([host, dbname, user, password, port]):
                print(f"⚠️  {name}: Missing credentials, skipping...")
                continue

            conn = psycopg2.connect(
                host=host,
                dbname=dbname,
                user=user,
                password=password,
                port=port
            )
            print(f"✅ {name} Database: Connected successfully")
            conn.close()
        except Exception as e:
            print(f"❌ {name} Database: {str(e)}")

    print("=" * 80)
