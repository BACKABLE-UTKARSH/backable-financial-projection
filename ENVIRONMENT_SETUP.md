# Environment Setup Guide - Financial Projection Engine

## Overview
This guide explains how to set up the Financial Projection Engine for local development and Azure deployment.

## Version
**Version 3.3.0** - Enhanced with Vertex AI support and environment variables

## Prerequisites
- Python 3.9+
- PostgreSQL database access
- Azure Storage account
- Google Cloud Platform (Vertex AI) access
- Gemini API keys (fallback)

## Local Development Setup

### 1. Create `.env` File
Copy `.env.example` to `.env` and fill in your actual credentials:

```bash
cp .env.example .env
```

### 2. Required Environment Variables

#### Vertex AI (Primary Method)
```env
VERTEX_PROJECT_ID=your-gcp-project-id
VERTEX_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS_JSON={"type":"service_account",...}
```

**Getting Vertex AI Credentials:**
1. Go to Google Cloud Console
2. Navigate to IAM & Admin > Service Accounts
3. Create or select a service account
4. Create a JSON key
5. Copy the entire JSON content to `GOOGLE_APPLICATION_CREDENTIALS_JSON`

#### Gemini API Keys (Fallback Method)
```env
GEMINI_API_KEY_1=AIzaSy...
GEMINI_API_KEY_2=AIzaSy...
# ... up to GEMINI_API_KEY_10
```

**Getting Gemini API Keys:**
1. Go to https://makersuite.google.com/app/apikey
2. Create API keys
3. Add them to your `.env` file

#### Azure Storage
```env
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
```

**Getting Azure Storage Connection String:**
1. Go to Azure Portal
2. Navigate to your Storage Account
3. Access Keys > Show keys
4. Copy Connection String

#### Database Configuration
```env
# Onboarding Database (Google Infrastructure)
ONBOARDING_DB_HOST=memberchat-db.postgres.database.azure.com
ONBOARDING_DB_NAME=BACKABLE-GOOGLE-RAG
ONBOARDING_DB_USER=backable
ONBOARDING_DB_PASSWORD=your-password
ONBOARDING_DB_PORT=5432

# Finance Engine Database
FINANCE_DB_HOST=memberchat-db.postgres.database.azure.com
FINANCE_DB_NAME=BACKABLE-FINANCE-ENGINE
FINANCE_DB_USER=backable
FINANCE_DB_PASSWORD=your-password
FINANCE_DB_PORT=5432

# Financial Projection Database
PROJECTIONS_DB_HOST=memberchat-db.postgres.database.azure.com
PROJECTIONS_DB_NAME=BACKABLE-FINANCIAL-PROJECTION
PROJECTIONS_DB_USER=backable
PROJECTIONS_DB_PASSWORD=your-password
PROJECTIONS_DB_PORT=5432
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Locally
```bash
python "BACKABLE NEW INFRASTRUCTURE FINANCIAL PROJECTION.py"
```

## Azure Deployment

### GitHub Secrets Setup
For Azure deployment via GitHub Actions, add these secrets to your repository:

1. Go to GitHub Repository > Settings > Secrets and variables > Actions
2. Add the following secrets:

```
VERTEX_PROJECT_ID
VERTEX_LOCATION
GOOGLE_APPLICATION_CREDENTIALS_JSON
GEMINI_API_KEY_1 through GEMINI_API_KEY_10
AZURE_STORAGE_CONNECTION_STRING
ONBOARDING_DB_PASSWORD
FINANCE_DB_PASSWORD
PROJECTIONS_DB_PASSWORD
```

### Azure App Service Configuration
Add these Application Settings in Azure Portal:

1. Go to Azure Portal > App Service > Configuration
2. Add all environment variables from `.env` file
3. Save and restart the app

## Architecture

### Primary Method: Vertex AI
- **Model**: gemini-2.5-pro
- **Thinking Budget**: 32,768 tokens
- **Temperature**: 0.0 (deterministic)
- **Location**: us-central1
- **Authentication**: Service Account (via JSON credentials)

### Fallback Method: Gemini API Keys
- **Keys**: 10 API keys in rotation
- **Usage**: Activated when Vertex AI fails
- **Rotation**: Round-robin distribution

## New Infrastructure Integration

### Dynamic Container Lookup
The engine dynamically fetches Azure container names from the database:
- Queries `client_onboarding` table
- Column: `azure_container_name`
- Supports multiple containers: `unified-clients-prod`, `unified-clients-prod-2`, etc.

### Dynamic Folder Structure
Client-specific folders are retrieved from the database:
- Queries `client_onboarding` table
- Column: `folder_name`
- Structure: `{container}/{client_folder}/financial intelligence report/`

### Database: BACKABLE-GOOGLE-RAG
All client onboarding data is stored in the new Google infrastructure database.

## Security Best Practices

### What NOT to Commit to GitHub
- `.env` file (contains real credentials)
- `vertex-key.json` (service account key file)
- Any files with credentials or API keys
- `publish-profile.xml` (Azure deployment credentials)

### What to Commit
- `.env.example` (template with placeholders)
- `.gitignore` (ensures sensitive files are excluded)
- Source code (no hardcoded credentials)

### Checking Before Commit
```bash
# Make sure .env is in .gitignore
cat .gitignore | grep .env

# Verify no credentials in code
grep -r "AIzaSy" *.py  # Should return nothing
grep -r "AccountKey=" *.py  # Should return nothing
```

## Testing

### Test Vertex AI Connection
```python
from google.oauth2 import service_account
import json
import os

creds_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
creds_dict = json.loads(creds_json)
print(f"Project ID: {creds_dict.get('project_id')}")
print(f"Client Email: {creds_dict.get('client_email')}")
```

### Test Database Connection
```python
import psycopg2
import os

conn = psycopg2.connect(
    host=os.getenv('ONBOARDING_DB_HOST'),
    dbname=os.getenv('ONBOARDING_DB_NAME'),
    user=os.getenv('ONBOARDING_DB_USER'),
    password=os.getenv('ONBOARDING_DB_PASSWORD'),
    port=os.getenv('ONBOARDING_DB_PORT')
)
print("✅ Database connected successfully")
conn.close()
```

### Test API Endpoint
```bash
curl http://localhost:8000/health
```

## Troubleshooting

### Issue: "No Gemini API keys found"
**Solution**: Ensure `GEMINI_API_KEY_1` through `GEMINI_API_KEY_10` are set in `.env`

### Issue: "Vertex AI initialization failed"
**Solution**:
1. Verify `GOOGLE_APPLICATION_CREDENTIALS_JSON` is valid JSON
2. Check service account has necessary permissions
3. Ensure `VERTEX_PROJECT_ID` is correct

### Issue: "Database connection failed"
**Solution**:
1. Verify database credentials in `.env`
2. Check firewall rules allow your IP
3. Confirm database exists and user has access

### Issue: "Azure Blob storage access denied"
**Solution**:
1. Verify `AZURE_STORAGE_CONNECTION_STRING` is correct
2. Check storage account access keys are valid
3. Ensure container exists

## Monitoring

### Logs Location
- **Local**: Console output
- **Azure**: App Service > Log Stream
- **Storage**: `projection_storage/` folder (if enabled)

### Health Check Endpoint
```
GET /health
```

Returns system status and configuration.

## Support
For issues or questions, contact the Backable development team.
