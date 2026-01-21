# Financial Projection Engine - Vertex AI Migration Guide

## Overview
Convert the Financial Projection engine from using Gemini API keys to Vertex AI (like the Risk Engine), with API keys as fallback.

## Benefits of Vertex AI
1. **Higher Rate Limits**: No API key rotation needed
2. **Better Performance**: Direct Google Cloud infrastructure
3. **Enhanced Security**: Service account authentication
4. **Cost Efficiency**: Pay-as-you-go pricing
5. **Better Token Limits**: Up to 1M context window with Gemini 2.0

## Implementation Steps

### 1. Add Required Imports
Add these imports after existing imports:
```python
from google.oauth2 import service_account
from dotenv import load_dotenv
import tempfile
import threading
```

### 2. Add Vertex AI Configuration
Add after the database configuration section:
```python
# Vertex AI Configuration (Primary Method)
VERTEX_PROJECT_ID = "backable-machine-learning-apis"
VERTEX_LOCATION = "us-central1"
USE_VERTEX_AI = True  # Primary method
```

### 3. Keep Existing API Keys as Fallback
Keep your existing `GEMINI_API_KEYS` array as fallback when Vertex AI is unavailable.

### 4. Add Vertex AI Initialization Function
Add the `initialize_vertex_ai_client()` function from `vertex_ai_update.py`

### 5. Add Vertex AI Request Function
Add the `try_vertex_ai_projection_request()` function to handle Vertex AI requests

### 6. Modify Main Projection Generation
Update your existing projection generation to:
1. Try Vertex AI first
2. Fall back to API keys if Vertex AI fails
3. Track which method was used

### 7. Environment Setup

#### For Local Development:
Create a `vertex-key.json` file with your service account credentials:
```json
{
  "type": "service_account",
  "project_id": "backable-machine-learning-apis",
  "private_key_id": "...",
  "private_key": "...",
  "client_email": "...",
  "client_id": "...",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "...",
  "client_x509_cert_url": "..."
}
```

#### For Azure Deployment:
Set environment variable:
```bash
GOOGLE_APPLICATION_CREDENTIALS_JSON='{"type":"service_account",...}'
```

### 8. Update Configuration
The Vertex AI method uses slightly different config structure:
- Vertex AI: Direct config dict
- API Keys: `types.GenerateContentConfig` object

### 9. Token Usage Tracking
Vertex AI provides detailed token usage metadata:
- Input tokens
- Output tokens
- Thinking tokens (for reasoning)
- Total tokens

## Testing

1. **Test Vertex AI Connection**:
```python
vertex_ai_client = initialize_vertex_ai_client()
if vertex_ai_client:
    print("✅ Vertex AI connected")
else:
    print("❌ Vertex AI not available, will use fallback")
```

2. **Test Projection Generation**:
- Should try Vertex AI first
- Should fall back to API keys if needed
- Should track which method was used

## Monitoring

Log entries will show:
- `🚀 Trying Vertex AI (Primary Method for Financial Projections)` - Vertex AI attempt
- `✅ Vertex AI request successful` - Vertex AI worked
- `❌ Vertex AI request failed: ... - Falling back to API keys` - Fallback triggered
- `🔄 Attempt 1/3 - Using API key fallback` - Using API keys

## Cost Comparison

### Vertex AI (Gemini 2.5 Pro):
- Input: $0.00125 per 1K tokens
- Output: $0.00375 per 1K tokens
- Thinking tokens: Same as input pricing

### API Keys (Gemini 2.5 Pro):
- Free tier: 2M tokens/month
- After free tier: Similar pricing to Vertex AI

## Advantages After Migration

1. **No More API Key Rotation**: Single service account handles all requests
2. **Better Error Handling**: Automatic retries at infrastructure level
3. **Improved Monitoring**: Cloud Console metrics and logs
4. **Higher Throughput**: No rate limiting between keys
5. **Future Ready**: Easy to upgrade to newer models (Gemini 2.0, etc.)

## Files to Update

1. Main file: `BACKABLE NEW INFRASTRUCTURE FINANCIAL PROJECTION.py`
2. Add the functions from: `vertex_ai_update.py`
3. Create: `vertex-key.json` (for local) or set env variable (for Azure)

## Rollback Plan

If issues occur, simply set:
```python
USE_VERTEX_AI = False
```
This will use API keys only, maintaining current functionality.