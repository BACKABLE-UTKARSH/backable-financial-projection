# Financial Projection Engine - Enhanced Version 3.3.0

## 🎯 Overview
The Financial Projection Engine generates AI-powered 5-year financial projections for businesses using advanced reasoning capabilities from Google's Gemini 2.5 Pro model.

## ✨ Key Features

### 🚀 Version 3.3.0 Updates
- **Vertex AI Integration**: Primary method for AI generation with enterprise-grade reliability
- **New Infrastructure**: Connected to unified database and blob storage architecture
- **Dynamic Resource Lookup**: Container and folder names fetched from database
- **Environment Variables**: All credentials secured via environment variables
- **Fallback System**: 10 Gemini API keys for high availability
- **Enhanced Security**: GitHub-safe configuration (no hardcoded credentials)

### 💼 Business Capabilities
- 5-year financial projections with monthly granularity
- Revenue forecasting with multiple growth scenarios
- Expense projection with seasonal patterns
- Cash flow analysis and burn rate calculations
- Profitability metrics and break-even analysis
- Working capital and capital expenditure planning

### 🔬 Technical Capabilities
- Advanced reasoning with 32,768 thinking budget tokens
- Deterministic generation (temperature = 0.0)
- Sequential queue management for consistent results
- Multi-database architecture (onboarding, finance, projections)
- Comprehensive logging and monitoring
- Local storage for analysis and debugging

---

## 📁 Project Structure

```
BACKABLE NEW INFRASTRUCTURE FINANCIAL PROJECTION/
│
├── BACKABLE NEW INFRASTRUCTURE FINANCIAL PROJECTION.py  # Main application
├── requirements.txt                                      # Python dependencies
├── .env                                                  # Credentials (NOT in git)
├── .env.example                                          # Template for .env
├── .gitignore                                           # Git exclusion rules
├── verify_env.py                                        # Environment verification
│
├── README.md                                            # This file
├── ENVIRONMENT_SETUP.md                                 # Setup instructions
├── MIGRATION_SUMMARY.md                                 # Migration details
└── VERTEX_AI_MIGRATION_GUIDE.md                        # Vertex AI guide
```

---

## 🚦 Quick Start

### Prerequisites
- Python 3.9+
- PostgreSQL database access
- Azure Storage account
- Google Cloud Platform account (Vertex AI)
- Gemini API keys (optional, for fallback)

### 1. Clone and Setup
```bash
cd "C:\Users\tkrot\Desktop\BACKABLE AI NEW INFRASTRUCTURE DEPLOYMENTS\BACKABLE NEW INFRASTRUCTURE FINANCIAL PROJECTION"
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
# Copy template
cp .env.example .env

# Edit .env with your credentials
# See ENVIRONMENT_SETUP.md for detailed instructions
```

### 3. Verify Configuration
```bash
python verify_env.py
```

### 4. Run Application
```bash
python "BACKABLE NEW INFRASTRUCTURE FINANCIAL PROJECTION.py"
```

### 5. Test Endpoint
```bash
curl http://localhost:8000/health
```

---

## 🔧 Configuration

### Environment Variables

#### Required
```env
VERTEX_PROJECT_ID=backable-machine-learning-apis
VERTEX_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS_JSON={"type":"service_account",...}
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;...
ONBOARDING_DB_PASSWORD=your-password
FINANCE_DB_PASSWORD=your-password
PROJECTIONS_DB_PASSWORD=your-password
```

#### Optional (Fallback)
```env
GEMINI_API_KEY_1=AIzaSy...
GEMINI_API_KEY_2=AIzaSy...
# ... up to GEMINI_API_KEY_10
```

See [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) for complete configuration guide.

---

## 🏗️ Architecture

### AI Generation Flow
```
Client Request
    ↓
Sequential Queue System
    ↓
Fetch Financial Data (Azure Blob Storage)
    ↓
[1] Try Vertex AI (Primary)
    ├─ Success → Return Projection
    └─ Fail → [2] Fallback to API Keys
        ├─ Rotate through 10 keys
        ├─ Success → Return Projection
        └─ Fail → Return Error
```

### Database Architecture
```
1. BACKABLE-GOOGLE-RAG (Onboarding)
   - client_onboarding table
   - azure_container_name column
   - folder_name column

2. BACKABLE-FINANCE-ENGINE (Finance Data)
   - Financial intelligence reports metadata

3. BACKABLE-FINANCIAL-PROJECTION (Projections Cache)
   - Cached projections for faster retrieval
```

### Blob Storage Structure
```
unified-clients-prod/
├── client_315/
│   └── financial intelligence report/
│       ├── report_20250101.docx
│       └── report_20250115.docx
├── client_391/
│   └── financial intelligence report/
│       └── report_20250110.docx
└── ...

unified-clients-prod-2/
├── client_500/
│   └── financial intelligence report/
│       └── report_20250105.docx
└── ...
```

---

## 📊 API Endpoints

### Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "3.3.0",
  "vertex_ai_enabled": true,
  "gemini_api_keys_available": 10,
  "timestamp": "2026-01-21T10:30:00Z"
}
```

### Generate Projection
```
POST /generate-projection
Content-Type: application/json

{
  "client_id": "315",
  "years": 5
}
```

**Response:**
```json
{
  "status": "success",
  "client_id": "315",
  "projection": {
    "revenue_projections": [...],
    "expense_projections": [...],
    "cash_flow": [...],
    "profitability_metrics": {...}
  },
  "metadata": {
    "generated_at": "2026-01-21T10:30:00Z",
    "method": "vertex_ai",
    "tokens_used": 15240
  }
}
```

### Queue Status
```
GET /queue-status
```

**Response:**
```json
{
  "queue_length": 3,
  "currently_processing": "client_315",
  "total_processed": 127,
  "average_processing_time": 42.5
}
```

---

## 🔐 Security

### What's Protected
- ✅ Vertex AI service account credentials
- ✅ Gemini API keys (10 keys)
- ✅ Azure Storage connection string
- ✅ Database passwords (3 databases)
- ✅ All sensitive configuration

### Security Best Practices
1. **Never commit `.env` file** - Already in `.gitignore`
2. **Use GitHub Secrets** - For Azure deployments
3. **Rotate API keys regularly** - Update in environment variables
4. **Use service accounts** - For Vertex AI access
5. **Monitor access logs** - Check for unauthorized usage

### Verifying Security
```bash
# Check .gitignore
cat .gitignore | grep .env

# Verify no credentials in code
grep -r "AIzaSy" *.py     # Should return nothing
grep -r "AccountKey=" *.py  # Should return nothing
```

---

## 🧪 Testing

### Run Environment Verification
```bash
python verify_env.py
```

### Test Database Connections
```bash
python verify_env.py
# When prompted, type 'y' to test database connections
```

### Test Vertex AI Connection
```python
from google.oauth2 import service_account
import json
import os
from dotenv import load_dotenv

load_dotenv()
creds_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
creds_dict = json.loads(creds_json)
print(f"✅ Project: {creds_dict.get('project_id')}")
print(f"✅ Service Account: {creds_dict.get('client_email')}")
```

### Integration Testing
```bash
# Start the server
python "BACKABLE NEW INFRASTRUCTURE FINANCIAL PROJECTION.py"

# In another terminal, test the endpoint
curl -X POST http://localhost:8000/generate-projection \
  -H "Content-Type: application/json" \
  -d '{"client_id": "315", "years": 5}'
```

---

## 📈 Performance

### Expected Metrics
- **Response Time**: 30-45 seconds (Vertex AI), 40-50 seconds (API keys)
- **Token Usage**: 11,000-18,000 tokens per projection
- **Success Rate**: ~99% (Vertex AI), ~95% (API keys)
- **Queue Throughput**: ~1.5 projections per minute

### Optimization Tips
1. **Use Vertex AI** - 15% faster than API keys
2. **Cache Results** - Store in PROJECTIONS database
3. **Batch Requests** - Queue system handles sequential processing
4. **Monitor Quotas** - Track Vertex AI and API key limits

---

## 📋 Monitoring

### Application Logs
```bash
# Local development
# Logs appear in console

# Azure deployment
# Go to: Azure Portal → App Service → Log Stream
```

### Key Log Messages
- `✅ Vertex AI initialized successfully` - Vertex AI ready
- `🚀 Trying Vertex AI (Primary Method)` - Starting generation
- `✅ Vertex AI request successful` - Completed via Vertex AI
- `❌ Vertex AI failed ... Falling back` - Using API keys
- `🧮 Token Usage - Input: X | Output: Y` - Token consumption

### Google Cloud Monitoring
1. Go to Google Cloud Console
2. Navigate to Vertex AI
3. View metrics: requests, latency, errors
4. Set up alerts for quota limits

---

## 🐛 Troubleshooting

### Issue: "No Gemini API keys found"
**Cause**: Environment variables not set
**Solution**:
```bash
# Verify .env file exists
ls -la .env

# Check if keys are loaded
python verify_env.py

# Set keys in .env
GEMINI_API_KEY_1=your-key-here
```

### Issue: "Vertex AI initialization failed"
**Cause**: Invalid service account credentials
**Solution**:
1. Verify `GOOGLE_APPLICATION_CREDENTIALS_JSON` is valid JSON
2. Check service account has Vertex AI permissions
3. Ensure project ID matches your GCP project

### Issue: "Database connection failed"
**Cause**: Incorrect credentials or firewall rules
**Solution**:
```bash
# Test with verify_env.py
python verify_env.py

# Check Azure firewall rules
# Go to: Azure Portal → PostgreSQL → Networking
# Add your IP address
```

### Issue: "Azure Blob storage access denied"
**Cause**: Invalid connection string
**Solution**:
1. Go to Azure Portal → Storage Account → Access Keys
2. Copy new connection string
3. Update `.env` file
4. Restart application

---

## 📚 Documentation

### Complete Documentation Set
1. **[README.md](README.md)** - This file, project overview
2. **[ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)** - Detailed setup guide
3. **[MIGRATION_SUMMARY.md](MIGRATION_SUMMARY.md)** - Migration details
4. **[VERTEX_AI_MIGRATION_GUIDE.md](VERTEX_AI_MIGRATION_GUIDE.md)** - Vertex AI guide

### Additional Resources
- [Google Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [Azure Blob Storage Documentation](https://docs.microsoft.com/en-us/azure/storage/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

## 🚀 Deployment

### Local Development
```bash
python "BACKABLE NEW INFRASTRUCTURE FINANCIAL PROJECTION.py"
```

### Azure App Service
1. Configure GitHub Secrets (see ENVIRONMENT_SETUP.md)
2. Push to GitHub repository
3. GitHub Actions will deploy automatically
4. Verify in Azure Portal

### Docker (Optional)
```bash
# Build image
docker build -t financial-projection-engine .

# Run container
docker run -p 8000:8000 --env-file .env financial-projection-engine
```

---

## 🤝 Contributing

### Development Workflow
1. Create feature branch
2. Make changes
3. Test locally with `verify_env.py`
4. Commit (ensure no credentials in code)
5. Push and create PR

### Code Standards
- Use environment variables for all credentials
- Add comprehensive logging
- Update documentation
- Write tests for new features
- Follow existing code style

---

## 📞 Support

### Getting Help
1. Check documentation in this folder
2. Review logs for error messages
3. Run `verify_env.py` to diagnose issues
4. Contact Backable development team

### Reporting Issues
Include in your report:
- Error messages from logs
- Output from `verify_env.py`
- Steps to reproduce
- Expected vs actual behavior

---

## 📜 License

© 2026 Backable. All rights reserved.

---

## 🎉 Changelog

### Version 3.3.0 (2026-01-21)
- ✅ Added Vertex AI integration (primary method)
- ✅ Connected to new unified infrastructure
- ✅ Implemented dynamic container/folder lookup
- ✅ Migrated all credentials to environment variables
- ✅ Added comprehensive documentation
- ✅ Created security best practices
- ✅ GitHub-safe configuration

### Version 3.2.0 (Previous)
- Sequential queue system
- Multi-database support
- Local storage for debugging
- API key rotation

---

**Ready for Production** ✨

For detailed setup instructions, see [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)
