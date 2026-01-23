from fastapi import FastAPI, HTTPException, Query, Depends, Header, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from google.oauth2 import service_account  # Added for Vertex AI
from azure.storage.blob import BlobServiceClient
import pathlib
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict
from datetime import datetime, timedelta, timezone
import json
import io
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
import mammoth
import os
import random
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from collections import defaultdict
from threading import Lock
import tempfile  # Added for Vertex AI
import threading  # Added for Vertex AI
from dotenv import load_dotenv  # Added for Vertex AI

# Load environment variables
load_dotenv()

# Set random seed for reproducible results
random.seed(42)

# GLOBAL SEQUENTIAL QUEUE SYSTEM
# Single global queue - only 1 client processes at a time
global_queue = asyncio.Queue()
currently_processing_client = None
queue_lock = Lock()

# Queue statistics
queue_stats = {
    "total_queued": 0,
    "total_processed": 0,
    "currently_processing": 0,
    "queue_length": 0
}
import re
import traceback
from pathlib import Path
import sys
import logging
import jwt
import hashlib

def get_projection_start_date():
    """Get the start date for projections (next January from current date)"""
    current_date = datetime.now()
    # Always start from next January
    start_year = current_date.year + 1
    return datetime(start_year, 1, 1)

def get_dynamic_year_range(years):
    """Get dynamic year range for projections starting from next January"""
    start_date = get_projection_start_date()
    base_year = start_date.year
    target_year = base_year + years - 1
    return base_year, target_year

# GLOBAL SEQUENTIAL QUEUE MANAGEMENT FUNCTIONS
async def queue_projection_request(client_id: str, request_type: str, request_func):
    """
    Global sequential queue system - only 1 client processes at a time
    Returns: (is_queued: bool, result: dict)
    """
    global currently_processing_client

    with queue_lock:
        queue_stats["total_queued"] += 1

        # Check if ANY client is currently processing
        if currently_processing_client is not None:
            logger.info(f"🔄 QUEUING REQUEST: System busy with {currently_processing_client}. Queuing {client_id} - {request_type}")

            # Create a future for the queued request
            future = asyncio.Future()
            queue_item = {
                "client_id": client_id,
                "request_type": request_type,
                "request_func": request_func,
                "future": future,
                "queued_at": datetime.now().isoformat()
            }

            await global_queue.put(queue_item)
            queue_stats["queue_length"] = global_queue.qsize()

            queue_info = {
                "status": "queued",
                "client_id": client_id,
                "request_type": request_type,
                "message": f"Request queued. System is processing {currently_processing_client}. Your position: {global_queue.qsize()}",
                "queue_position": global_queue.qsize(),
                "estimated_wait_time": f"{global_queue.qsize() * 45} seconds",
                "currently_processing": currently_processing_client
            }
            save_to_local_storage(queue_info, 'queue_logs', 'request_queued', client_id)

            return True, queue_info
        else:
            # Mark this client as processing (system was free)
            currently_processing_client = client_id
            queue_stats["currently_processing"] = 1

            logger.info(f"🚀 PROCESSING IMMEDIATELY: {client_id} - {request_type} (System was free)")
            return False, None

async def process_global_queue():
    """Process the next queued request from global queue"""
    global currently_processing_client

    try:
        while not global_queue.empty():
            queued_item = await global_queue.get()
            client_id = queued_item["client_id"]

            logger.info(f"🔄 PROCESSING NEXT IN QUEUE: {client_id} - {queued_item['request_type']}")

            # Update processing client
            with queue_lock:
                currently_processing_client = client_id
                queue_stats["queue_length"] = global_queue.qsize()

            try:
                # Execute the queued request
                result = await queued_item["request_func"]()
                queued_item["future"].set_result(result)

                queue_stats["total_processed"] += 1

                processed_info = {
                    "client_id": client_id,
                    "request_type": queued_item["request_type"],
                    "queued_at": queued_item["queued_at"],
                    "processed_at": datetime.now().isoformat(),
                    "status": "completed"
                }
                save_to_local_storage(processed_info, 'queue_logs', 'request_processed', client_id)

            except Exception as e:
                logger.error(f"❌ QUEUED REQUEST FAILED: {client_id}, Error: {str(e)}")
                queued_item["future"].set_exception(e)

    finally:
        # Clear processing state - system is now free
        with queue_lock:
            currently_processing_client = None
            queue_stats["currently_processing"] = 0
            queue_stats["queue_length"] = global_queue.qsize()

        logger.info(f"✅ SYSTEM NOW FREE - All queued requests processed")

# Configure enhanced logging with detailed formatting
# Configure enhanced logging with detailed formatting and UTF-8 encoding
# Configure logging - simple and reliable
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced Financial Projection API with Vertex AI Support", version="3.3.0")

# Add CORS middleware to handle cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins including local file (null)
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════════
# JWT AUTHENTICATION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
JWT_SECRET = os.getenv("JWT_SECRET", "philotimo-global-jwt-secret-2024!!")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

# Philotimo database configuration for token validation
PHILOTIMO_DB_CONFIG = {
    "host": "philotimo-staging-db.postgres.database.azure.com",
    "database": "philotimodb",
    "user": "wchen",
    "password": "DevPhilot2024!!",
    "port": 5432,
    "sslmode": "require"
}

def hash_token(token: str) -> str:
    """Hash a JWT token using SHA256 for database comparison"""
    return hashlib.sha256(token.encode()).hexdigest()

async def verify_jwt_token(authorization: str = Header(None)) -> Dict:
    """
    Verify JWT token from Authorization header and validate against database
    Returns user information if token is valid
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header"
        )

    # Extract Bearer token
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme"
            )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format"
        )

    # Decode and validate JWT
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        jti = payload.get("jti")

        if not jti:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing JTI"
            )

        # Hash the token for database lookup
        token_hash = hash_token(token)

        # Validate token against database
        conn = psycopg2.connect(**PHILOTIMO_DB_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        try:
            cursor.execute("""
                SELECT
                    t.user_id,
                    t.is_revoked AS revoked,
                    t.expires_at,
                    u.client_id,
                    u.email
                FROM api_tokens t
                JOIN users u ON t.user_id = u.id
                WHERE t.jti = %s AND t.token_hash = %s
            """, (jti, token_hash))

            result = cursor.fetchone()

            if not result:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token"
                )

            if result['revoked']:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )

            if result['expires_at'] and result['expires_at'] < datetime.now(timezone.utc):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired"
                )

            return {
                "user_id": result['user_id'],
                "client_id": result['client_id'],
                "email": result['email']
            }

        finally:
            cursor.close()
            conn.close()

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}"
        )
    except Exception as e:
        error_details = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Token validation error: {error_details}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Token validation failed: {error_details}"
        )

# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL STORAGE CONFIGURATION - SAVES EVERYTHING IN CURRENT DIRECTORY
# ═══════════════════════════════════════════════════════════════════════════════
LOCAL_STORAGE_ENABLED = True
LOCAL_STORAGE_BASE_PATH = "./local_backend_storage"  # Current directory storage
DETAILED_LOGGING_ENABLED = True

# Create subdirectories for organized storage
STORAGE_SUBDIRS = {
    'requests': 'requests',
    'database_queries': 'database_queries', 
    'financial_data': 'financial_data',
    'prompts': 'prompts',
    'api_responses': 'api_responses',
    'final_results': 'final_results',
    'errors': 'errors',
    'system_logs': 'system_logs',
    'health_checks': 'health_checks',
    'calculations': 'calculations'
}

def ensure_storage_directories():
    """Create all necessary storage directories"""
    if not LOCAL_STORAGE_ENABLED:
        return
        
    try:
        # Create base directory
        Path(LOCAL_STORAGE_BASE_PATH).mkdir(exist_ok=True)
        logger.info(f"✓ Base storage directory ensured: {LOCAL_STORAGE_BASE_PATH}")
        
        # Create subdirectories
        for category, subdir in STORAGE_SUBDIRS.items():
            full_path = Path(LOCAL_STORAGE_BASE_PATH) / subdir
            full_path.mkdir(exist_ok=True)
            logger.debug(f"✓ Storage subdirectory created: {full_path}")
            
        logger.info(f"✓ All storage directories initialized successfully")
        
    except Exception as e:
        logger.error(f"✗ Failed to create storage directories: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

def save_to_local_storage(data, category, filename_prefix, client_id=None, additional_info=None):
    """
    Enhanced local storage function with detailed logging and error handling
    """
    if not LOCAL_STORAGE_ENABLED:
        logger.debug("Local storage disabled, skipping save operation")
        return None
        
    try:
        ensure_storage_directories()
        
        # Generate detailed filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        client_suffix = f"_client_{client_id}" if client_id else ""
        filename = f"{filename_prefix}{client_suffix}_{timestamp}.json"
        
        # Determine storage path based on category
        if category not in STORAGE_SUBDIRS:
            logger.warning(f"Unknown storage category '{category}', using 'system_logs'")
            category = 'system_logs'
            
        storage_dir = Path(LOCAL_STORAGE_BASE_PATH) / STORAGE_SUBDIRS[category]
        filepath = storage_dir / filename
        
        # Prepare enhanced data structure
        storage_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "client_id": client_id,
                "category": category,
                "filename_prefix": filename_prefix,
                "data_type": type(data).__name__,
                "additional_info": additional_info,
                "file_size_estimate": len(str(data)) if data else 0
            },
            "content": data
        }
        
        # Handle different data types  
        if hasattr(data, 'model_dump'):
            storage_data["content"] = data.model_dump()
        elif hasattr(data, 'dict'):
            storage_data["content"] = data.dict()
        elif isinstance(data, (dict, list, str, int, float, bool)):
            storage_data["content"] = data
        else:
            storage_data["content"] = str(data)
            
        # Write to file with error handling
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(storage_data, f, indent=2, default=str, ensure_ascii=False)
        
        file_size = filepath.stat().st_size
        logger.info(f"✓ STORAGE SUCCESS: {filepath} ({file_size:,} bytes)")
        
        if DETAILED_LOGGING_ENABLED:
            logger.debug(f"Storage details - Category: {category}, Client: {client_id}, Size: {file_size:,} bytes")
            
        return str(filepath)
        
    except Exception as e:
        logger.error(f"✗ STORAGE FAILED: Category={category}, Prefix={filename_prefix}, Client={client_id}")
        logger.error(f"Storage error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Try to save error information
        try:
            error_data = {
                "storage_error": str(e),
                "category": category,
                "filename_prefix": filename_prefix,
                "client_id": client_id,
                "timestamp": datetime.now().isoformat()
            }
            error_filepath = Path(LOCAL_STORAGE_BASE_PATH) / STORAGE_SUBDIRS['errors'] / f"storage_error_{timestamp}.json"
            with open(error_filepath, 'w') as f:
                json.dump(error_data, f, indent=2, default=str)
            logger.info(f"✓ Error details saved to: {error_filepath}")
        except:
            logger.error("✗ Failed to save storage error details")
            
        return None

def log_and_store_system_event(event_type, event_data, client_id=None):
    """Log and store system events with detailed information"""
    logger.info(f"SYSTEM EVENT: {event_type} - Client: {client_id}")
    
    system_event = {
        "event_type": event_type,
        "event_data": event_data,
        "client_id": client_id,
        "timestamp": datetime.now().isoformat()
    }
    
    save_to_local_storage(
        system_event, 
        'system_logs', 
        f"system_{event_type.lower()}", 
        client_id
    )

# Azure Storage Configuration - Updated for new infrastructure
AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
if not AZURE_STORAGE_CONNECTION_STRING:
    logger.warning("⚠️ AZURE_STORAGE_CONNECTION_STRING not found in environment variables!")
    logger.warning("⚠️ Azure blob storage features will be disabled")

# Database Configuration - Updated for new Google infrastructure
ONBOARDING_DB_HOST = os.getenv("ONBOARDING_DB_HOST")
ONBOARDING_DB_NAME = os.getenv("ONBOARDING_DB_NAME")
ONBOARDING_DB_USER = os.getenv("ONBOARDING_DB_USER")
ONBOARDING_DB_PASSWORD = os.getenv('ONBOARDING_DB_PASSWORD')
ONBOARDING_DB_PORT = int(os.getenv("ONBOARDING_DB_PORT", "5432"))

# Validate onboarding database credentials
if not all([ONBOARDING_DB_HOST, ONBOARDING_DB_NAME, ONBOARDING_DB_USER, ONBOARDING_DB_PASSWORD]):
    logger.error("⚠️ CRITICAL: Onboarding database credentials not found in environment variables!")
    logger.error("⚠️ Please set ONBOARDING_DB_HOST, ONBOARDING_DB_NAME, ONBOARDING_DB_USER, and ONBOARDING_DB_PASSWORD")

# Database Configuration - Updated for Finance Engine
FINANCE_DB_HOST = os.getenv("FINANCE_DB_HOST")
FINANCE_DB_NAME = os.getenv("FINANCE_DB_NAME")
FINANCE_DB_USER = os.getenv("FINANCE_DB_USER")
FINANCE_DB_PASSWORD = os.getenv('FINANCE_DB_PASSWORD')
FINANCE_DB_PORT = int(os.getenv("FINANCE_DB_PORT", "5432"))

# Validate finance database credentials
if not all([FINANCE_DB_HOST, FINANCE_DB_NAME, FINANCE_DB_USER, FINANCE_DB_PASSWORD]):
    logger.error("⚠️ CRITICAL: Finance database credentials not found in environment variables!")
    logger.error("⚠️ Please set FINANCE_DB_HOST, FINANCE_DB_NAME, FINANCE_DB_USER, and FINANCE_DB_PASSWORD")

# New Projections Cache Database
PROJECTIONS_DB_HOST = os.getenv("PROJECTIONS_DB_HOST")
PROJECTIONS_DB_NAME = os.getenv("PROJECTIONS_DB_NAME")
PROJECTIONS_DB_USER = os.getenv("PROJECTIONS_DB_USER")
PROJECTIONS_DB_PASSWORD = os.getenv('PROJECTIONS_DB_PASSWORD')
PROJECTIONS_DB_PORT = int(os.getenv("PROJECTIONS_DB_PORT", "5432"))

# Validate projections database credentials
if not all([PROJECTIONS_DB_HOST, PROJECTIONS_DB_NAME, PROJECTIONS_DB_USER, PROJECTIONS_DB_PASSWORD]):
    logger.error("⚠️ CRITICAL: Projections database credentials not found in environment variables!")
    logger.error("⚠️ Please set PROJECTIONS_DB_HOST, PROJECTIONS_DB_NAME, PROJECTIONS_DB_USER, and PROJECTIONS_DB_PASSWORD")

# Google Gemini API Configuration - Load from environment variables
GEMINI_API_KEYS = []
for i in range(1, 11):  # Try to load up to 10 API keys
    key = os.getenv(f'GEMINI_API_KEY_{i}')
    if key:
        GEMINI_API_KEYS.append(key)

# Validate that API keys are loaded
if not GEMINI_API_KEYS:
    logger.error("⚠️ CRITICAL: No Gemini API keys found in environment variables!")
    logger.error("⚠️ Please set GEMINI_API_KEY_1 through GEMINI_API_KEY_10 in your .env file")
    logger.error("⚠️ Using Vertex AI only - API key fallback unavailable")

# ======================================================
#           Vertex AI Configuration (Primary Method)
# ======================================================
VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", "backable-machine-learning-apis")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
USE_VERTEX_AI = True  # Primary method - will fallback to API keys if fails

# API Key Management Variables (for fallback)
api_key_stats = defaultdict(lambda: {"requests": 0, "failures": 0, "last_used": 0, "cooldown_until": 0})
api_key_lock = threading.Lock()

# API Key rotation index (for fallback)
current_key_index = 0

# Global Vertex AI client
vertex_ai_client = None

def get_next_api_key():
    """Get next API key in rotation with detailed logging"""
    global current_key_index
    api_key = GEMINI_API_KEYS[current_key_index]
    old_index = current_key_index
    current_key_index = (current_key_index + 1) % len(GEMINI_API_KEYS)
    
    logger.debug(f"API Key rotation: Index {old_index} -> {current_key_index}")
    logger.debug(f"Selected API key: {api_key[:20]}...")
    
    # Store API key selection event
    key_selection_data = {
        "old_index": old_index,
        "new_index": current_key_index,
        "selected_key_preview": api_key[:20] + "...",
        "total_keys_available": len(GEMINI_API_KEYS)
    }
    save_to_local_storage(key_selection_data, 'system_logs', 'api_key_selection')
    
    return api_key

def get_deterministic_api_key(client_id: str = "default"):
    """Get deterministic API key based on client ID for consistent results"""
    # Use hash of client_id to deterministically select same API key every time
    selected_index = hash(client_id) % len(GEMINI_API_KEYS)
    selected_key = GEMINI_API_KEYS[selected_index]
    
    logger.debug(f"Deterministic API key selected: Index {selected_index}, Key: {selected_key[:20]}...")

    # Store deterministic key selection
    deterministic_key_data = {
        "selected_index": selected_index,
        "selected_key_preview": selected_key[:20] + "...",
        "selection_method": "deterministic",
        "client_id": client_id
    }
    save_to_local_storage(deterministic_key_data, 'system_logs', 'deterministic_api_key_selection')
    
    return selected_key

def create_gemini_client(api_key=None, client_id="default"):
    """Create a new Gemini client with detailed logging"""
    if api_key is None:
        api_key = get_deterministic_api_key(client_id)
    
    logger.info(f"Creating Gemini client with API key: {api_key[:20]}...")
    
    client_creation_data = {
        "api_key_preview": api_key[:20] + "...",
        "creation_timestamp": datetime.now().isoformat()
    }
    
    try:
        client = genai.Client(api_key=api_key)
        logger.info("✓ Gemini client created successfully")
        
        client_creation_data["status"] = "success"
        save_to_local_storage(client_creation_data, 'system_logs', 'gemini_client_creation')
        
        return client
    except Exception as e:
        logger.error(f"✗ Failed to create Gemini client: {str(e)}")
        
        client_creation_data["status"] = "failed"
        client_creation_data["error"] = str(e)
        save_to_local_storage(client_creation_data, 'errors', 'gemini_client_creation_error')
        
        raise

def initialize_vertex_ai_client():
    """
    Initialize Google GenAI client for Vertex AI.
    Supports both file-based and environment variable credentials.
    Returns None if initialization fails (will use API keys fallback).
    """
    try:
        # Try loading credentials from environment variable first (Azure deployment)
        creds_json = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')

        if creds_json:
            logger.info("Loading Vertex AI credentials from environment variable")
            creds_dict = json.loads(creds_json)
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
                json.dump(creds_dict, temp_file)
                temp_path = temp_file.name

            credentials = service_account.Credentials.from_service_account_file(
                temp_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            os.unlink(temp_path)
        else:
            # Fall back to file-based credentials (local development)
            creds_file = "vertex-key.json"
            if os.path.exists(creds_file):
                logger.info(f"Loading Vertex AI credentials from {creds_file}")
                credentials = service_account.Credentials.from_service_account_file(
                    creds_file,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
            else:
                logger.warning("No Vertex AI credentials found - will use API keys fallback")
                return None

        # Initialize GenAI client for Vertex AI
        client = genai.Client(
            vertexai=True,
            credentials=credentials,
            project=VERTEX_PROJECT_ID,
            location=VERTEX_LOCATION
        )

        logger.info(f"✅ Vertex AI GenAI client initialized successfully (Project: {VERTEX_PROJECT_ID})")
        return client

    except Exception as e:
        logger.warning(f"⚠️ Vertex AI initialization failed: {str(e)} - Will use API keys fallback")
        return None

# Initialize Vertex AI client at startup
logger.info("Attempting to initialize Vertex AI client...")
vertex_ai_client = initialize_vertex_ai_client() if USE_VERTEX_AI else None

# Initialize regular Gemini client as fallback
logger.info("Initializing Gemini client as fallback...")
client = create_gemini_client()

# Pydantic Models (keeping existing models)
class GrowthRateAssumptions(BaseModel):
    """Growth rate assumptions with fixed fields"""
    revenue_cagr: float = Field(description="Revenue compound annual growth rate")
    expense_inflation: float = Field(description="Expense inflation rate")
    profit_margin_target: float = Field(description="Target profit margin")

class KeyFinancialRatios(BaseModel):
    """Key financial ratios with fixed fields"""
    gross_margin: float = Field(description="Gross profit margin")
    net_margin: float = Field(description="Net profit margin")
    current_ratio: float = Field(description="Current assets / Current liabilities")
    debt_to_equity: float = Field(description="Total debt / Total equity")

class HistoricalBaseline(BaseModel):
    """Current historical performance baseline - MUST be consistent across all timeframes"""
    current_annual_revenue: float = Field(description="Most recent 12-month revenue total")
    current_annual_expenses: float = Field(description="Most recent 12-month operating expenses total")
    current_annual_net_profit: float = Field(description="Most recent 12-month net profit total")
    current_annual_gross_profit: float = Field(description="Most recent 12-month gross profit total")
    baseline_period: str = Field(description="Period this baseline represents (e.g., 'Last 12 months ending June 2025')")
    revenue_growth_rate: float = Field(description="Actual historical revenue growth rate used for projections")
    expense_inflation_rate: float = Field(description="Actual expense growth rate from historical data")
    profit_margin_trend: float = Field(description="Historical profit margin trend")

class MonthlyProjection(BaseModel):
    """Monthly projection data structure"""
    month: str = Field(description="Format: YYYY-MM (starting from next January, e.g., 2026-01, 2026-02, etc.)")
    revenue: float
    net_profit: float
    gross_profit: float
    expenses: float

class QuarterlyProjection(BaseModel):
    """Quarterly projection data structure"""
    quarter: str = Field(description="Format: YYYY-QN (starting from next January, e.g., 2026-Q1, 2026-Q2, etc.)")
    revenue: float
    net_profit: float
    gross_profit: float
    expenses: float

class AnnualProjection(BaseModel):
    """Annual projection data structure"""
    year: int
    revenue: float
    net_profit: float
    gross_profit: float
    expenses: float

class ProjectionsData(BaseModel):
    """Comprehensive projections data structure"""
    one_year_monthly: List[MonthlyProjection] = Field(description="12 months of monthly data")
    three_years_monthly: List[MonthlyProjection] = Field(description="36 months of monthly data")
    five_years_quarterly: List[QuarterlyProjection] = Field(description="20 quarters of quarterly data")
    ten_years_annual: List[AnnualProjection] = Field(description="10 years of annual data")
    fifteen_years_annual: List[AnnualProjection] = Field(description="15 years of annual data")


class MethodologyDetails(BaseModel):
    """Details about the projection methodology used"""
    forecasting_methods_used: List[str]
    seasonal_adjustments_applied: bool
    trend_analysis_period: str
    growth_rate_assumptions: GrowthRateAssumptions

class QualityScores(BaseModel):
    score: float = Field(description="0.0 to 1.0", ge=0.0, le=1.0)
    rationale: str = Field(description="The 1 sentence explanation for the score given")

class EnhancedProjectionSchema(BaseModel):
    """Enhanced schema for comprehensive financial projections"""
    executive_summary: str
    business_name: str
    completion_score: QualityScores
    data_quality_score: QualityScores
    projection_confidence_score: QualityScores
    historical_baseline: HistoricalBaseline
    projection_drivers_found: List[str]
    assumptions_made: List[str]
    anomalies_found: List[str]
    methodology: MethodologyDetails
    projections_data: ProjectionsData
    key_financial_ratios: KeyFinancialRatios
    risk_factors: List[str]
    recommendations: List[str]

class QueueResponseSchema(BaseModel):
    """Schema for queued request responses"""
    status: str
    client_id: str
    request_type: str
    message: str
    queue_position: int
    estimated_wait_time: str
    currently_processing: str
    from_cache: Optional[bool] = False
    force_regenerated: Optional[bool] = False
    generated_at: Optional[str] = None
    processing_time: Optional[float] = None

class ForceRegeneratedProjectionSchema(EnhancedProjectionSchema):
    """Schema for force-regenerated projection responses with metadata"""
    from_cache: bool = False
    force_regenerated: bool = True
    generated_at: str
    processing_time: float

def get_client_folder_name(user_id: str) -> str:
    """Get client folder name from database for new unified structure"""
    conn = None
    try:
        conn = psycopg2.connect(
            host=ONBOARDING_DB_HOST,
            dbname=ONBOARDING_DB_NAME,
            user=ONBOARDING_DB_USER,
            password=ONBOARDING_DB_PASSWORD,
            port=ONBOARDING_DB_PORT
        )
        with conn.cursor() as cur:
            sql = "SELECT folder_name FROM client_onboarding WHERE client_id = %s"
            cur.execute(sql, (user_id,))
            row = cur.fetchone()

            if row and row[0]:
                logger.info(f"✅ Found folder name: {row[0]} for user_id: {user_id}")
                return row[0]
            else:
                # Fallback to client_{id} format
                folder_name = f"client_{user_id}"
                logger.info(f"⚠️ No folder name found, using fallback: {folder_name}")
                return folder_name
    except Exception as e:
        logger.error(f"Error getting folder name: {str(e)}")
        return f"client_{user_id}"
    finally:
        if conn:
            conn.close()

def get_azure_container_name(user_id: str) -> str:
    """Get Azure container name dynamically from database - supports multiple containers"""
    logger.info(f"🔍 CONTAINER LOOKUP: Getting container for user_id: {user_id}")
    conn = None
    try:
        conn = psycopg2.connect(
            host=ONBOARDING_DB_HOST,
            dbname=ONBOARDING_DB_NAME,
            user=ONBOARDING_DB_USER,
            password=ONBOARDING_DB_PASSWORD,
            port=ONBOARDING_DB_PORT
        )
        with conn.cursor() as cur:
            # Get the azure_container_name which stores the actual container (unified-clients-prod, unified-clients-prod-2, etc.)
            sql = "SELECT azure_container_name FROM client_onboarding WHERE client_id = %s"
            cur.execute(sql, (user_id,))
            row = cur.fetchone()

            if row and row[0]:
                container_name = row[0]
                logger.info(f"✅ Found container: {container_name} for user_id: {user_id}")
                return container_name
            else:
                # Default to unified-clients-prod if not found
                logger.info(f"⚠️ No container found for user_id: {user_id}, using default: unified-clients-prod")
                return "unified-clients-prod"
    except Exception as e:
        logger.error(f"Error getting container name: {str(e)}, using default")
        return "unified-clients-prod"
    finally:
        if conn:
            conn.close()

def get_financial_data_from_database(client_id: str) -> str:
    """
    Fetch and combine all financial data from PostgreSQL with complete logging
    """
    logger.info(f"📊 FINANCIAL DATA RETRIEVAL: Starting for client_id: {client_id}")
    
    # Store the data retrieval request
    data_retrieval_request = {
        "client_id": client_id,
        "operation": "get_financial_data_from_database",
        "timestamp": datetime.now().isoformat(),
        "database": FINANCE_DB_NAME
    }
    save_to_local_storage(data_retrieval_request, 'database_queries', 'financial_data_request', client_id)
    
    conn = None
    try:
        logger.debug("📡 Connecting to Finance PostgreSQL database...")
        conn = psycopg2.connect(
            host=FINANCE_DB_HOST,
            dbname=FINANCE_DB_NAME,
            user=FINANCE_DB_USER,
            password=FINANCE_DB_PASSWORD,
            port=FINANCE_DB_PORT,
            connect_timeout=10
        )
        
        logger.debug("✓ Finance database connection established successfully")
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            sql = """
                SELECT id, original_name, category, 
                       extracted_data, gemini_analysis, financial_ratios, analysis_metadata
                FROM finance_documents 
                WHERE user_id = %s 
                ORDER BY id
            """
            
            logger.debug(f"🔎 Executing financial data query for client_id: {client_id}")
            
            # Store the SQL query details
            query_details = {
                "sql": sql,
                "client_id": client_id,
                "operation": "fetch_finance_documents",
                "database": FINANCE_DB_NAME
            }
            save_to_local_storage(query_details, 'database_queries', 'financial_data_sql', client_id)
            
            cur.execute(sql, (client_id,))
            rows = cur.fetchall()
            
            # Store raw database results immediately
            raw_db_results = {
                "client_id": client_id,
                "total_documents_found": len(rows),
                "query_timestamp": datetime.now().isoformat(),
                "raw_rows": [dict(row) for row in rows] if rows else []
            }
            save_to_local_storage(raw_db_results, 'database_queries', 'raw_db_results', client_id)
            
            if not rows:
                logger.warning(f"⚠️ No financial documents found for client_id={client_id}")
                no_docs_error = {
                    "error_type": "no_documents_found",
                    "client_id": client_id,
                    "message": f"No financial documents found for client_id: {client_id}"
                }
                save_to_local_storage(no_docs_error, 'errors', 'no_financial_documents', client_id)
                raise HTTPException(
                    status_code=404, 
                    detail=f"No financial documents found for client_id: {client_id}"
                )
            
            logger.info(f"📄 Found {len(rows)} financial documents for client_id: {client_id}")
            
            # Combine all data into comprehensive text format
            combined_content = f"=== FINANCIAL DATA ANALYSIS FOR CLIENT {client_id} ===\n\n"
            
            document_processing_log = {
                "client_id": client_id,
                "total_documents": len(rows),
                "processing_start": datetime.now().isoformat(),
                "documents_details": []
            }
            
            for idx, row in enumerate(rows, 1):
                logger.debug(f"📋 Processing document {idx}: ID={row['id']}, Name={row['original_name']}")
                
                doc_details = {
                    "index": idx,
                    "document_id": row['id'],
                    "original_name": row['original_name'],
                    "category": row['category'],
                    "has_extracted_data": bool(row['extracted_data']),
                    "has_gemini_analysis": bool(row['gemini_analysis']),
                    "has_financial_ratios": bool(row['financial_ratios']),
                    "has_analysis_metadata": bool(row['analysis_metadata'])
                }
                document_processing_log["documents_details"].append(doc_details)
                
                # Add document header
                combined_content += f"\n=== DOCUMENT {idx}: {row['original_name']} ===\n"
                combined_content += f"Document ID: {row['id']}\n"
                combined_content += f"Category: {row['category']}\n\n"
                
                # Add extracted data
                if row['extracted_data']:
                    combined_content += "--- EXTRACTED FINANCIAL DATA ---\n"
                    if isinstance(row['extracted_data'], dict):
                        combined_content += json.dumps(row['extracted_data'], indent=2)
                    else:
                        combined_content += str(row['extracted_data'])
                    combined_content += "\n\n"
                
                # Add Gemini analysis
                if row['gemini_analysis']:
                    combined_content += "--- AI ANALYSIS ---\n"
                    if isinstance(row['gemini_analysis'], dict):
                        combined_content += json.dumps(row['gemini_analysis'], indent=2)
                    else:
                        combined_content += str(row['gemini_analysis'])
                    combined_content += "\n\n"
                
                # Add financial ratios
                if row['financial_ratios']:
                    combined_content += "--- FINANCIAL RATIOS ---\n"
                    if isinstance(row['financial_ratios'], dict):
                        combined_content += json.dumps(row['financial_ratios'], indent=2)
                    else:
                        combined_content += str(row['financial_ratios'])
                    combined_content += "\n\n"
                
                # Add analysis metadata
                if row['analysis_metadata']:
                    combined_content += "--- ANALYSIS METADATA ---\n"
                    if isinstance(row['analysis_metadata'], dict):
                        combined_content += json.dumps(row['analysis_metadata'], indent=2)
                    else:
                        combined_content += str(row['analysis_metadata'])
                    combined_content += "\n\n"
                
                combined_content += f"=== END DOCUMENT {idx} ===\n\n"
            
            # Store document processing log
            document_processing_log["processing_end"] = datetime.now().isoformat()
            document_processing_log["combined_content_length"] = len(combined_content)
            save_to_local_storage(document_processing_log, 'financial_data', 'document_processing_log', client_id)
            
            # Store the final combined financial content
            combined_financial_data = {
                "client_id": client_id,
                "total_documents_processed": len(rows),
                "combined_content_length": len(combined_content),
                "combined_content": combined_content,
                "processing_completed": datetime.now().isoformat()
            }
            save_to_local_storage(combined_financial_data, 'financial_data', 'combined_financial_content', client_id)
            
            logger.info(f"✓ Successfully combined {len(rows)} financial documents for client_id: {client_id} (Total: {len(combined_content):,} characters)")
            return combined_content
            
    except HTTPException:
        raise
    except psycopg2.Error as e:
        logger.error(f"💥 Database error retrieving financial data: {str(e)}")
        db_error_data = {
            "error_type": "database_error",
            "client_id": client_id,
            "error_message": str(e),
            "operation": "get_financial_data_from_database",
            "traceback": traceback.format_exc()
        }
        save_to_local_storage(db_error_data, 'errors', 'financial_data_db_error', client_id)
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"💥 Error retrieving financial data: {str(e)}")
        general_error_data = {
            "error_type": "general_error",
            "client_id": client_id,
            "error_message": str(e),
            "operation": "get_financial_data_from_database",
            "traceback": traceback.format_exc()
        }
        save_to_local_storage(general_error_data, 'errors', 'financial_data_general_error', client_id)
        raise HTTPException(status_code=500, detail=f"Error retrieving financial data: {str(e)}")
    
    finally:
        if conn:
            conn.close()
            logger.debug("🔐 Finance database connection closed")

@app.get("/")
async def root():
    logger.info("🏠 Root endpoint accessed")
    
    root_access_data = {
        "endpoint": "/",
        "access_time": datetime.now().isoformat(),
        "message": "Enhanced Financial Projection API v3.3 with Vertex AI Support"
    }
    save_to_local_storage(root_access_data, 'system_logs', 'root_access')

    return {"message": "Enhanced Financial Projection API v3.3 with Vertex AI Support is running"}

# ═══════════════════════════════════════════════════════════════════════════════
# CACHING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_projections_db_connection():
    """Get connection to projections cache database"""
    return psycopg2.connect(
        host=PROJECTIONS_DB_HOST,
        dbname=PROJECTIONS_DB_NAME,
        user=PROJECTIONS_DB_USER,
        password=PROJECTIONS_DB_PASSWORD,
        port=PROJECTIONS_DB_PORT,
        connect_timeout=10
    )

def check_cached_projection(client_id: str, timeframe: str = "1 Year") -> Optional[dict]:
    """Check if we have a cached projection for this client and timeframe"""
    conn = None
    try:
        logger.info(f"🔍 Checking cache for client {client_id}, timeframe {timeframe}")
        conn = get_projections_db_connection()
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            sql = """
            SELECT 
                id, client_id, timeframe, business_name,
                projections_data, methodology, executive_summary,
                total_revenue, total_expenses, total_net_profit, total_gross_profit,
                revenue_cagr, expense_inflation, profit_margin_target,
                completion_score, data_quality_score, projection_confidence_score,
                historical_revenue, historical_expenses, historical_net_profit, historical_gross_profit,
                historical_period, historical_revenue_growth, historical_expense_growth, historical_profit_margin,
                created_at, updated_at, is_cached
            FROM financial_projections 
            WHERE client_id = %s AND timeframe = %s
            ORDER BY created_at DESC 
            LIMIT 1
            """
            
            cur.execute(sql, (client_id, timeframe))
            result = cur.fetchone()
            
            if result:
                logger.info(f"✅ Cache HIT for client {client_id}, timeframe {timeframe}")
                
                # Update last accessed timestamp
                cur.execute(
                    "UPDATE financial_projections SET last_accessed = NOW() WHERE id = %s",
                    (result['id'],)
                )
                conn.commit()
                
                return dict(result)
            else:
                logger.info(f"❌ Cache MISS for client {client_id}, timeframe {timeframe}")
                return None
                
    except Exception as e:
        logger.error(f"🚨 Error checking cache: {str(e)}")
        return None
    finally:
        if conn:
            conn.close()

def save_projection_to_cache(client_id: str, projection_data: dict, processing_duration: float, timeframe: str = "1 Year"):
    """Save a projection to the cache database"""
    conn = None
    try:
        logger.info(f"💾 Saving projection to cache for client {client_id}, timeframe {timeframe}")
        conn = get_projections_db_connection()
        
        # Extract data from projection_data
        projections = projection_data.get('projections_data', {})
        methodology = projection_data.get('methodology', {})
        
        with conn.cursor() as cur:
            # Updated SQL with historical baseline columns
            sql = """
            INSERT INTO financial_projections (
                client_id, timeframe, business_name,
                projections_data, methodology, executive_summary,
                total_revenue, total_expenses, total_net_profit, total_gross_profit,
                revenue_cagr, expense_inflation, profit_margin_target,
                completion_score, data_quality_score, projection_confidence_score,
                processing_duration, is_cached, cache_version,
                historical_revenue, historical_expenses, historical_net_profit, historical_gross_profit,
                historical_period, historical_revenue_growth, historical_expense_growth, historical_profit_margin
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (client_id, timeframe) 
            DO UPDATE SET
                business_name = EXCLUDED.business_name,
                projections_data = EXCLUDED.projections_data,
                methodology = EXCLUDED.methodology,
                executive_summary = EXCLUDED.executive_summary,
                total_revenue = EXCLUDED.total_revenue,
                total_expenses = EXCLUDED.total_expenses,
                total_net_profit = EXCLUDED.total_net_profit,
                total_gross_profit = EXCLUDED.total_gross_profit,
                revenue_cagr = EXCLUDED.revenue_cagr,
                expense_inflation = EXCLUDED.expense_inflation,
                profit_margin_target = EXCLUDED.profit_margin_target,
                completion_score = EXCLUDED.completion_score,
                data_quality_score = EXCLUDED.data_quality_score,
                projection_confidence_score = EXCLUDED.projection_confidence_score,
                processing_duration = EXCLUDED.processing_duration,
                historical_revenue = EXCLUDED.historical_revenue,
                historical_expenses = EXCLUDED.historical_expenses,
                historical_net_profit = EXCLUDED.historical_net_profit,
                historical_gross_profit = EXCLUDED.historical_gross_profit,
                historical_period = EXCLUDED.historical_period,
                historical_revenue_growth = EXCLUDED.historical_revenue_growth,
                historical_expense_growth = EXCLUDED.historical_expense_growth,
                historical_profit_margin = EXCLUDED.historical_profit_margin,
                updated_at = NOW(),
                cache_version = financial_projections.cache_version + 1
            RETURNING id
            """
            
            # Extract historical baseline data
            historical_baseline = projection_data.get('historical_baseline', {})
            
            cur.execute(sql, (
                client_id, timeframe, projection_data.get('business_name'),
                json.dumps(projections), json.dumps(methodology), projection_data.get('executive_summary'),
                projection_data.get('total_revenue', 0), projection_data.get('total_expenses', 0), 
                projection_data.get('total_net_profit', 0), projection_data.get('total_gross_profit', 0),
                methodology.get('growth_rate_assumptions', {}).get('revenue_cagr', 0),
                methodology.get('growth_rate_assumptions', {}).get('expense_inflation', 0),
                methodology.get('growth_rate_assumptions', {}).get('profit_margin_target', 0),
                projection_data.get('completion_score', {}).get('score', 0),
                projection_data.get('data_quality_score', {}).get('score', 0),
                projection_data.get('projection_confidence_score', {}).get('score', 0),
                processing_duration, True, 1,
                # Historical baseline values
                historical_baseline.get('current_annual_revenue', 0),
                historical_baseline.get('current_annual_expenses', 0),
                historical_baseline.get('current_annual_net_profit', 0),
                historical_baseline.get('current_annual_gross_profit', 0),
                historical_baseline.get('baseline_period', ''),
                historical_baseline.get('revenue_growth_rate', 0) / 100.0,  # Convert from percentage
                historical_baseline.get('expense_inflation_rate', 0) / 100.0,  # Convert from percentage
                historical_baseline.get('profit_margin_trend', 0) / 100.0  # Convert from percentage
            ))
            
            projection_id = cur.fetchone()[0]
            conn.commit()
            
            logger.info(f"✅ Projection saved to cache with ID {projection_id}")
            
    except Exception as e:
        logger.error(f"🚨 Error saving projection to cache: {str(e)}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

def clear_client_cache(client_id: str, timeframe: str = None):
    """Clear cache for a specific client and optionally specific timeframe"""
    conn = None
    try:
        logger.info(f"🗑️ Clearing cache for client {client_id}" + (f", timeframe {timeframe}" if timeframe else ""))
        conn = get_projections_db_connection()
        
        with conn.cursor() as cur:
            if timeframe:
                sql = "DELETE FROM financial_projections WHERE client_id = %s AND timeframe = %s"
                cur.execute(sql, (client_id, timeframe))
            else:
                sql = "DELETE FROM financial_projections WHERE client_id = %s"
                cur.execute(sql, (client_id,))
            
            deleted_count = cur.rowcount
            conn.commit()
            
            logger.info(f"✅ Cleared {deleted_count} cached projections")
            return deleted_count
            
    except Exception as e:
        logger.error(f"🚨 Error clearing cache: {str(e)}")
        return 0
    finally:
        if conn:
            conn.close()

# ═══════════════════════════════════════════════════════════════════════════════
# PARALLEL PROJECTION GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

TIMEFRAME_CONFIGS = {
    "1 Year": {"years": 1, "api_key_index": 0},
    "3 Years": {"years": 3, "api_key_index": 1},
    "5 Years": {"years": 5, "api_key_index": 2},
    "10 Years": {"years": 10, "api_key_index": 3},
    "15 Years": {"years": 15, "api_key_index": 4}  # End Game
}

def try_vertex_ai_projection_request(
    combined_financial_content: str,
    config: dict,
    client_id: str
) -> Optional[Dict]:
    """
    Try making request using Vertex AI (PRIMARY METHOD for Financial Projections).
    Returns response if successful, None if fails (will fallback to API keys).
    """
    if not vertex_ai_client:
        logger.info("Vertex AI client not available, skipping to API keys fallback")
        return None

    try:
        logger.info("🚀 Trying Vertex AI (Primary Method for Financial Projections)")

        # Convert config to Vertex AI format
        vertex_config = {
            "temperature": config.get("temperature", 0.0),
            "max_output_tokens": config.get("max_output_tokens", 8192),
            "top_p": config.get("top_p", 1.0),
            "top_k": config.get("top_k", 1),
        }

        # If we have a thinking budget, add it
        if hasattr(config, "thinking_config"):
            vertex_config["thinking_config"] = config.thinking_config

        # Call Vertex AI using GenAI SDK with gemini-2.5-pro
        response = vertex_ai_client.models.generate_content(
            model="gemini-2.5-pro",
            contents=combined_financial_content,
            config=vertex_config
        )

        if response and response.text:
            logger.info(f"✅ Vertex AI request successful for client {client_id}")

            # Track token usage if available
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                token_usage = {
                    "input_tokens": getattr(usage, 'prompt_token_count', 0) or getattr(usage, 'input_token_count', 0),
                    "output_tokens": getattr(usage, 'candidates_token_count', 0) or getattr(usage, 'output_token_count', 0),
                    "thinking_tokens": getattr(usage, 'thoughts_token_count', 0) or 0,
                    "total_tokens": getattr(usage, 'total_token_count', 0),
                    "method": "vertex_ai"
                }
                logger.info(f"🧮 Vertex AI Token Usage - Input: {token_usage['input_tokens']:,} | Output: {token_usage['output_tokens']:,} | Total: {token_usage['total_tokens']:,}")
                save_to_local_storage(token_usage, 'api_responses', 'vertex_ai_token_usage', client_id)

            return response
        else:
            logger.warning(f"⚠️ Vertex AI returned empty response for client {client_id}")
            return None

    except Exception as e:
        logger.warning(f"❌ Vertex AI request failed: {str(e)} - Falling back to API keys")
        vertex_error = {
            "error": str(e),
            "client_id": client_id,
            "method": "vertex_ai",
            "timestamp": datetime.now().isoformat()
        }
        save_to_local_storage(vertex_error, 'errors', 'vertex_ai_error', client_id)
        return None

def generate_single_timeframe_projection(client_id: str, timeframe: str, financial_content: str, api_key_index: int, base_prompt: str = None):
    """Generate projection for a single timeframe using specific API key"""
    try:
        # Add staggered delay to avoid simultaneous API calls
        delay = api_key_index * 0.5  # 0.5s delay between each request
        logger.info(f"⏱️ Waiting {delay}s before starting {timeframe} projection (API key {api_key_index})")
        time.sleep(delay)
        
        logger.info(f"🚀 Starting {timeframe} projection for client {client_id} using API key {api_key_index}")
        
        # Create separate Gemini client for this timeframe with deterministic API key
        from google import genai as genai_module
        deterministic_key = get_deterministic_api_key(client_id)
        timeframe_client = genai_module.Client(api_key=deterministic_key)
        
        # Create timeframe-specific prompt
        years = TIMEFRAME_CONFIGS[timeframe]["years"]
        base_year, target_year = get_dynamic_year_range(years)
        
        prompt = f"""
        REALISTIC FINANCIAL PROJECTION ANALYSIS - {timeframe.upper()} TIMEFRAME

        You are a senior financial analyst with expertise in business cycles and realistic long-term projections.
        Create comprehensive {timeframe} financial projections that account for market dynamics, competitive pressure, and economic cycles.

        TIMEFRAME SPECIFICATION:
        - Projection Period: {years} year(s)
        - Analysis Type: {timeframe} Strategic Planning with Business Reality Constraints
        - Base Year: {base_year}
        - Target Year: {target_year}
        - Projection Start: January {base_year}

        BUSINESS REALITY FOR {timeframe.upper()} PROJECTIONS:

        {("SHORT-TERM REALISM (1-3 Years):" if years <= 3 else
          "MEDIUM-TERM REALISM (3-7 Years):" if years <= 7 else
          "LONG-TERM REALISM (7+ Years):")}

        {"- Use historical growth rates with market trend adjustments" if years <= 3 else
         "- Apply 20-35% growth rate reduction from historical averages" if years <= 7 else
         "- Cap growth at 5-15% annually due to market maturity"}

        {"- Maintain current expense ratios with gradual improvements" if years <= 3 else
         "- Expense ratios stabilize at 15-20% (operational reality)" if years <= 7 else
         "- Expense ratios remain 15-25% (cannot go below due to scaling complexity)"}

        {"- Account for market conditions and competition" if years <= 3 else
         "- Include 1 challenging year (economic slowdown)" if years <= 7 else
         "- Include 2-3 challenging years (economic cycles every 7-10 years)"}

        {"- Maintain current profit margin trends" if years <= 3 else
         "- Profit margins approach industry norms (8-18% for service businesses)" if years <= 7 else
         "- Profit margins stabilize within industry benchmarks (8-18%)"}

        CRITICAL CONSTRAINTS FOR {timeframe.upper()}:
        - NEVER project expense ratios below 12% for service businesses
        - Growth rates must decelerate over time (market saturation effect)
        - Include realistic economic cycle impacts
        - Maintain industry-appropriate profit margins (8-18% for BPO/staffing)

        FINANCIAL DATA:
        {financial_content}

        PROJECTION REQUIREMENTS FOR {timeframe.upper()}:

        For {timeframe} projections, provide:
        
        1. BUSINESS OVERVIEW:
           - Business name and industry
           - Current market position
           - Key value propositions
        
        2. GROWTH STRATEGY FOR {timeframe}:
           - Revenue growth assumptions (CAGR %)
           - Market expansion opportunities
           - Operational scaling plans
           - Investment requirements
        
        3. DETAILED FINANCIAL PROJECTIONS:
        
        {"Monthly breakdown for 12 months" if years == 1 else 
         "Quarterly breakdown" if years <= 5 else 
         "Annual breakdown"}
        
        For each period, calculate:
        - Revenue (with growth trajectory)
        - Cost of Goods Sold (COGS)
        - Gross Profit
        - Operating Expenses
        - EBITDA
        - Net Profit
        
        4. KEY ASSUMPTIONS:
        - Revenue CAGR: [specify %]
        - Expense inflation: [specify %]
        - Profit margin targets
        - Market growth factors
        
        5. RISK ASSESSMENT:
        - Key business risks for {timeframe}
        - Mitigation strategies
        - Scenario planning (best/worst case)
        
        RESPONSE FORMAT:
        Provide a comprehensive JSON response with the structure matching the EnhancedProjectionSchema.
        
        Focus on realistic, data-driven projections that account for {timeframe} business cycles and market dynamics.
        """
        
        # Generate content using timeframe-specific client with SAME config as single generation
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EnhancedProjectionSchema,
            temperature=0.0,  # ← DETERMINISTIC: Same input = Same output
            top_p=1.0,        # ← Use all possible tokens
            top_k=1,          # ← Always pick most likely token
            system_instruction=prompt,
        )

        response = timeframe_client.models.generate_content(
            model="gemini-2.5-pro",
            contents="Generate comprehensive financial projections based on the provided data and analysis requirements.",
            config=config
        )
        
        if not response or not response.text:
            raise Exception(f"No response from Gemini for {timeframe}")
        
        # Parse the JSON response with error handling
        try:
            projection_data = json.loads(response.text)
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing error for {timeframe}: {str(e)}")
            logger.error(f"Raw response text (first 500 chars): {response.text[:500]}")
            
            # Try to clean the JSON by removing common issues
            cleaned_text = response.text
            
            # Remove any trailing commas before closing brackets/braces
            import re
            cleaned_text = re.sub(r',(\s*[}\]])', r'\1', cleaned_text)
            
            # Try parsing the cleaned text
            try:
                projection_data = json.loads(cleaned_text)
                logger.info(f"✅ JSON parsing succeeded after cleaning for {timeframe}")
            except json.JSONDecodeError as e2:
                logger.error(f"❌ JSON parsing failed even after cleaning: {str(e2)}")
                raise Exception(f"Failed to parse JSON response for {timeframe}: {str(e)}")
        
        logger.info(f"✅ {timeframe} projection completed for client {client_id}")
        
        return {
            "timeframe": timeframe,
            "data": projection_data,
            "success": True,
            "api_key_index": api_key_index,
            "processing_time": time.time()
        }
        
    except Exception as e:
        logger.error(f"❌ Error generating {timeframe} projection: {str(e)}")
        return {
            "timeframe": timeframe,
            "error": str(e),
            "success": False,
            "api_key_index": api_key_index
        }

async def generate_all_timeframes_parallel(client_id: str, financial_content: str, base_prompt: str):
    """Generate projections for all timeframes in parallel with rate limiting"""
    start_time = time.time()
    logger.info(f"🚀 Starting PARALLEL projection generation for client {client_id}")
    
    # Create executor with limited concurrency to avoid rate limits
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all timeframe tasks
        future_to_timeframe = {
            executor.submit(
                generate_single_timeframe_projection,
                client_id,
                timeframe,
                financial_content,
                config["api_key_index"],
                base_prompt
            ): timeframe
            for timeframe, config in TIMEFRAME_CONFIGS.items()
        }
        
        results = {}
        
        # Collect results as they complete
        for future in asyncio.as_completed([asyncio.wrap_future(f) for f in future_to_timeframe.keys()]):
            try:
                result = await future
                timeframe = result["timeframe"]
                results[timeframe] = result
                
                if result["success"]:
                    logger.info(f"✅ {timeframe} projection completed successfully")
                else:
                    logger.error(f"❌ {timeframe} projection failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"❌ Error in parallel processing: {str(e)}")
    
    total_time = time.time() - start_time
    successful_projections = sum(1 for r in results.values() if r["success"])
    
    logger.info(f"🎉 Parallel projection generation completed in {total_time:.2f}s")
    logger.info(f"✅ Successful: {successful_projections}/{len(TIMEFRAME_CONFIGS)} projections")
    
    return results

async def save_all_projections_to_cache(client_id: str, projection_results: dict, total_processing_time: float):
    """Save all successful projections to cache"""
    saved_count = 0
    
    for timeframe, result in projection_results.items():
        if result["success"]:
            try:
                save_projection_to_cache(
                    client_id, 
                    result["data"], 
                    total_processing_time,
                    timeframe
                )
                saved_count += 1
                logger.info(f"💾 Cached {timeframe} projection for client {client_id}")
            except Exception as e:
                logger.error(f"❌ Failed to cache {timeframe}: {str(e)}")
    
    logger.info(f"💾 Successfully cached {saved_count} projections for client {client_id}")
    return saved_count

@app.post("/clear-cache")
async def clear_cache_endpoint(client_id: str = Query(..., description="Client ID to clear cache for")):
    """Clear all cached projections for a client"""
    try:
        cleared_count = clear_client_cache(client_id)
        return {"status": "success", "cleared_count": cleared_count, "message": f"Cleared {cleared_count} cached projections for client {client_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@app.post("/predict-all-timeframes")
async def predict_all_timeframes_parallel(client_id: str = Query(..., description="Client ID for comprehensive analysis")):
    """
    Generate projections for ALL timeframes in parallel using multiple API keys
    """
    start_time = time.time()
    logger.info(f"🚀 PARALLEL ALL-TIMEFRAMES REQUEST: Client ID: {client_id}")
    
    try:
        # Check if we have cached data for all timeframes
        cached_timeframes = {}
        missing_timeframes = []
        
        for timeframe in TIMEFRAME_CONFIGS.keys():
            cached = check_cached_projection(client_id, timeframe)
            if cached:
                cached_timeframes[timeframe] = cached
                logger.info(f"✅ Cache HIT for {timeframe}")
            else:
                missing_timeframes.append(timeframe)
                logger.info(f"❌ Cache MISS for {timeframe}")
        
        if not missing_timeframes:
            # All timeframes are cached, return combined result
            logger.info(f"🚀 ALL timeframes cached for client {client_id}")
            
            combined_result = {
                "status": "success",
                "client_id": client_id,
                "from_cache": True,
                "cached_timeframes": list(cached_timeframes.keys()),
                "projections": {}
            }
            
            for timeframe, cached_data in cached_timeframes.items():
                # Handle projections_data (could be str or dict)
                projections_data = cached_data.get('projections_data', '{}')
                if isinstance(projections_data, str):
                    projections_data = json.loads(projections_data)
                elif projections_data is None:
                    projections_data = {}
                
                # Handle methodology (could be str or dict)
                methodology_data = cached_data.get('methodology', '{}')
                if isinstance(methodology_data, str):
                    methodology_data = json.loads(methodology_data)
                elif methodology_data is None:
                    methodology_data = {}
                
                combined_result["projections"][timeframe] = {
                    "business_name": cached_data.get('business_name'),
                    "projections_data": projections_data,
                    "methodology": methodology_data,
                    "cached_at": cached_data.get('created_at').isoformat() if cached_data.get('created_at') else None
                }
            
            return combined_result
        
        # Get financial data
        logger.info(f"📊 Fetching financial data for parallel processing")
        financial_content = get_financial_data_from_database(client_id)
        
        if not financial_content:
            raise HTTPException(status_code=404, detail=f"No financial data found for client {client_id}")
        
        # Define the same advanced prompt as used in single generation
        base_prompt ="""
ROLE
You are a senior financial analyst specializing in data-driven projections. You analyze historical patterns and project forward based SOLELY on what the actual data shows, accounting for natural business evolution patterns observed in the data itself.

CRITICAL REQUIREMENT: You MUST extract and calculate ALL assumptions from the provided financial documents. DO NOT use external benchmarks or preset values. Everything must be derived from this client's actual historical performance.

DATA-DRIVEN ADVANCED ANALYSIS METHODOLOGY

1. TIME SERIES DECOMPOSITION:
   From the provided data, perform STL decomposition to extract:
   - TREND: Calculate the underlying growth/decline trajectory (no linear assumptions)
   - SEASONALITY: Identify actual monthly/quarterly patterns in the data
   - RESIDUAL: Measure random variations and volatility patterns
   - Use these components separately for different projection horizons

2. COMPONENT COST STRUCTURE ANALYSIS:
   From historical expense data, separate and calculate:
   - FIXED COSTS: Expenses that remained constant regardless of revenue level
   - VARIABLE COSTS: Expenses that correlate with revenue changes (calculate correlation coefficient)
   - STEP COSTS: Expenses that jumped at specific revenue/time thresholds in the data
   - Total Expense Model = Fixed + (Variable_Rate × Revenue) + Step_Function_Costs

3. STRUCTURAL BREAK DETECTION:
   Analyze the data for business model changes:
   - Identify when cost structures fundamentally shifted
   - Detect revenue growth regime changes
   - Use only post-break data for projections if breaks are found
   - Document any operational changes that caused breaks

4. AUTOCORRELATION & LAG ANALYSIS:
   From monthly data, calculate:
   - How strongly does previous month's performance predict current month?
   - What is the optimal lag period for revenue/expense predictions?
   - Build autoregressive components based on actual correlations found

5. ROLLING-ORIGIN BACKTESTING:
   Before making projections, validate methodology:
   - Test forecast accuracy using expanding windows on historical data
   - Calculate prediction errors (MAPE, RMSE) for your specific business
   - Select forecasting method with lowest error on YOUR data

6. VOLATILITY & VARIANCE ANALYSIS:
   From the actual monthly data, calculate:
   - Month-to-month coefficient of variation for each metric
   - Seasonal volatility patterns (which months are most/least volatile?)
   - Revenue-expense correlation strength and lag effects
   - Apply SAME volatility patterns to future projections

7. CAPACITY CONSTRAINT IDENTIFICATION:
   From historical scaling patterns, identify:
   - Revenue per employee trends from your actual hiring data
   - When did quality/efficiency decline due to rapid growth?
   - What revenue levels required operational changes/new hires?
   - Use YOUR scaling constraints, not external assumptions

8. OUTLIER & REGIME ANALYSIS:
   Statistical analysis of your data:
   - Identify and classify outliers (temporary vs structural changes)
   - Detect regime shifts in your business model
   - Separate one-time events from recurring patterns
   - Project only stable, recurring patterns forward

9. DRIVER CORRELATION ANALYSIS:
   From your operational metrics (if available):
   - Which factors actually drive revenue changes in YOUR business?
   - Calculate correlation coefficients between operational and financial metrics
   - Build driver-based models using YOUR correlation strengths
   - Include only statistically significant relationships (p < 0.05)

10. CASH FLOW TIMING ANALYSIS:
    From your payment/billing data:
    - Calculate working capital patterns from your data
    - Identify payment timing cycles specific to your business
    - Model cash flow timing based on YOUR collection patterns
    - Account for seasonal working capital needs from your data

MATHEMATICAL VALIDATION REQUIREMENTS:

Before making any projections, the AI must:

1. STATISTICAL SIGNIFICANCE TESTING:
   - All correlations must be statistically significant (p < 0.05)
   - Report R-squared values for all relationships used
   - Use only relationships with sufficient historical data points (n ≥ 24 months minimum)

2. MODEL ACCURACY VALIDATION:
   - Perform out-of-sample testing on last 6-12 months of data
   - Calculate and report MAPE (Mean Absolute Percentage Error) for all components
   - Document which forecasting method performs best on YOUR data specifically

3. RESIDUAL ANALYSIS:
   - Check for autocorrelation in residuals (no patterns left unexplained)
   - Validate homoscedasticity (consistent variance over time)
   - Identify and document any remaining unexplained patterns

4. SENSITIVITY ANALYSIS:
   - Test how projections change with ±10% changes in key parameters
   - Identify which variables have the most impact on outcomes
   - Base sensitivity on actual historical variations in YOUR data

5. RECOVERY/TURNAROUND ANALYSIS:
   From historical data, analyze realistic recovery patterns:
   - If company is currently unprofitable, look for signs of improvement trajectory
   - Calculate the historical time it took to recover from previous losses (if any)
   - Identify what operational changes led to past recoveries
   - Determine realistic timeline for return to profitability based on historical patterns
   - CRITICAL: No healthy business stays unprofitable for 10+ years - model realistic recovery

6. CYCLICAL PATTERN DETECTION:
   From historical data, identify:
   - Any recurring down periods or challenging years
   - Recovery patterns following difficult periods
   - Seasonal variations and their magnitude
   - Economic sensitivity indicators in the historical performance

HISTORICAL BASELINE CALCULATION (CRITICAL - USE MOST RECENT DATA)
Calculate from the MOST RECENT complete 12-month period:
- current_annual_revenue: Most recent 12 months revenue
- current_annual_expenses: Most recent 12 months operating expenses
- current_annual_net_profit: Most recent 12 months net profit
- current_annual_gross_profit: Most recent 12 months gross profit
- baseline_period: Exact period this represents
- revenue_growth_rate: Calculate from historical trend analysis
- expense_inflation_rate: Calculate from historical expense scaling patterns
- profit_margin_trend: Calculate from historical margin evolution

Apply these baseline values consistently across all timeframes, then apply realistic scaling constraints.

DATA-DRIVEN PROJECTION METHODOLOGY

Based EXCLUSIVELY on patterns found in the historical data:

1. GROWTH RATE PROJECTION:
   - Calculate the mathematical trend in growth rate evolution from the data
   - If data shows growth deceleration over time, continue that deceleration pattern
   - If data shows consistent growth rates, project those consistently
   - If data shows cyclical patterns, incorporate those cycles
   - Use the ACTUAL mathematical trend, not assumed deceleration

2. EXPENSE SCALING PROJECTION:
   - Analyze how expense ratios have evolved as the business grew
   - Project expense ratios based on the historical scaling pattern observed
   - If ratios improved with scale, continue that improvement at the observed rate
   - If ratios remained stable, maintain stability
   - If ratios worsened, project continued challenge
   - Use the ACTUAL scaling pattern from the data

3. PROFITABILITY EVOLUTION:
   - Calculate profit margin trends from the historical data
   - Project margins based on the mathematical trend observed
   - Account for the relationship between scale and profitability shown in the data
   - Use the ACTUAL profitability evolution pattern

4. LOSS-TO-RECOVERY MODELING (CRITICAL FOR UNPROFITABLE COMPANIES):
   - If currently unprofitable, analyze historical recovery patterns from the data
   - Look for signs of operational improvement: revenue growth, expense control, efficiency gains
   - Model realistic recovery timeline based on observed business trajectory improvements
   - NEVER project continuous losses beyond 2-3 years unless data shows terminal decline
   - For 5+ year projections, model gradual return to profitability unless data contradicts this
   - Use actual recovery rates from historical turnarounds if they exist in the data

5. CYCLICAL PATTERN APPLICATION:
   - If historical data shows periodic downturns, incorporate similar patterns
   - Use the ACTUAL magnitude and frequency of variations observed
   - If no cyclical patterns exist in the data, don't assume they will occur
   - Project the natural business rhythm shown in the historical performance

6. VOLATILITY MODELING:
   - Calculate actual month-to-month and year-to-year variations
   - Apply the SAME level of volatility observed historically
   - Maintain mathematical consistency with observed patterns

VALIDATION CHECK: Before responding, verify:
1. Are all assumptions calculated from the actual historical data?
2. Do growth rates follow the mathematical trend observed in the data?
3. Do expense ratios follow the scaling pattern shown historically?
4. Are profit margins evolving according to the data-driven trend?
5. Do projections reflect the actual volatility observed in the data?
6. Do the profit figures make mathematical sense (Revenue - Expenses = Net Profit)?
7. LOSS SCENARIOS: If currently unprofitable, have you modeled realistic recovery (2-3 years max for continuous losses)?
8. BUSINESS REALITY: Does the long-term projection show a viable, sustainable business model?

IMPORTANT: These baseline values MUST be identical for all timeframes (1Y, 3Y, 5Y, 10Y, 15Y)
as they represent the current state before projections begin.

PROJECTION METHODOLOGY - PURELY DATA-DRIVEN:

For monthly projections:
- Use the calculated historical monthly patterns and variations
- Apply growth trends exactly as observed in the data evolution
- Include seasonal patterns at the same magnitude observed historically

For annual projections:
- Apply growth rate evolution as calculated from historical data analysis
- Use expense scaling patterns derived from actual historical scaling
- Include cyclical variations only if they exist in the historical data
- Maintain the same business rhythm observed in the historical performance

For the growth_rate_assumptions section:
- revenue_cagr: Calculate from actual historical trend analysis (not external assumptions)
- expense_inflation: Calculate from actual expense evolution patterns
- profit_margin_target: Calculate from actual margin evolution trend
- All values must be DERIVED from the historical data analysis

CONFIDENCE SCORING
Base your confidence on:
- Quality and completeness of historical data available
- Clarity and consistency of patterns observed in the data
- Mathematical reliability of calculated trends and relationships
- Strength of correlations found in the historical analysis

Your confidence score reflects confidence in DATA-DRIVEN PROJECTION METHODOLOGY.
State clearly: "This projection follows the mathematical patterns and business evolution observed in the historical data."

MATHEMATICAL CONSISTENCY FROM DATA
Ensure for every period:
- Net Profit = Gross Profit - Expenses (must balance exactly)
- Gross Profit = Revenue * historical_gross_margin_pattern
- Expenses = Revenue * observed_expense_ratio_evolution
- Growth rates follow the mathematical trend calculated from historical data

VALIDATION CHECKS
Before outputting:
- Verify all ratios follow patterns calculated from historical data
- Confirm growth rates match mathematical trends observed in the data
- Check that expense scaling follows the historical scaling pattern
- Ensure cyclical variations match those found in historical performance
- Verify all financial relationships maintain mathematical consistency

OUTPUT REQUIREMENTS
1. All growth rates must be calculated from historical trend analysis
2. All expense ratios must follow patterns observed in the data
3. Include variations only as observed in historical performance
4. All margin evolution must follow data-calculated trends
5. Document exactly how each pattern was derived from the data

DOCUMENTATION
In your response, clearly state:
- Exact historical patterns identified and how they were calculated
- Mathematical trends derived from the data and their formulas
- Specific data points used for trend calculations
- How historical volatility was measured and applied
- Why your projections accurately continue observed historical patterns

Generate projections that mathematically continue the specific patterns, trends, and relationships found in this company's historical data while maintaining perfect mathematical consistency.

CONFIDENCE SCORING VALIDATION:
- Base confidence purely on data quality and pattern clarity
- Higher confidence for clear, consistent historical patterns
- Lower confidence for volatile or inconsistent historical data
- Document data quality factors that influenced confidence score
- Explain mathematical rigor of trend analysis in confidence assessment
"""

        # Generate all missing projections in parallel
        logger.info(f"⚡ Generating {len(missing_timeframes)} timeframes in parallel")
        projection_results = await generate_all_timeframes_parallel(client_id, financial_content, base_prompt)
        
        # Save all successful projections to cache
        total_time = time.time() - start_time
        await save_all_projections_to_cache(client_id, projection_results, total_time)
        
        # Combine cached and newly generated data
        combined_result = {
            "status": "success",
            "client_id": client_id,
            "from_cache": False,
            "processing_time": total_time,
            "generated_timeframes": [t for t, r in projection_results.items() if r["success"]],
            "cached_timeframes": list(cached_timeframes.keys()),
            "projections": {}
        }
        
        # Add cached projections
        for timeframe, cached_data in cached_timeframes.items():
            # Handle projections_data (could be str or dict)
            projections_data = cached_data.get('projections_data', '{}')
            if isinstance(projections_data, str):
                projections_data = json.loads(projections_data)
            elif projections_data is None:
                projections_data = {}
            
            # Handle methodology (could be str or dict)
            methodology_data = cached_data.get('methodology', '{}')
            if isinstance(methodology_data, str):
                methodology_data = json.loads(methodology_data)
            elif methodology_data is None:
                methodology_data = {}
            
            combined_result["projections"][timeframe] = {
                "business_name": cached_data.get('business_name'),
                "projections_data": projections_data,
                "methodology": methodology_data,
                "cached_at": cached_data.get('created_at').isoformat() if cached_data.get('created_at') else None,
                "from_cache": True
            }
        
        # Add newly generated projections
        for timeframe, result in projection_results.items():
            if result["success"]:
                combined_result["projections"][timeframe] = {
                    **result["data"],
                    "generated_at": datetime.now().isoformat(),
                    "from_cache": False,
                    "api_key_index": result["api_key_index"]
                }
        
        logger.info(f"🎉 Parallel all-timeframes prediction completed for client {client_id}")
        
        return combined_result
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"💥 Parallel prediction error for client {client_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Parallel prediction failed: {str(e)}")

async def _predict_internal(client_id: str):
    """Internal prediction function that does the actual work"""
    start_time = datetime.now()
    logger.info(f"🎯🎯🎯 INTERNAL PREDICT FUNCTION: Client ID: {client_id} 🎯🎯🎯")
    
    # Store the complete request details
    request_details = {
        "client_id": client_id,
        "endpoint": "/predict",
        "request_timestamp": datetime.now().isoformat(),
        "request_type": "standard"
    }
    save_to_local_storage(request_details, 'requests', 'predict_request', client_id)
    
    try:
        # Check cache first for standard projections
        cached_projection = check_cached_projection(client_id, "1 Year")
        
        if cached_projection:
            logger.info(f"🚀 Returning CACHED projection for client {client_id}")
            
            # Convert cached data back to API response format
            projections_data = cached_projection.get('projections_data')
            if isinstance(projections_data, str):
                projections_data = json.loads(projections_data)
            elif projections_data is None:
                projections_data = {}
            
            methodology_data = cached_projection.get('methodology')
            if isinstance(methodology_data, str):
                methodology_data = json.loads(methodology_data)
            elif methodology_data is None:
                methodology_data = {}
            
            cached_response = {
                "status": "success",
                "business_name": cached_projection.get('business_name', ''),
                "executive_summary": cached_projection.get('executive_summary', ''),
                "projections_data": projections_data,
                "methodology": methodology_data,
                "completion_score": {"score": float(cached_projection.get('completion_score', 0))},
                "data_quality_score": {"score": float(cached_projection.get('data_quality_score', 0))},
                "projection_confidence_score": {"score": float(cached_projection.get('projection_confidence_score', 0))},
                "historical_baseline": {
                    "current_annual_revenue": float(cached_projection.get('historical_revenue', 0)),
                    "current_annual_expenses": float(cached_projection.get('historical_expenses', 0)),
                    "current_annual_net_profit": float(cached_projection.get('historical_net_profit', 0)),
                    "current_annual_gross_profit": float(cached_projection.get('historical_gross_profit', 0)),
                    "baseline_period": cached_projection.get('historical_period', 'Unknown'),
                    "revenue_growth_rate": float(cached_projection.get('historical_revenue_growth', 0) * 100),
                    "expense_inflation_rate": float(cached_projection.get('historical_expense_growth', 0) * 100),
                    "profit_margin_trend": float(cached_projection.get('historical_profit_margin', 0) * 100)
                },
                "from_cache": True,
                "cached_at": cached_projection.get('created_at').isoformat() if cached_projection.get('created_at') else None,
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
            
            # Add the full projection structure expected by frontend
            if 'projections_data' not in cached_response or not cached_response['projections_data']:
                # Create basic projection structure from cached totals
                cached_response["projections_data"] = {
                    "one_year_monthly": [
                        {"month": f"{get_projection_start_date().year}-{i:02d}", 
                         "revenue": float(cached_projection.get('total_revenue', 0)) / 12, 
                         "expenses": float(cached_projection.get('total_expenses', 0)) / 12,
                         "net_profit": float(cached_projection.get('total_net_profit', 0)) / 12,
                         "gross_profit": float(cached_projection.get('total_gross_profit', 0)) / 12}
                        for i in range(1, 13)
                    ]
                }
            
            return cached_response
        
        # If not cached, generate new projection
        # Step 1: Fetch financial data from database
        logger.info(f"📊 CACHE BYPASSED - Generating FRESH projection for client_id: {client_id}")
        logger.info(f"📊 Fetching financial data from database for client_id: {client_id}")
        fetch_start_time = datetime.now()
        
        combined_financial_content = get_financial_data_from_database(client_id)
        logger.info(f"📊 Financial data fetch result: {len(combined_financial_content) if combined_financial_content else 0} characters")
        
        fetch_end_time = datetime.now()
        fetch_duration = (fetch_end_time - fetch_start_time).total_seconds()
        
        logger.info(f"✓ Financial data retrieved successfully. Content: {len(combined_financial_content):,} chars, Duration: {fetch_duration:.2f}s")
        
        # Step 2: Prepare enhanced prompt - purely data-driven approach
        base_prompt ="""
ROLE
You are a senior financial analyst specializing in data-driven projections. You analyze historical patterns and project forward based SOLELY on what the actual data shows, accounting for natural business evolution patterns observed in the data itself.

CRITICAL REQUIREMENT: You MUST extract and calculate ALL assumptions from the provided financial documents. DO NOT use external benchmarks or preset values. Everything must be derived from this client's actual historical performance.

DATA-DRIVEN ADVANCED ANALYSIS METHODOLOGY

1. TIME SERIES DECOMPOSITION:
   From the provided data, perform STL decomposition to extract:
   - TREND: Calculate the underlying growth/decline trajectory (no linear assumptions)
   - SEASONALITY: Identify actual monthly/quarterly patterns in the data
   - RESIDUAL: Measure random variations and volatility patterns
   - Use these components separately for different projection horizons

2. COMPONENT COST STRUCTURE ANALYSIS:
   From historical expense data, separate and calculate:
   - FIXED COSTS: Expenses that remained constant regardless of revenue level
   - VARIABLE COSTS: Expenses that correlate with revenue changes (calculate correlation coefficient)
   - STEP COSTS: Expenses that jumped at specific revenue/time thresholds in the data
   - Total Expense Model = Fixed + (Variable_Rate × Revenue) + Step_Function_Costs

3. STRUCTURAL BREAK DETECTION:
   Analyze the data for business model changes:
   - Identify when cost structures fundamentally shifted
   - Detect revenue growth regime changes
   - Use only post-break data for projections if breaks are found
   - Document any operational changes that caused breaks

4. AUTOCORRELATION & LAG ANALYSIS:
   From monthly data, calculate:
   - How strongly does previous month's performance predict current month?
   - What is the optimal lag period for revenue/expense predictions?
   - Build autoregressive components based on actual correlations found

5. ROLLING-ORIGIN BACKTESTING:
   Before making projections, validate methodology:
   - Test forecast accuracy using expanding windows on historical data
   - Calculate prediction errors (MAPE, RMSE) for your specific business
   - Select forecasting method with lowest error on YOUR data

6. VOLATILITY & VARIANCE ANALYSIS:
   From the actual monthly data, calculate:
   - Month-to-month coefficient of variation for each metric
   - Seasonal volatility patterns (which months are most/least volatile?)
   - Revenue-expense correlation strength and lag effects
   - Apply SAME volatility patterns to future projections

7. CAPACITY CONSTRAINT IDENTIFICATION:
   From historical scaling patterns, identify:
   - Revenue per employee trends from your actual hiring data
   - When did quality/efficiency decline due to rapid growth?
   - What revenue levels required operational changes/new hires?
   - Use YOUR scaling constraints, not external assumptions

8. OUTLIER & REGIME ANALYSIS:
   Statistical analysis of your data:
   - Identify and classify outliers (temporary vs structural changes)
   - Detect regime shifts in your business model
   - Separate one-time events from recurring patterns
   - Project only stable, recurring patterns forward

9. DRIVER CORRELATION ANALYSIS:
   From your operational metrics (if available):
   - Which factors actually drive revenue changes in YOUR business?
   - Calculate correlation coefficients between operational and financial metrics
   - Build driver-based models using YOUR correlation strengths
   - Include only statistically significant relationships (p < 0.05)

10. CASH FLOW TIMING ANALYSIS:
    From your payment/billing data:
    - Calculate working capital patterns from your data
    - Identify payment timing cycles specific to your business
    - Model cash flow timing based on YOUR collection patterns
    - Account for seasonal working capital needs from your data

MATHEMATICAL VALIDATION REQUIREMENTS:

Before making any projections, the AI must:

1. STATISTICAL SIGNIFICANCE TESTING:
   - All correlations must be statistically significant (p < 0.05)
   - Report R-squared values for all relationships used
   - Use only relationships with sufficient historical data points (n ≥ 24 months minimum)

2. MODEL ACCURACY VALIDATION:
   - Perform out-of-sample testing on last 6-12 months of data
   - Calculate and report MAPE (Mean Absolute Percentage Error) for all components
   - Document which forecasting method performs best on YOUR data specifically

3. RESIDUAL ANALYSIS:
   - Check for autocorrelation in residuals (no patterns left unexplained)
   - Validate homoscedasticity (consistent variance over time)
   - Identify and document any remaining unexplained patterns

4. SENSITIVITY ANALYSIS:
   - Test how projections change with ±10% changes in key parameters
   - Identify which variables have the most impact on outcomes
   - Base sensitivity on actual historical variations in YOUR data

5. RECOVERY/TURNAROUND ANALYSIS:
   From historical data, analyze realistic recovery patterns:
   - If company is currently unprofitable, look for signs of improvement trajectory
   - Calculate the historical time it took to recover from previous losses (if any)
   - Identify what operational changes led to past recoveries
   - Determine realistic timeline for return to profitability based on historical patterns
   - CRITICAL: No healthy business stays unprofitable for 10+ years - model realistic recovery

6. CYCLICAL PATTERN DETECTION:
   From historical data, identify:
   - Any recurring down periods or challenging years
   - Recovery patterns following difficult periods
   - Seasonal variations and their magnitude
   - Economic sensitivity indicators in the historical performance

HISTORICAL BASELINE CALCULATION (CRITICAL - USE MOST RECENT DATA)
Calculate from the MOST RECENT complete 12-month period:
- current_annual_revenue: Most recent 12 months revenue
- current_annual_expenses: Most recent 12 months operating expenses
- current_annual_net_profit: Most recent 12 months net profit
- current_annual_gross_profit: Most recent 12 months gross profit
- baseline_period: Exact period this represents
- revenue_growth_rate: Calculate from historical trend analysis
- expense_inflation_rate: Calculate from historical expense scaling patterns
- profit_margin_trend: Calculate from historical margin evolution

DATA-DRIVEN PROJECTION METHODOLOGY

Based EXCLUSIVELY on patterns found in the historical data:

1. GROWTH RATE PROJECTION:
   - Calculate the mathematical trend in growth rate evolution from the data
   - If data shows growth deceleration over time, continue that deceleration pattern
   - If data shows consistent growth rates, project those consistently
   - If data shows cyclical patterns, incorporate those cycles
   - Use the ACTUAL mathematical trend, not assumed deceleration

2. EXPENSE SCALING PROJECTION:
   - Analyze how expense ratios have evolved as the business grew
   - Project expense ratios based on the historical scaling pattern observed
   - If ratios improved with scale, continue that improvement at the observed rate
   - If ratios remained stable, maintain stability
   - If ratios worsened, project continued challenge
   - Use the ACTUAL scaling pattern from the data

3. PROFITABILITY EVOLUTION:
   - Calculate profit margin trends from the historical data
   - Project margins based on the mathematical trend observed
   - Account for the relationship between scale and profitability shown in the data
   - Use the ACTUAL profitability evolution pattern

4. LOSS-TO-RECOVERY MODELING (CRITICAL FOR UNPROFITABLE COMPANIES):
   - If currently unprofitable, analyze historical recovery patterns from the data
   - Look for signs of operational improvement: revenue growth, expense control, efficiency gains
   - Model realistic recovery timeline based on observed business trajectory improvements
   - NEVER project continuous losses beyond 2-3 years unless data shows terminal decline
   - For 5+ year projections, model gradual return to profitability unless data contradicts this
   - Use actual recovery rates from historical turnarounds if they exist in the data

5. CYCLICAL PATTERN APPLICATION:
   - If historical data shows periodic downturns, incorporate similar patterns
   - Use the ACTUAL magnitude and frequency of variations observed
   - If no cyclical patterns exist in the data, don't assume they will occur
   - Project the natural business rhythm shown in the historical performance

6. VOLATILITY MODELING:
   - Calculate actual month-to-month and year-to-year variations
   - Apply the SAME level of volatility observed historically
   - Maintain mathematical consistency with observed patterns

VALIDATION CHECK: Before responding, verify:
1. Are all assumptions calculated from the actual historical data?
2. Do growth rates follow the mathematical trend observed in the data?
3. Do expense ratios follow the scaling pattern shown historically?
4. Are profit margins evolving according to the data-driven trend?
5. Do projections reflect the actual volatility observed in the data?
6. Do the profit figures make mathematical sense (Revenue - Expenses = Net Profit)?
7. LOSS SCENARIOS: If currently unprofitable, have you modeled realistic recovery (2-3 years max for continuous losses)?
8. BUSINESS REALITY: Does the long-term projection show a viable, sustainable business model?

IMPORTANT: These baseline values MUST be identical for all timeframes (1Y, 3Y, 5Y, 10Y, 15Y)
as they represent the current state before projections begin.

PROJECTION METHODOLOGY - PURELY DATA-DRIVEN:

For monthly projections:
- Use the calculated historical monthly patterns and variations
- Apply growth trends exactly as observed in the data evolution
- Include seasonal patterns at the same magnitude observed historically

For annual projections:
- Apply growth rate evolution as calculated from historical data analysis
- Use expense scaling patterns derived from actual historical scaling
- Include cyclical variations only if they exist in the historical data
- Maintain the same business rhythm observed in the historical performance

For the growth_rate_assumptions section:
- revenue_cagr: Calculate from actual historical trend analysis (not external assumptions)
- expense_inflation: Calculate from actual expense evolution patterns
- profit_margin_target: Calculate from actual margin evolution trend
- All values must be DERIVED from the historical data analysis

CONFIDENCE SCORING
Base your confidence on:
- Quality and completeness of historical data available
- Clarity and consistency of patterns observed in the data
- Mathematical reliability of calculated trends and relationships
- Strength of correlations found in the historical analysis

Your confidence score reflects confidence in DATA-DRIVEN PROJECTION METHODOLOGY.
State clearly: "This projection follows the mathematical patterns and business evolution observed in the historical data."

MATHEMATICAL CONSISTENCY FROM DATA
Ensure for every period:
- Net Profit = Gross Profit - Expenses (must balance exactly)
- Gross Profit = Revenue * historical_gross_margin_pattern
- Expenses = Revenue * observed_expense_ratio_evolution
- Growth rates follow the mathematical trend calculated from historical data

VALIDATION CHECKS
Before outputting:
- Verify all ratios follow patterns calculated from historical data
- Confirm growth rates match mathematical trends observed in the data
- Check that expense scaling follows the historical scaling pattern
- Ensure cyclical variations match those found in historical performance
- Verify all financial relationships maintain mathematical consistency

OUTPUT REQUIREMENTS
1. All growth rates must be calculated from historical trend analysis
2. All expense ratios must follow patterns observed in the data
3. Include variations only as observed in historical performance
4. All margin evolution must follow data-calculated trends
5. Document exactly how each pattern was derived from the data

DOCUMENTATION
In your response, clearly state:
- Exact historical patterns identified and how they were calculated
- Mathematical trends derived from the data and their formulas
- Specific data points used for trend calculations
- How historical volatility was measured and applied
- Why your projections accurately continue observed historical patterns

Generate projections that mathematically continue the specific patterns, trends, and relationships found in this company's historical data while maintaining perfect mathematical consistency.

CONFIDENCE SCORING VALIDATION:
- Base confidence purely on data quality and pattern clarity
- Higher confidence for clear, consistent historical patterns
- Lower confidence for volatile or inconsistent historical data
- Document data quality factors that influenced confidence score
- Explain mathematical rigor of trend analysis in confidence assessment
"""

        # Use base prompt for standard projections
        full_prompt = base_prompt
        
        # Store the complete prompt
        prompt_data = {
            "client_id": client_id,
            "full_prompt": full_prompt,
            "prompt_length": len(full_prompt),
            "prompt_creation_timestamp": datetime.now().isoformat()
        }
        save_to_local_storage(prompt_data, 'prompts', 'full_prompt', client_id)
        
        logger.info(f"📝 Prompt prepared. Length: {len(full_prompt):,} characters")

        # Step 3: Configure AI request
        logger.info("🤖 Preparing Google Generative AI API request...")
        
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EnhancedProjectionSchema,
            temperature=0.0,  # ← DETERMINISTIC: Same input = Same output
            top_p=1.0,        # ← Use all possible tokens
            top_k=1,          # ← Always pick most likely token
            system_instruction=full_prompt,
        )
        
        # Store API configuration
        api_config_data = {
            "client_id": client_id,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "thinking_budget": 32768,
            "response_mime_type": "application/json",
            "model": "gemini-2.5-pro",
            "config_timestamp": datetime.now().isoformat()
        }
        save_to_local_storage(api_config_data, 'api_responses', 'api_config', client_id)
        
        logger.debug("✓ GenerateContentConfig created successfully")

        # Step 4: Try Vertex AI first, then fallback to API keys
        response = None
        last_error = None

        # ===================================================================
        # STEP 4A: TRY VERTEX AI (PRIMARY METHOD)
        # ===================================================================
        vertex_response = try_vertex_ai_projection_request(
            combined_financial_content=combined_financial_content,
            config=config,
            client_id=client_id
        )

        if vertex_response:
            # Vertex AI succeeded
            response = vertex_response
            logger.info(f"🎉 [{client_id}] Projection completed via Vertex AI")
            api_attempts_log = {
                "client_id": client_id,
                "method": "vertex_ai",
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
            save_to_local_storage(api_attempts_log, 'api_responses', 'api_method', client_id)
        else:
            # ===================================================================
            # STEP 4B: FALLBACK TO API KEYS
            # ===================================================================
            logger.info(f"Falling back to API keys for client {client_id}")

            max_retries = 3
            retry_count = 0

            api_attempts_log = {
                "client_id": client_id,
                "method": "api_keys_fallback",
                "max_retries": max_retries,
                "attempts": [],
                "start_time": datetime.now().isoformat()
            }

            while retry_count < max_retries:
                attempt_start = datetime.now()
                try:
                    # Use deterministic API key for consistent results
                    deterministic_key = get_deterministic_api_key(client_id)
                    current_client = create_gemini_client(api_key=deterministic_key, client_id=client_id)
                    key_index = GEMINI_API_KEYS.index(deterministic_key)
                    logger.info(f"🔄 Attempt {retry_count + 1}/{max_retries} - Using API key fallback index: {key_index}")

                    attempt_data = {
                        "attempt_number": retry_count + 1,
                        "api_key_index": key_index,
                        "start_time": attempt_start.isoformat()
                    }

                    logger.info("📡 Making API call to Gemini 2.5 Pro via API key...")
                    response = current_client.models.generate_content(
                        model="gemini-2.5-pro",
                        contents=combined_financial_content,
                        config=config,
                    )

                    attempt_end = datetime.now()
                    attempt_duration = (attempt_end - attempt_start).total_seconds()

                    attempt_data["status"] = "success"
                    attempt_data["end_time"] = attempt_end.isoformat()
                    attempt_data["duration_seconds"] = attempt_duration
                    api_attempts_log["attempts"].append(attempt_data)

                    logger.info(f"✅ API call completed successfully in {attempt_duration:.2f}s")
                    break

                except Exception as e:
                    last_error = e
                    retry_count += 1
                    attempt_end = datetime.now()
                    attempt_duration = (attempt_end - attempt_start).total_seconds()

                    attempt_data["status"] = "failed"
                    attempt_data["error"] = str(e)
                    attempt_data["end_time"] = attempt_end.isoformat()
                    attempt_data["duration_seconds"] = attempt_duration
                    api_attempts_log["attempts"].append(attempt_data)

                    logger.warning(f"⚠️ API call attempt {retry_count} failed after {attempt_duration:.2f}s: {str(e)}")

                    if retry_count < max_retries:
                        logger.info(f"🔄 Retrying with different API key... ({retry_count}/{max_retries})")
                    else:
                        logger.error(f"❌ All {max_retries} API call attempts failed. Last error: {str(last_error)}")

            api_attempts_log["end_time"] = datetime.now().isoformat()
            api_attempts_log["final_status"] = "success" if response else "failed"
            save_to_local_storage(api_attempts_log, 'api_responses', 'api_attempts_log', client_id)
        
        if not response:
            logger.error("❌ No response received from AI service")
            final_error = {
                "client_id": client_id,
                "error_type": "no_response_from_ai",
                "attempts_made": max_retries,
                "last_error": str(last_error) if last_error else "Unknown error"
            }
            save_to_local_storage(final_error, 'errors', 'ai_service_failure', client_id)
            raise HTTPException(status_code=500, detail="Failed to get response from AI service")
        
        # Step 5: Process and log response
        logger.info("📋 Processing AI response...")
        
        # Store raw response immediately
        raw_response_data = {
            "client_id": client_id,
            "response_text": response.text if response.text else "No text response",
            "response_length": len(response.text) if response.text else 0,
            "has_parsed_response": hasattr(response, 'parsed') and response.parsed is not None,
            "response_timestamp": datetime.now().isoformat()
        }
        
        # Enhanced token usage logging
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            token_usage = {
                "input_tokens": getattr(usage, 'prompt_token_count', None) or getattr(usage, 'input_token_count', None) or 0,
                "output_tokens": getattr(usage, 'candidates_token_count', None) or getattr(usage, 'output_token_count', None) or 0,
                "thinking_tokens": getattr(usage, 'thoughts_token_count', None) or 0,
                "total_tokens": getattr(usage, 'total_token_count', None) or 0
            }
            raw_response_data["usage_metadata"] = token_usage
            
            logger.info(f"🧮 Token Usage - Input: {token_usage['input_tokens']:,} | Output: {token_usage['output_tokens']:,} | Thinking: {token_usage['thinking_tokens']:,} | Total: {token_usage['total_tokens']:,}")
            
            # Store detailed token usage
            save_to_local_storage(token_usage, 'api_responses', 'token_usage', client_id)
        else:
            logger.warning("⚠️ Usage metadata not available in response")
        
        save_to_local_storage(raw_response_data, 'api_responses', 'raw_response', client_id)
        
        if not response.text:
            logger.error("❌ Empty response received from AI service")
            empty_response_error = {
                "client_id": client_id,
                "error_type": "empty_response",
                "message": "AI service returned empty response"
            }
            save_to_local_storage(empty_response_error, 'errors', 'empty_ai_response', client_id)
            raise HTTPException(status_code=500, detail="Empty response from AI service")
        
        logger.info(f"📄 Response received. Length: {len(response.text):,} characters")

        # Step 6: Parse and validate response
        logger.info("🔍 Parsing and validating AI response...")
        
        try:
            validated_projection = response.parsed
            if not validated_projection:
                logger.debug("📝 response.parsed empty; falling back to manual JSON parse")
                parsed_response = json.loads(response.text)
                
                # Store parsed JSON for debugging
                parsed_json_data = {
                    "client_id": client_id,
                    "parsed_response": parsed_response,
                    "response_keys": list(parsed_response.keys()),
                    "parsing_method": "manual_json_parse"
                }
                save_to_local_storage(parsed_json_data, 'api_responses', 'parsed_json', client_id)
                
                # Check for None values
                none_values = {k: v for k, v in parsed_response.items() if v is None}
                if none_values:
                    logger.warning(f"⚠️ Found None values: {list(none_values.keys())}")
                    save_to_local_storage(none_values, 'api_responses', 'none_values_found', client_id)
                
                try:
                    validated_projection = EnhancedProjectionSchema(**parsed_response)
                except Exception as validation_error:
                    logger.error(f"❌ Schema validation error: {str(validation_error)}")
                    
                    validation_error_data = {
                        "client_id": client_id,
                        "error_type": "schema_validation_error",
                        "error_message": str(validation_error),
                        "parsed_response_preview": str(parsed_response)[:1000] + "..." if len(str(parsed_response)) > 1000 else str(parsed_response)
                    }
                    
                    if hasattr(validation_error, 'errors'):
                        validation_error_data["detailed_errors"] = validation_error.errors()
                    
                    save_to_local_storage(validation_error_data, 'errors', 'schema_validation_error', client_id)
                    raise HTTPException(status_code=500, detail=f"Schema validation failed: {str(validation_error)}")
            
            logger.info("✅ AI response successfully parsed and validated")
            
            # Step 6.5: Log and validate historical baseline for frontend integration
            if hasattr(validated_projection, 'historical_baseline'):
                baseline = validated_projection.historical_baseline
                baseline_log = {
                    "client_id": client_id,
                    "historical_baseline_validation": {
                        "current_annual_revenue": baseline.current_annual_revenue,
                        "current_annual_expenses": baseline.current_annual_expenses, 
                        "current_annual_net_profit": baseline.current_annual_net_profit,
                        "current_annual_gross_profit": baseline.current_annual_gross_profit,
                        "baseline_period": baseline.baseline_period,
                        "revenue_growth_rate": baseline.revenue_growth_rate,
                        "expense_inflation_rate": baseline.expense_inflation_rate,
                        "profit_margin_trend": baseline.profit_margin_trend
                    },
                    "validation_status": "SUCCESS",
                    "validation_timestamp": datetime.now().isoformat()
                }
                save_to_local_storage(baseline_log, 'validation', 'historical_baseline_validation', client_id)
                
                logger.info(f"📊 Historical Baseline Validated:")
                logger.info(f"   📈 Annual Revenue: ${baseline.current_annual_revenue:,.2f}")
                logger.info(f"   💰 Annual Expenses: ${baseline.current_annual_expenses:,.2f}")
                logger.info(f"   💵 Annual Net Profit: ${baseline.current_annual_net_profit:,.2f}")
                logger.info(f"   📊 Period: {baseline.baseline_period}")
                logger.info(f"   📈 Revenue Growth: {baseline.revenue_growth_rate:.1f}%")
            else:
                logger.warning("⚠️  Historical baseline missing from AI response!")
                missing_baseline_error = {
                    "error_type": "missing_historical_baseline",
                    "client_id": client_id,
                    "message": "AI response missing required historical_baseline field"
                }
                save_to_local_storage(missing_baseline_error, 'errors', 'missing_baseline', client_id)
            
            # Step 7: Store final validated results
            final_results = {
                "client_id": client_id,
                "business_name": validated_projection.business_name,
                "completion_score": validated_projection.completion_score.score if validated_projection.completion_score else None,
                "data_quality_score": validated_projection.data_quality_score.score if validated_projection.data_quality_score else None,
                "projection_confidence_score": validated_projection.projection_confidence_score.score if validated_projection.projection_confidence_score else None,
                "total_projections_generated": {
                    "monthly_1_year": len(validated_projection.projections_data.one_year_monthly),
                    "monthly_3_years": len(validated_projection.projections_data.three_years_monthly),
                    "quarterly_5_years": len(validated_projection.projections_data.five_years_quarterly),
                    "annual_10_years": len(validated_projection.projections_data.ten_years_annual),
                    "annual_15_years": len(validated_projection.projections_data.fifteen_years_annual)
                },
                "validation_timestamp": datetime.now().isoformat()
            }
            
            # Convert to dict for storage if possible
            try:
                if hasattr(validated_projection, 'model_dump'):
                    final_results["full_validated_projection"] = validated_projection.model_dump()
                elif hasattr(validated_projection, 'dict'):
                    final_results["full_validated_projection"] = validated_projection.dict()
                else:
                    final_results["full_validated_projection"] = str(validated_projection)
            except:
                final_results["full_validated_projection"] = "Could not serialize validated projection"
            
            save_to_local_storage(final_results, 'final_results', 'validated_projection', client_id)
            
            # Log key metrics
            logger.info(f"🏢 Business analyzed: {validated_projection.business_name or 'Unknown'}")
            
            if validated_projection.completion_score:
                logger.info(f"📊 Completion score: {validated_projection.completion_score.score:.2f}")
            if validated_projection.data_quality_score:
                logger.info(f"📊 Data quality score: {validated_projection.data_quality_score.score:.2f}")
            if validated_projection.projection_confidence_score:
                logger.info(f"📊 Projection confidence score: {validated_projection.projection_confidence_score.score:.2f}")
            
            # Log successful completion
            completion_summary = {
                "client_id": client_id,
                "operation": "predict",
                "status": "success",
                "processing_duration_seconds": (datetime.now() - fetch_start_time).total_seconds(),
                "data_points_generated": sum([
                    len(validated_projection.projections_data.one_year_monthly),
                    len(validated_projection.projections_data.three_years_monthly),
                    len(validated_projection.projections_data.five_years_quarterly),
                    len(validated_projection.projections_data.ten_years_annual),
                    len(validated_projection.projections_data.fifteen_years_annual)
                ]),
                "completion_timestamp": datetime.now().isoformat()
            }
            save_to_local_storage(completion_summary, 'system_logs', 'successful_prediction', client_id)
            
            # Save to cache for standard projections
            processing_duration = (datetime.now() - start_time).total_seconds()
            projection_dict = validated_projection.model_dump() if hasattr(validated_projection, 'model_dump') else validated_projection.dict()
            save_projection_to_cache(client_id, projection_dict, processing_duration, "1 Year")
            
            logger.info("🎉 Prediction completed successfully")
            return validated_projection
            
        except json.JSONDecodeError as json_error:
            logger.error(f"❌ JSON parsing failed: {str(json_error)}")
            json_error_data = {
                "client_id": client_id,
                "error_type": "json_decode_error",
                "error_message": str(json_error),
                "response_preview": response.text[:500] + "..." if len(response.text) > 500 else response.text
            }
            save_to_local_storage(json_error_data, 'errors', 'json_decode_error', client_id)
            raise HTTPException(status_code=500, detail=f"Invalid JSON response from AI service: {str(json_error)}")
        except HTTPException:
            raise
        except Exception as validation_error:
            logger.error(f"❌ Unexpected validation error: {str(validation_error)}")
            unexpected_error_data = {
                "client_id": client_id,
                "error_type": "unexpected_validation_error",
                "error_message": str(validation_error),
                "traceback": traceback.format_exc()
            }
            save_to_local_storage(unexpected_error_data, 'errors', 'unexpected_validation_error', client_id)
            raise HTTPException(status_code=500, detail=f"Response processing failed: {str(validation_error)}")
            
    except HTTPException as he:
        logger.error(f"❌ HTTP Exception: {he.detail}")
        http_exception_data = {
            "client_id": client_id,
            "error_type": "http_exception",
            "status_code": he.status_code,
            "detail": he.detail,
            "timestamp": datetime.now().isoformat()
        }
        save_to_local_storage(http_exception_data, 'errors', 'http_exception', client_id)
        raise
    except Exception as e:
        logger.critical(f"💥 Unhandled internal server error: {str(e)}", exc_info=True)
        critical_error_data = {
            "client_id": client_id,
            "error_type": "critical_internal_error",
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }
        save_to_local_storage(critical_error_data, 'errors', 'critical_internal_error', client_id)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/predict")
async def predict(
    client_id: str = Query(..., description="Client ID to fetch financial documents from database"),
    auth: Dict = Depends(verify_jwt_token)
):
    """
    Main prediction endpoint with queue system (JWT Protected)
    """
    # Permission check: verify client_id matches authenticated user's client_id
    if str(client_id) != str(auth.get("client_id")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only access your own financial projections"
        )

    logger.info(f"🎯 PREDICT REQUEST: Client ID: {client_id} (User: {auth['user_id']})")

    # Check if request should be queued
    is_queued, queue_info = await queue_projection_request(
        client_id,
        "standard_prediction",
        lambda: _predict_internal(client_id)
    )

    if is_queued:
        return queue_info

    try:
        # Execute prediction immediately
        result = await _predict_internal(client_id)

        # Process any queued requests from global queue
        await process_global_queue()

        return result

    except Exception as e:
        # Clean up processing state on error
        global currently_processing_client
        with queue_lock:
            currently_processing_client = None
            queue_stats["currently_processing"] = 0

        logger.error(f"❌ PREDICTION ERROR: Client {client_id}, Error: {str(e)}")
        raise

async def _force_regenerate_internal(client_id: str):
    """Internal force regeneration function that does the actual work"""
    start_time = datetime.now()
    logger.info(f"🔄 INTERNAL FORCE REGENERATION: Client ID: {client_id}")
    
    try:
        timeframe = "1 Year"
        
        # Clear existing cache
        cleared_count = clear_client_cache(client_id, timeframe)
        logger.info(f"🗑️ Cleared {cleared_count} cached projections")
        
        # Store the request details
        request_details = {
            "client_id": client_id,
            "endpoint": "/predict/force-regenerate",
            "request_timestamp": datetime.now().isoformat(),
            "request_type": "force_regeneration",
            "cleared_cache_count": cleared_count
        }
        save_to_local_storage(request_details, 'requests', 'force_regenerate_request', client_id)
        
        # Generate fresh projection by calling the regular predict function
        # but temporarily disable caching by setting a flag
        logger.info(f"🔄 Generating fresh projection for client {client_id}")
        
        # Get financial data from database
        combined_financial_content = get_financial_data_from_database(client_id)
        
        if not combined_financial_content:
            raise HTTPException(status_code=404, detail=f"No financial data found for client {client_id}")
        
        # Here we would call your existing prediction generation logic
        # For now, I'll redirect to the regular predict function
        # but we can expand this later with the full Gemini logic
        
        from fastapi import Request
        import urllib.parse
        
        # Call the internal predict function directly to avoid queue issues
        prediction_response = await _predict_internal(client_id)
        
        # Mark as force regenerated and update cache
        processing_duration = (datetime.now() - start_time).total_seconds()
        
        # Update the database record to mark it as force regenerated
        conn = None
        try:
            conn = get_projections_db_connection()
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE financial_projections SET force_regenerated_at = NOW() WHERE client_id = %s AND timeframe = %s",
                    (client_id, timeframe)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating force regeneration timestamp: {str(e)}")
        finally:
            if conn:
                conn.close()
        
        logger.info(f"🎉 Force regeneration completed successfully for client {client_id}")
        
        # Add force regeneration metadata to response
        if hasattr(prediction_response, 'model_dump'):
            response_dict = prediction_response.model_dump()
        elif hasattr(prediction_response, 'dict'):
            response_dict = prediction_response.dict()
        else:
            response_dict = prediction_response
            
        response_dict["from_cache"] = False
        response_dict["force_regenerated"] = True
        response_dict["generated_at"] = datetime.now().isoformat()
        response_dict["processing_time"] = processing_duration
        
        return response_dict
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"💥 Force regeneration error for client {client_id}: {str(e)}")
        error_details = {
            "client_id": client_id,
            "error_type": "force_regeneration_error",
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
        save_to_local_storage(error_details, 'errors', 'force_regeneration_error', client_id)
        raise HTTPException(status_code=500, detail=f"Force regeneration failed: {str(e)}")

@app.post("/predict/force-regenerate", response_model=Union[EnhancedProjectionSchema, QueueResponseSchema])
async def force_regenerate_prediction(
    client_id: str = Query(..., description="Client ID for financial analysis"),
    auth: Dict = Depends(verify_jwt_token)
):
    """
    Force regenerate projection with queue system (bypass cache) - JWT Protected
    """
    # Permission check: verify client_id matches authenticated user's client_id
    if str(client_id) != str(auth.get("client_id")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only access your own financial projections"
        )

    logger.info(f"🔄 FORCE REGENERATE REQUEST: Client ID: {client_id} (User: {auth['user_id']})")

    # Check if request should be queued
    is_queued, queue_info = await queue_projection_request(
        client_id,
        "force_regeneration",
        lambda: _force_regenerate_internal(client_id)
    )

    if is_queued:
        return queue_info

    try:
        # Execute force regeneration immediately
        result = await _force_regenerate_internal(client_id)

        # Process any queued requests from global queue
        await process_global_queue()

        return result

    except Exception as e:
        # Clean up processing state on error
        global currently_processing_client
        with queue_lock:
            currently_processing_client = None
            queue_stats["currently_processing"] = 0

        logger.error(f"❌ FORCE REGENERATION ERROR: Client {client_id}, Error: {str(e)}")
        raise

@app.get("/queue-status")
async def get_queue_status(
    client_id: str = Query(None, description="Optional client ID to get specific queue info"),
    auth: Dict = Depends(verify_jwt_token)
):
    """
    Get queue status - overall stats or specific client queue info (JWT Protected)
    """
    if client_id:
        # Permission check: verify client_id matches authenticated user's client_id
        if str(client_id) != str(auth.get("client_id")):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only check your own queue status"
            )
        # Check if this specific client is processing or queued
        is_processing = (currently_processing_client == client_id)

        # Find client's position in global queue
        queue_position = 0
        if not is_processing and global_queue.qsize() > 0:
            # Would need to peek at queue to find exact position
            # For simplicity, we'll estimate based on total queue size
            queue_position = global_queue.qsize()

        client_status = {
            "client_id": client_id,
            "is_processing": is_processing,
            "queue_position": queue_position,
            "estimated_wait_time": f"{queue_position * 45} seconds" if queue_position > 0 else "0 seconds",
            "status": "processing" if is_processing else ("queued" if queue_position > 0 else "available"),
            "currently_processing_client": currently_processing_client
        }

        return client_status
    else:
        # Get overall system queue stats
        overall_stats = {
            "system_status": "busy" if currently_processing_client else "available",
            "currently_processing_client": currently_processing_client,
            "total_queued_requests": global_queue.qsize(),
            "queue_stats": queue_stats,
            "estimated_completion_time": f"{global_queue.qsize() * 45} seconds" if global_queue.qsize() > 0 else "Now available"
        }

        return overall_stats

@app.post("/test-fresh-prediction")
async def test_fresh_prediction(client_id: str = Query(..., description="Client ID for testing fresh predictions")):
    """
    Test endpoint that completely bypasses cache and forces fresh AI generation
    """
    start_time = datetime.now()
    logger.info(f"🧪 TEST FRESH PREDICTION ENDPOINT: Client ID: {client_id}")
    
    try:
        # Step 1: Get financial data directly
        combined_financial_content = get_financial_data_from_database(client_id)
        logger.info(f"📊 Got {len(combined_financial_content) if combined_financial_content else 0} chars of financial data")
        
        if not combined_financial_content:
            return {"error": "No financial data found", "client_id": client_id}
        
        # Step 2: Prepare AI prompt (use the improved version)
        base_prompt = """
ROLE
You are a financial analytics system that analyzes historical data and creates projections based solely on patterns found in that data.

CRITICAL REQUIREMENT: You MUST extract and calculate actual numerical values from the provided financial documents. DO NOT return zero values.

TASK
1. CAREFULLY extract all financial data from the provided documents (look for Revenue, Sales, Income, Expenses, Costs, Profit figures)
2. Calculate the actual statistical properties and trends from this specific client's data
3. Project forward using these exact patterns found in the data - do not assume growth if data shows decline
4. Generate projections that logically follow the historical trend direction

HISTORICAL BASELINE CALCULATION (CRITICAL):
Calculate from the most recent complete 12-month period in the data:
- current_annual_revenue: Sum of most recent 12 months revenue (extract actual figures from documents)
- current_annual_expenses: Sum of most recent 12 months operating expenses (extract actual figures)
- current_annual_net_profit: Sum of most recent 12 months net profit (calculate: revenue - expenses)
- current_annual_gross_profit: Sum of most recent 12 months gross profit (extract from documents)
- baseline_period: Exact period this represents
- revenue_growth_rate: Actual calculated year-over-year revenue growth rate (positive for growth, negative for decline)
- expense_inflation_rate: Actual calculated expense growth rate
- profit_margin_trend: Actual profit margin from the baseline period

VALIDATION: Before responding, verify revenue is NOT zero and matches actual business scale found in documents.
"""
        
        # Step 3: Simple test response with extracted baseline only
        response_data = {
            "test_mode": True,
            "client_id": client_id,
            "financial_data_length": len(combined_financial_content),
            "timestamp": datetime.now().isoformat(),
            "baseline_test": "Testing if AI can extract non-zero values",
            "processing_time": (datetime.now() - start_time).total_seconds()
        }
        
        logger.info(f"🧪 Test response generated for client {client_id}")
        return response_data
        
    except Exception as e:
        logger.error(f"🧪 Test endpoint failed: {str(e)}")
        return {"error": str(e), "client_id": client_id}


@app.get("/auth/me")
async def get_authenticated_user(auth: Dict = Depends(verify_jwt_token)):
    """
    Get authenticated user information from JWT token
    This endpoint allows the frontend to verify authentication and get user_id
    """
    return {
        "status": "success",
        "user_id": str(auth["user_id"]),
        "client_id": auth.get("client_id"),
        "email": auth.get("email"),
        "authenticated": True
    }

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint accessed")
    
    health_check_start = datetime.now()
    health_data = {
        "endpoint": "/health",
        "check_timestamp": health_check_start.isoformat(),
        "status": "checking"
    }
    
    try:
        # Test finance database connection
        logger.debug("Testing finance database connection for health check")
        conn = psycopg2.connect(
            host=FINANCE_DB_HOST,
            dbname=FINANCE_DB_NAME,
            user=FINANCE_DB_USER,
            password=FINANCE_DB_PASSWORD,
            port=FINANCE_DB_PORT,
            connect_timeout=5
        )
        
        # Test a simple query to ensure database is functioning
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM finance_documents LIMIT 1")
            document_count = cur.fetchone()[0]
            logger.debug(f"Finance database query successful. Total documents: {document_count}")
        
        conn.close()
        logger.debug("Finance database connection test successful")
        
        # Test storage directories
        storage_status = {}
        for category, subdir in STORAGE_SUBDIRS.items():
            full_path = Path(LOCAL_STORAGE_BASE_PATH) / subdir
            storage_status[category] = {
                "exists": full_path.exists(),
                "path": str(full_path),
                "writable": os.access(full_path, os.W_OK) if full_path.exists() else False
            }
        
        health_check_end = datetime.now()
        check_duration = (health_check_end - health_check_start).total_seconds()
        
        health_response = {
            "status": "healthy", 
            "timestamp": health_check_end.isoformat(), 
            "version": "3.2.0",
            "finance_database": "connected",
            "total_financial_documents": document_count,
            "data_source": "postgresql_finance_engine",
            "gemini_keys_available": len(GEMINI_API_KEYS),
            "local_storage": {
                "enabled": LOCAL_STORAGE_ENABLED,
                "base_path": LOCAL_STORAGE_BASE_PATH,
                "directories_status": storage_status,
                "detailed_logging": DETAILED_LOGGING_ENABLED
            },
            "check_duration_seconds": round(check_duration, 3),
            "features": {
                "goal_based_projections": True,
                "database_integration": True,
                "multi_document_analysis": True,
                "financial_ratios_integration": True,
                "complete_local_storage": True,
                "detailed_logging": True
            }
        }
        
        # Update health data for storage
        health_data.update({
            "status": "healthy",
            "check_duration_seconds": check_duration,
            "database_connection": "successful",
            "document_count": document_count,
            "storage_directories_ok": all(status["exists"] for status in storage_status.values())
        })
        
        save_to_local_storage(health_data, 'health_checks', 'health_check_success')
        logger.info(f"Health check completed successfully in {check_duration:.3f}s")
        
        return health_response
        
    except Exception as e:
        health_check_end = datetime.now()
        check_duration = (health_check_end - health_check_start).total_seconds()
        
        logger.error(f"Health check failed: {str(e)}")
        
        health_failure = {
            "status": "unhealthy",
            "error": str(e),
            "check_duration_seconds": check_duration,
            "traceback": traceback.format_exc(),
            "timestamp": health_check_end.isoformat()
        }
        
        save_to_local_storage(health_failure, 'health_checks', 'health_check_failure')
        
        return {
            "status": "unhealthy", 
            "timestamp": health_check_end.isoformat(), 
            "version": "3.2.0",
            "error": str(e),
            "check_duration_seconds": round(check_duration, 3),
            "data_source": "postgresql_finance_engine",
            "local_storage": {
                "enabled": LOCAL_STORAGE_ENABLED,
                "base_path": LOCAL_STORAGE_BASE_PATH
            }
        }

@app.get("/api-keys-status")
async def check_api_keys_status():
    """
    Check the status and availability of API keys
    """
    logger.info("API keys status check requested")
    
    api_status_request = {
        "endpoint": "/api-keys-status",
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        total_keys = len(GEMINI_API_KEYS)
        current_index = current_key_index
        
        logger.debug(f"Total API keys: {total_keys}, Current index: {current_index}")
        
        api_status_data = {
            "total_api_keys": total_keys,
            "current_key_index": current_index,
            "api_keys_available": True,
            "rotation_enabled": True,
            "retry_mechanism": "3 attempts with different keys",
            "status": "healthy",
            "key_preview": [key[:20] + "..." for key in GEMINI_API_KEYS[:5]],  # Show first 5 keys preview
            "check_timestamp": datetime.now().isoformat()
        }
        
        api_status_request.update(api_status_data)
        save_to_local_storage(api_status_request, 'system_logs', 'api_keys_status_check')
        
        return api_status_data
        
    except Exception as e:
        logger.error(f"API keys status check failed: {str(e)}")
        
        api_status_error = {
            "error_type": "api_keys_status_error",
            "error_message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        save_to_local_storage(api_status_error, 'errors', 'api_keys_status_error')
        
        raise HTTPException(status_code=500, detail=f"API keys status check failed: {str(e)}")

@app.get("/test-azure-connection/{client_id}")
async def test_azure_connection(client_id: str):
    """
    Test endpoint to verify Azure connection and document retrieval for a client
    """
    logger.info(f"Testing Azure connection for client_id: {client_id}")
    
    azure_test_request = {
        "client_id": client_id,
        "endpoint": "/test-azure-connection",
        "timestamp": datetime.now().isoformat()
    }
    save_to_local_storage(azure_test_request, 'system_logs', 'azure_connection_test_request', client_id)
    
    try:
        logger.debug("Getting container name from database")
        # Get container name and client folder
        container_name = get_azure_container_name(client_id)
        client_folder = get_client_folder_name(client_id)

        logger.debug("Initializing Azure Blob Service Client")
        # Initialize Azure Blob Service Client
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(container_name)

        logger.debug(f"Listing blobs in {client_folder}/financial intelligence report folder")
        # List blobs in the client's financial intelligence report folder
        folder_prefix = f"{client_folder}/financial intelligence report/"
        blobs = list(container_client.list_blobs(name_starts_with=folder_prefix))
        logger.debug(f"Found {len(blobs)} total blobs in folder")
        
        # Filter for finance documents
        finance_blobs = [blob for blob in blobs if "finance_report_" in blob.name and blob.name.endswith('.docx')]
        logger.debug(f"Filtered to {len(finance_blobs)} finance documents")
        
        azure_test_result = {
            "client_id": client_id,
            "container_name": container_name,
            "total_blobs_in_folder": len(blobs),
            "finance_document_count": len(finance_blobs),
            "finance_documents": [blob.name for blob in finance_blobs],
            "status": "success" if finance_blobs else "no_documents_found",
            "test_timestamp": datetime.now().isoformat(),
            "blob_details": [
                {
                    "name": blob.name,
                    "size": blob.size,
                    "last_modified": blob.last_modified.isoformat() if blob.last_modified else None
                }
                for blob in finance_blobs
            ]
        }
        
        save_to_local_storage(azure_test_result, 'system_logs', 'azure_connection_test_result', client_id)
        
        logger.info(f"Azure connection test completed. Status: {azure_test_result['status']}")
        return azure_test_result
        
    except Exception as e:
        logger.error(f"Azure connection test failed: {str(e)}")
        
        azure_test_error = {
            "client_id": client_id,
            "error_type": "azure_connection_test_error",
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }
        save_to_local_storage(azure_test_error, 'errors', 'azure_connection_test_error', client_id)
        
        raise HTTPException(status_code=500, detail=f"Azure connection test failed: {str(e)}")

@app.get("/local-storage-status")
async def local_storage_status():
    """Check local storage status and statistics with detailed information"""
    logger.info("Local storage status check requested")
    
    status_request = {
        "endpoint": "/local-storage-status",
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        if not LOCAL_STORAGE_ENABLED:
            disabled_status = {
                "status": "disabled", 
                "enabled": False,
                "message": "Local storage is disabled in configuration"
            }
            save_to_local_storage(disabled_status, 'system_logs', 'storage_status_disabled')
            return disabled_status
        
        ensure_storage_directories()
        
        # Count files by type and collect statistics
        file_counts = {}
        total_size = 0
        recent_files = []
        oldest_files = []
        
        for category, subdir in STORAGE_SUBDIRS.items():
            category_path = Path(LOCAL_STORAGE_BASE_PATH) / subdir
            if category_path.exists():
                category_files = [f for f in category_path.iterdir() if f.is_file() and f.suffix == '.json']
                file_counts[category] = {
                    "count": len(category_files),
                    "size_bytes": sum(f.stat().st_size for f in category_files),
                    "path": str(category_path)
                }
                
                # Add to total size
                total_size += file_counts[category]["size_bytes"]
                
                # Collect file details for recent/oldest tracking
                for f in category_files:
                    file_info = {
                        "name": f.name,
                        "category": category,
                        "size": f.stat().st_size,
                        "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                        "created": datetime.fromtimestamp(f.stat().st_ctime).isoformat()
                    }
                    recent_files.append(file_info)
        
        # Sort files by modification time
        recent_files.sort(key=lambda x: x['modified'], reverse=True)
        oldest_files = sorted(recent_files, key=lambda x: x['modified'])
        
        # Calculate total files
        total_files = sum(cat['count'] for cat in file_counts.values())
        
        storage_status_data = {
            "status": "enabled",
            "enabled": True,
            "storage_path": LOCAL_STORAGE_BASE_PATH,
            "detailed_logging": DETAILED_LOGGING_ENABLED,
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024*1024), 2),
            "total_size_gb": round(total_size / (1024*1024*1024), 3),
            "categories": file_counts,
            "recent_files": recent_files[:10],  # Last 10 files
            "oldest_files": oldest_files[:5],   # First 5 files
            "storage_health": {
                "all_directories_exist": len(file_counts) == len(STORAGE_SUBDIRS),
                "writable": os.access(LOCAL_STORAGE_BASE_PATH, os.W_OK),
                "free_space_gb": round(os.statvfs(LOCAL_STORAGE_BASE_PATH).f_bavail * os.statvfs(LOCAL_STORAGE_BASE_PATH).f_frsize / (1024**3), 2) if hasattr(os, 'statvfs') else "unknown"
            },
            "check_timestamp": datetime.now().isoformat()
        }
        
        save_to_local_storage(storage_status_data, 'system_logs', 'storage_status_check')
        
        logger.info(f"Local storage status: {total_files} files, {storage_status_data['total_size_mb']} MB total")
        return storage_status_data
        
    except Exception as e:
        logger.error(f"Local storage status check failed: {str(e)}")
        
        storage_error = {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }
        save_to_local_storage(storage_error, 'errors', 'storage_status_error')
        
        return storage_error

@app.get("/storage-cleanup")
async def storage_cleanup(
    older_than_days: int = Query(7, description="Delete files older than this many days"),
    category: Optional[str] = Query(None, description="Specific category to clean up"),
    dry_run: bool = Query(True, description="If true, only show what would be deleted")
):
    """Clean up old files from local storage"""
    logger.info(f"Storage cleanup requested: older_than_days={older_than_days}, category={category}, dry_run={dry_run}")
    
    cleanup_request = {
        "older_than_days": older_than_days,
        "category": category,
        "dry_run": dry_run,
        "endpoint": "/storage-cleanup",
        "timestamp": datetime.now().isoformat()
    }
    save_to_local_storage(cleanup_request, 'system_logs', 'cleanup_request')
    
    if not LOCAL_STORAGE_ENABLED:
        return {"message": "Local storage is disabled", "files_processed": 0}
    
    try:
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        files_to_delete = []
        total_size_to_free = 0
        
        categories_to_process = [category] if category else STORAGE_SUBDIRS.keys()
        
        for cat in categories_to_process:
            if cat not in STORAGE_SUBDIRS:
                continue
                
            category_path = Path(LOCAL_STORAGE_BASE_PATH) / STORAGE_SUBDIRS[cat]
            if not category_path.exists():
                continue
                
            for file_path in category_path.iterdir():
                if file_path.is_file() and file_path.suffix == '.json':
                    file_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_modified < cutoff_date:
                        file_info = {
                            "path": str(file_path),
                            "category": cat,
                            "size": file_path.stat().st_size,
                            "modified": file_modified.isoformat(),
                            "name": file_path.name
                        }
                        files_to_delete.append(file_info)
                        total_size_to_free += file_info["size"]
        
        cleanup_result = {
            "dry_run": dry_run,
            "older_than_days": older_than_days,
            "cutoff_date": cutoff_date.isoformat(),
            "files_found": len(files_to_delete),
            "total_size_to_free_mb": round(total_size_to_free / (1024*1024), 2),
            "categories_processed": list(categories_to_process),
            "files_details": files_to_delete
        }
        
        if not dry_run and files_to_delete:
            deleted_count = 0
            for file_info in files_to_delete:
                try:
                    os.remove(file_info["path"])
                    deleted_count += 1
                    logger.debug(f"Deleted: {file_info['path']}")
                except Exception as e:
                    logger.warning(f"Failed to delete {file_info['path']}: {str(e)}")
            
            cleanup_result["files_deleted"] = deleted_count
            cleanup_result["deletion_completed"] = True
            logger.info(f"Cleanup completed: {deleted_count} files deleted, {cleanup_result['total_size_to_free_mb']} MB freed")
        else:
            cleanup_result["message"] = "Dry run - no files deleted" if dry_run else "No files found to delete"
        
        save_to_local_storage(cleanup_result, 'system_logs', 'cleanup_result')
        return cleanup_result
        
    except Exception as e:
        logger.error(f"Storage cleanup failed: {str(e)}")
        
        cleanup_error = {
            "error_type": "storage_cleanup_error",
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }
        save_to_local_storage(cleanup_error, 'errors', 'cleanup_error')
        
        raise HTTPException(status_code=500, detail=f"Storage cleanup failed: {str(e)}")

# Initialize storage on startup
if LOCAL_STORAGE_ENABLED:
    try:
        ensure_storage_directories()
        startup_info = {
            "event": "application_startup",
            "storage_enabled": True,
            "storage_path": LOCAL_STORAGE_BASE_PATH,
            "detailed_logging": DETAILED_LOGGING_ENABLED,
            "gemini_keys_count": len(GEMINI_API_KEYS),
            "timestamp": datetime.now().isoformat()
        }
        save_to_local_storage(startup_info, 'system_logs', 'application_startup')
        logger.info(f"Local storage initialized at: {LOCAL_STORAGE_BASE_PATH}")
    except Exception as e:
        logger.error(f"Failed to initialize local storage: {str(e)}")
else:
    logger.info("Local storage is disabled")

# Log application startup with Vertex AI status
if vertex_ai_client:
    logger.info(f"✅ Vertex AI initialized successfully (Project: {VERTEX_PROJECT_ID}, Location: {VERTEX_LOCATION})")
    logger.info("🎯 Using Vertex AI as PRIMARY method with API keys as fallback")
else:
    logger.warning("⚠️ Vertex AI not available - using API keys only")

logger.info("Enhanced Financial Projection API v3.3 with Vertex AI Support initialized")
logger.info(f"🔑 Loaded {len(GEMINI_API_KEYS)} API keys for fallback")
logger.info(f"Local storage: {'ENABLED' if LOCAL_STORAGE_ENABLED else 'DISABLED'}")
logger.info(f"Detailed logging: {'ENABLED' if DETAILED_LOGGING_ENABLED else 'DISABLED'}")

if __name__ == "__main__":
    import uvicorn
    
    # Log server startup
    server_startup = {
        "event": "server_startup",
        "host": "0.0.0.0",
        "port": 8005,
        "version": "3.2.0",
        "features": {
            "local_storage": LOCAL_STORAGE_ENABLED,
            "detailed_logging": DETAILED_LOGGING_ENABLED,
            "database_integration": True
        },
        "timestamp": datetime.now().isoformat()
    }
    
    if LOCAL_STORAGE_ENABLED:
        save_to_local_storage(server_startup, 'system_logs', 'server_startup')
    
    logger.info("Starting Enhanced Financial Projection API v3.2...")
    logger.info("Server starting on host 0.0.0.0, port 8005")
    
    uvicorn.run(app, host="0.0.0.0", port=8005)