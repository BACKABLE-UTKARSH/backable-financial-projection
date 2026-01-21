# Add these imports at the top of the file (after existing imports)
from google.oauth2 import service_account
from dotenv import load_dotenv
import tempfile
import threading

# Load environment variables
load_dotenv()

# ======================================================
#           Vertex AI Configuration (Primary Method)
# ======================================================
VERTEX_PROJECT_ID = "backable-machine-learning-apis"
VERTEX_LOCATION = "us-central1"
USE_VERTEX_AI = True  # Primary method - will fallback to API keys if fails

# Keep existing API keys as fallback
GEMINI_API_KEYS = [
    "AIzaSyA1YGlaMM7Cx0KaduaWHZZGtAqPSuCD34s",  # Back_Fin11
    "AIzaSyC7rJFOZQeiDscqFdA7Cu96p8cV0fTkB_A",  # Back_Fin12
    "AIzaSyCOFZqiEXQdYZJdue0oQINhVkusIR5Q13M",  # Back_Fin13
    "AIzaSyDqo_HCxO-UnQVwIY5G_3fWrPpnc3oZvGc",  # Back_Fin14
    "AIzaSyAO35odPsSyelz3qZ9Nfass5qyZbrditnw",  # Back_Fin15
    # ... keep all existing keys
]

# API Key Management Variables (for fallback)
api_key_stats = defaultdict(lambda: {"requests": 0, "failures": 0, "last_used": 0, "cooldown_until": 0})
api_key_lock = threading.Lock()
current_key_index = 0

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
            logging.info("Loading Vertex AI credentials from environment variable")
            import tempfile
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
                logging.info(f"Loading Vertex AI credentials from {creds_file}")
                credentials = service_account.Credentials.from_service_account_file(
                    creds_file,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
            else:
                logging.warning("No Vertex AI credentials found - will use API keys fallback")
                return None

        # Initialize GenAI client for Vertex AI
        client = genai.Client(
            vertexai=True,
            credentials=credentials,
            project=VERTEX_PROJECT_ID,
            location=VERTEX_LOCATION
        )

        logging.info(f"✅ Vertex AI GenAI client initialized successfully (Project: {VERTEX_PROJECT_ID})")
        return client

    except Exception as e:
        logging.warning(f"⚠️ Vertex AI initialization failed: {str(e)} - Will use API keys fallback")
        return None

# Initialize Vertex AI client at startup
vertex_ai_client = initialize_vertex_ai_client() if USE_VERTEX_AI else None

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
        logging.info("Vertex AI client not available, skipping to API keys fallback")
        return None

    try:
        logging.info("🚀 Trying Vertex AI (Primary Method for Financial Projections)")

        # Call Vertex AI using GenAI SDK with gemini-2.5-pro
        response = vertex_ai_client.models.generate_content(
            model="gemini-2.5-pro",
            contents=combined_financial_content,
            config=config
        )

        if response and response.text:
            logging.info(f"✅ Vertex AI request successful for client {client_id}")

            # Track token usage if available
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                token_usage = {
                    "input_tokens": getattr(usage, 'prompt_token_count', 0),
                    "output_tokens": getattr(usage, 'candidates_token_count', 0),
                    "thinking_tokens": getattr(usage, 'thoughts_token_count', 0),
                    "total_tokens": getattr(usage, 'total_token_count', 0),
                    "method": "vertex_ai"
                }
                logging.info(f"🧮 Vertex AI Token Usage - Input: {token_usage['input_tokens']:,} | Output: {token_usage['output_tokens']:,} | Total: {token_usage['total_tokens']:,}")

            return response
        else:
            logging.warning(f"⚠️ Vertex AI returned empty response for client {client_id}")
            return None

    except Exception as e:
        logging.warning(f"❌ Vertex AI request failed: {str(e)} - Falling back to API keys")
        return None

# Modified generate_projection function to use Vertex AI first
async def generate_projection_with_vertex_ai(client_id: str, combined_financial_content: str, prompt: str):
    """
    Generate projection using Vertex AI as primary method with API keys as fallback
    """
    start_time = time.time()

    # Prepare configuration for Vertex AI
    config = {
        "temperature": 0.0,  # Deterministic
        "max_output_tokens": 8192,  # Explicit token limit
        "top_p": 1.0,
        "top_k": 1,
        "response_mime_type": "application/json",
        "response_schema": EnhancedProjectionSchema,  # Your existing schema
        "thinking_config": types.ThinkingConfig(
            thinking_budget=32768
        ),
        "system_instruction": prompt
    }

    # ===================================================================
    # STEP 1: TRY VERTEX AI (PRIMARY METHOD)
    # ===================================================================
    vertex_result = try_vertex_ai_projection_request(
        combined_financial_content=combined_financial_content,
        config=config,
        client_id=client_id
    )

    if vertex_result:
        # Vertex AI succeeded - parse and return
        total_time = time.time() - start_time
        logging.info(f"🎉 [{client_id}] Projection completed via Vertex AI in {total_time:.2f}s")

        try:
            parsed_response = json.loads(vertex_result.text)
            return {
                "success": True,
                "method": "vertex_ai",
                "data": parsed_response,
                "processing_time": total_time
            }
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse Vertex AI response: {str(e)}")
            # Continue to fallback

    # ===================================================================
    # STEP 2: FALLBACK TO API KEYS
    # ===================================================================
    logging.info(f"Falling back to API keys for client {client_id}")

    # Use existing API key logic
    for retry_count in range(3):  # Max 3 retries
        try:
            # Get deterministic API key for consistent results
            deterministic_key = get_deterministic_api_key(client_id)
            current_client = create_gemini_client(api_key=deterministic_key, client_id=client_id)

            logging.info(f"🔄 Attempt {retry_count + 1}/3 - Using API key fallback")

            # Create config for API key method (slightly different format)
            api_config = types.GenerateContentConfig(
                temperature=0.0,
                top_p=1.0,
                top_k=1,
                response_mime_type="application/json",
                response_schema=EnhancedProjectionSchema,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=32768
                ),
                system_instruction=prompt
            )

            response = current_client.models.generate_content(
                model="gemini-2.5-pro",
                contents=combined_financial_content,
                config=api_config
            )

            if response and response.text:
                total_time = time.time() - start_time
                logging.info(f"✅ [{client_id}] Projection completed via API keys in {total_time:.2f}s")

                parsed_response = json.loads(response.text)
                return {
                    "success": True,
                    "method": "api_keys",
                    "data": parsed_response,
                    "processing_time": total_time
                }

        except Exception as e:
            logging.error(f"API key attempt {retry_count + 1} failed: {str(e)}")
            if retry_count == 2:  # Last attempt
                raise HTTPException(status_code=500, detail=f"All projection methods failed: {str(e)}")

    raise HTTPException(status_code=500, detail="Failed to generate projection with both Vertex AI and API keys")

# Update startup logging
if __name__ == "__main__":
    # Log Vertex AI status
    if vertex_ai_client:
        logger.info(f"✅ Vertex AI initialized successfully (Project: {VERTEX_PROJECT_ID}, Location: {VERTEX_LOCATION})")
        logger.info("🎯 Using Vertex AI as PRIMARY method with API keys as fallback")
    else:
        logger.warning("⚠️ Vertex AI not available - using API keys only")

    logger.info(f"🔑 Loaded {len(GEMINI_API_KEYS)} API keys for fallback")
    logger.info("Enhanced Financial Projection API v3.3 with Vertex AI support initialized")