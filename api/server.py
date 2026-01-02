"""
FastAPI Backend Server

This server provides REST API endpoints for interacting with both the base and
fine-tuned LLM models. It connects to local Ollama instance and provides:
- Text generation from base or fine-tuned model
- Model information and availability
- Side-by-side model comparison
- Health checks

The server uses pydantic-ai to communicate with Ollama via OpenAI-compatible API.

Endpoints:
- POST /api/generate - Generate text from a model
- GET /api/models - List available models
- POST /api/compare - Compare base vs fine-tuned outputs
- GET /health - Health check
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import requests
from typing import Dict

# Import our Pydantic models
from .models.schemas import (
    GenerateRequest,
    GenerateResponse,
    ModelsResponse,
    ModelInfo,
    ModelType,
    CompareRequest,
    CompareResponse,
    ComparisonResult,
    HealthResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="LLM Blogging Twin API",
    description="API for interacting with base and fine-tuned LLM models",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI at /docs
    redoc_url="/redoc"  # ReDoc at /redoc
)

# ============================================================================
# CORS Configuration
# ============================================================================
# This allows the React frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Create React App default
        "http://localhost:5173",  # Vite default
        "http://localhost:5174",  # Vite alternate
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# ============================================================================
# Configuration
# ============================================================================
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_URL = "http://localhost:11434"
BASE_MODEL_NAME = "llama3.2:1b"
FINETUNED_MODEL_NAME = "blogging-twin:latest"

# ============================================================================
# Model Initialization
# ============================================================================
# Initialize pydantic-ai models for both base and fine-tuned versions
# These will be used throughout the API
logger.info("Initializing models...")

try:
    # Base model
    base_model = OpenAIModel(
        model_name=BASE_MODEL_NAME,
        provider=OpenAIProvider(base_url=OLLAMA_BASE_URL)
    )
    logger.info(f"✓ Base model initialized: {BASE_MODEL_NAME}")

    # Fine-tuned model
    finetuned_model = OpenAIModel(
        model_name=FINETUNED_MODEL_NAME,
        provider=OpenAIProvider(base_url=OLLAMA_BASE_URL)
    )
    logger.info(f"✓ Fine-tuned model initialized: {FINETUNED_MODEL_NAME}")

except Exception as e:
    logger.error(f"Error initializing models: {e}")
    logger.error("Models will be initialized on first request")


# ============================================================================
# Helper Functions
# ============================================================================

def format_instruction_prompt(prompt: str) -> str:
    """
    Format user prompt into instruction format.

    Args:
        prompt: User's input prompt

    Returns:
        Formatted prompt string
    """
    return f"""### Instruction:
{prompt}

### Response:
"""


def check_ollama_connection() -> bool:
    """
    Check if Ollama server is running and accessible.

    Returns:
        True if connected, False otherwise
    """
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def check_model_available(model_name: str) -> bool:
    """
    Check if a specific model is available in Ollama.

    Args:
        model_name: Name of the model to check

    Returns:
        True if model is available, False otherwise
    """
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return any(model['name'] == model_name for model in models)
        return False
    except:
        return False


async def generate_with_ollama(
    model_name: str,
    prompt: str,
    max_tokens: int = 500,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> Dict:
    """
    Generate text using Ollama API directly.

    This bypasses pydantic-ai for more control over generation parameters.

    Args:
        model_name: Name of the model in Ollama
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter

    Returns:
        Dictionary with response and metadata
    """
    try:
        # Format the prompt
        formatted_prompt = format_instruction_prompt(prompt)

        # Call Ollama API
        response = requests.post(
            f"{OLLAMA_API_URL}/api/generate",
            json={
                "model": model_name,
                "prompt": formatted_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens,
                }
            },
            timeout=120  # 2 minute timeout for generation
        )

        if response.status_code == 200:
            result = response.json()
            return {
                'success': True,
                'text': result.get('response', ''),
                'tokens_generated': result.get('eval_count', 0),
                'generation_time': result.get('total_duration', 0) / 1e9  # Convert nanoseconds to seconds
            }
        else:
            return {
                'success': False,
                'error': f"Ollama API returned status {response.status_code}",
                'text': ''
            }

    except requests.exceptions.Timeout:
        return {
            'success': False,
            'error': "Generation timed out (>2 minutes)",
            'text': ''
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'text': ''
        }


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """
    Root endpoint - provides basic API information.
    """
    return {
        "message": "LLM Blogging Twin API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns API status and Ollama connection info.
    """
    ollama_connected = check_ollama_connection()

    models_count = 0
    if ollama_connected:
        if check_model_available(BASE_MODEL_NAME):
            models_count += 1
        if check_model_available(FINETUNED_MODEL_NAME):
            models_count += 1

    return HealthResponse(
        status="healthy" if ollama_connected else "degraded",
        ollama_connected=ollama_connected,
        models_available=models_count
    )


@app.get("/api/models", response_model=ModelsResponse)
async def get_models():
    """
    Get information about available models.

    Returns list of models with their status and descriptions.
    """
    base_available = check_model_available(BASE_MODEL_NAME)
    finetuned_available = check_model_available(FINETUNED_MODEL_NAME)

    models = [
        ModelInfo(
            name=BASE_MODEL_NAME,
            type=ModelType.BASE,
            available=base_available,
            description="Base Llama 3.2 1B model - General purpose language model"
        ),
        ModelInfo(
            name=FINETUNED_MODEL_NAME,
            type=ModelType.FINETUNED,
            available=finetuned_available,
            description="Fine-tuned blogging twin - Trained on your personal blog posts"
        )
    ]

    return ModelsResponse(models=models)


@app.post("/api/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """
    Generate text from the specified model.

    This is the main generation endpoint. It accepts a prompt and generates
    text using either the base or fine-tuned model.

    Args:
        request: GenerateRequest with prompt and generation parameters

    Returns:
        GenerateResponse with generated text and metadata
    """
    logger.info(f"Generate request - Model: {request.model}, Prompt: {request.prompt[:50]}...")

    # Determine which model to use
    model_name = FINETUNED_MODEL_NAME if request.model == ModelType.FINETUNED else BASE_MODEL_NAME

    # Check if model is available
    if not check_model_available(model_name):
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_name}' is not available. Please ensure it's loaded in Ollama."
        )

    # Generate text
    result = await generate_with_ollama(
        model_name=model_name,
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p
    )

    if not result['success']:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {result.get('error', 'Unknown error')}"
        )

    return GenerateResponse(
        success=True,
        text=result['text'],
        model_used=model_name,
        tokens_generated=result.get('tokens_generated'),
        generation_time=result.get('generation_time')
    )


@app.post("/api/compare", response_model=CompareResponse)
async def compare_models(request: CompareRequest):
    """
    Generate text from both base and fine-tuned models for comparison.

    This endpoint is useful for evaluating the fine-tuned model against
    the base model on the same prompt.

    Args:
        request: CompareRequest with prompt and generation parameters

    Returns:
        CompareResponse with results from both models
    """
    logger.info(f"Compare request - Prompt: {request.prompt[:50]}...")

    # Check if both models are available
    if not check_model_available(BASE_MODEL_NAME):
        raise HTTPException(
            status_code=503,
            detail=f"Base model '{BASE_MODEL_NAME}' is not available"
        )

    if not check_model_available(FINETUNED_MODEL_NAME):
        raise HTTPException(
            status_code=503,
            detail=f"Fine-tuned model '{FINETUNED_MODEL_NAME}' is not available"
        )

    # Generate from base model
    base_result = await generate_with_ollama(
        model_name=BASE_MODEL_NAME,
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )

    # Generate from fine-tuned model
    finetuned_result = await generate_with_ollama(
        model_name=FINETUNED_MODEL_NAME,
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )

    return CompareResponse(
        prompt=request.prompt,
        base_model=ComparisonResult(
            model_name=BASE_MODEL_NAME,
            model_type=ModelType.BASE,
            text=base_result.get('text', ''),
            tokens_generated=base_result.get('tokens_generated'),
            generation_time=base_result.get('generation_time'),
            success=base_result['success']
        ),
        finetuned_model=ComparisonResult(
            model_name=FINETUNED_MODEL_NAME,
            model_type=ModelType.FINETUNED,
            text=finetuned_result.get('text', ''),
            tokens_generated=finetuned_result.get('tokens_generated'),
            generation_time=finetuned_result.get('generation_time'),
            success=finetuned_result['success']
        )
    )


# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Runs when the server starts up.

    Logs startup information and checks Ollama connection.
    """
    logger.info("=" * 60)
    logger.info("LLM Blogging Twin API - Server Starting")
    logger.info("=" * 60)
    logger.info(f"Ollama base URL: {OLLAMA_BASE_URL}")
    logger.info(f"Base model: {BASE_MODEL_NAME}")
    logger.info(f"Fine-tuned model: {FINETUNED_MODEL_NAME}")
    logger.info("")

    # Check Ollama connection
    if check_ollama_connection():
        logger.info("✓ Ollama server is connected")

        # Check model availability
        if check_model_available(BASE_MODEL_NAME):
            logger.info(f"✓ Base model '{BASE_MODEL_NAME}' is available")
        else:
            logger.warning(f"⚠ Base model '{BASE_MODEL_NAME}' not found")

        if check_model_available(FINETUNED_MODEL_NAME):
            logger.info(f"✓ Fine-tuned model '{FINETUNED_MODEL_NAME}' is available")
        else:
            logger.warning(f"⚠ Fine-tuned model '{FINETUNED_MODEL_NAME}' not found")
    else:
        logger.error("✗ Cannot connect to Ollama server")
        logger.error("  Please ensure Ollama is running: ollama serve")

    logger.info("")
    logger.info("Server ready! API docs available at /docs")
    logger.info("=" * 60)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
