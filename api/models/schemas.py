"""
API Schema Models

Pydantic models for request/response validation in the FastAPI backend.
These ensure type safety and automatic API documentation.

Models:
- GenerateRequest: Request body for text generation
- GenerateResponse: Response format for generated text
- ModelInfo: Information about available models
- CompareRequest: Request for side-by-side comparison
- CompareResponse: Comparison results
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class ModelType(str, Enum):
    """
    Enum for available model types.
    """
    BASE = "base"
    FINETUNED = "finetuned"


class GenerateRequest(BaseModel):
    """
    Request model for text generation.

    Example:
        {
            "prompt": "Write about AI",
            "model": "finetuned",
            "max_tokens": 500,
            "temperature": 0.7
        }
    """
    prompt: str = Field(
        ...,
        description="The input prompt for text generation",
        min_length=1,
        max_length=5000,
        example="Write a blog post about artificial intelligence"
    )

    model: ModelType = Field(
        default=ModelType.FINETUNED,
        description="Which model to use: 'base' or 'finetuned'"
    )

    max_tokens: int = Field(
        default=500,
        description="Maximum number of tokens to generate",
        ge=50,
        le=2000
    )

    temperature: float = Field(
        default=0.7,
        description="Sampling temperature (higher = more creative)",
        ge=0.0,
        le=2.0
    )

    top_p: float = Field(
        default=0.9,
        description="Nucleus sampling parameter",
        ge=0.0,
        le=1.0
    )


class GenerateResponse(BaseModel):
    """
    Response model for text generation.

    Example:
        {
            "success": true,
            "text": "Generated blog post...",
            "model_used": "blogging-twin:latest",
            "tokens_generated": 342,
            "generation_time": 2.5
        }
    """
    success: bool = Field(
        ...,
        description="Whether generation was successful"
    )

    text: str = Field(
        ...,
        description="The generated text"
    )

    model_used: str = Field(
        ...,
        description="Name of the model that generated the text"
    )

    tokens_generated: Optional[int] = Field(
        None,
        description="Number of tokens in the generated text"
    )

    generation_time: Optional[float] = Field(
        None,
        description="Time taken to generate (in seconds)"
    )

    error: Optional[str] = Field(
        None,
        description="Error message if generation failed"
    )


class ModelInfo(BaseModel):
    """
    Information about a model.

    Example:
        {
            "name": "llama3.2:1b",
            "type": "base",
            "available": true,
            "description": "Base Llama 3.2 1B model"
        }
    """
    name: str = Field(
        ...,
        description="Model name in Ollama"
    )

    type: ModelType = Field(
        ...,
        description="Model type: base or finetuned"
    )

    available: bool = Field(
        ...,
        description="Whether the model is currently available"
    )

    description: str = Field(
        ...,
        description="Human-readable description of the model"
    )


class ModelsResponse(BaseModel):
    """
    Response with information about all available models.

    Example:
        {
            "models": [
                {"name": "llama3.2:1b", "type": "base", ...},
                {"name": "blogging-twin:latest", "type": "finetuned", ...}
            ]
        }
    """
    models: List[ModelInfo] = Field(
        ...,
        description="List of available models"
    )


class CompareRequest(BaseModel):
    """
    Request model for comparing base and fine-tuned models.

    Example:
        {
            "prompt": "Write about AI",
            "max_tokens": 500,
            "temperature": 0.7
        }
    """
    prompt: str = Field(
        ...,
        description="The input prompt for both models",
        min_length=1,
        max_length=5000
    )

    max_tokens: int = Field(
        default=500,
        ge=50,
        le=2000
    )

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0
    )


class ComparisonResult(BaseModel):
    """
    Result from a single model in a comparison.
    """
    model_name: str
    model_type: ModelType
    text: str
    tokens_generated: Optional[int] = None
    generation_time: Optional[float] = None
    success: bool


class CompareResponse(BaseModel):
    """
    Response with side-by-side comparison of models.

    Example:
        {
            "prompt": "Write about AI",
            "base_model": {...},
            "finetuned_model": {...}
        }
    """
    prompt: str = Field(
        ...,
        description="The prompt that was used"
    )

    base_model: ComparisonResult = Field(
        ...,
        description="Results from the base model"
    )

    finetuned_model: ComparisonResult = Field(
        ...,
        description="Results from the fine-tuned model"
    )


class HealthResponse(BaseModel):
    """
    Health check response.

    Example:
        {
            "status": "healthy",
            "ollama_connected": true,
            "models_available": 2
        }
    """
    status: str = Field(
        ...,
        description="Overall API health status"
    )

    ollama_connected: bool = Field(
        ...,
        description="Whether Ollama server is reachable"
    )

    models_available: int = Field(
        ...,
        description="Number of models currently available"
    )
