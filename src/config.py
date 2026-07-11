
"""
src/config.py
─────────────
Centralised configuration loaded from environment variables / .env file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict




PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent




class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    
    
    

    GROQ_API_KEY: str = Field(
        default="",
        description="Groq API Key.",
    )

    OPENAI_BASE_URL: str = Field(
        default="https://api.groq.com/openai/v1",
        description="Groq OpenAI-compatible endpoint.",
    )

    LLM_MODEL: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model.",
    )

    LLM_TEMPERATURE: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
    )

    LLM_MAX_TOKENS: int = Field(
        default=1024,
        gt=0,
    )

    
    
    

    EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
    )

    
    
    

    CHROMA_DB_PATH: Path = Field(
        default=PROJECT_ROOT / "chroma_db",
    )

    CHROMA_COLLECTION_NAME: str = Field(
        default="research_assistant",
    )

    
    
    

    CHUNK_SIZE: int = Field(default=1000)
    CHUNK_OVERLAP: int = Field(default=200)

    
    
    

    RETRIEVER_K: int = Field(default=5)

    
    
    

    MAX_HISTORY_TURNS: int = Field(default=6)

    
    
    

    DATA_PATH: Path = Field(default=PROJECT_ROOT / "data")
    LOG_PATH: Path = Field(default=PROJECT_ROOT / "logs")

    LOG_LEVEL: str = Field(default="INFO")

    
    
    

    @field_validator("CHROMA_DB_PATH", "DATA_PATH", "LOG_PATH", mode="after")
    @classmethod
    def _ensure_directory(cls, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path

    @field_validator("LLM_TEMPERATURE", mode="before")
    @classmethod
    def _parse_temperature(cls, value):
        return float(value)

    
    
    

    @property
    def has_llm_key(self) -> bool:
        return bool(self.GROQ_API_KEY.strip())

    @property
    def log_file(self) -> Optional[Path]:
        return self.LOG_PATH / "rag_assistant.log"




settings = Settings()