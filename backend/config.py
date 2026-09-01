"""
Configuration management via pydantic-settings.
Loads from .env file and validates required fields.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from functools import lru_cache
import warnings

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator


class LLMProvider(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM ---
    llm_provider: LLMProvider = LLMProvider.GEMINI
    gemini_api_key: str = ""
    openai_api_key: str = ""

    # --- Neo4j AuraDB / Cloud / Local ---
    neo4j_uri: str = "neo4j+s://your-instance.databases.neo4j.io"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # --- Qdrant (Embedded Local Storage or Cloud) ---
    # Default to local disk embedded store (Zero Docker required!)
    qdrant_path: Path = Path("data/qdrant_db")
    qdrant_url: str = ""  # Optional: For Qdrant Cloud cluster
    qdrant_api_key: str = ""  # Optional: For Qdrant Cloud

    # --- Paths ---
    data_dir: Path = Path("data/raw")
    db_path: Path = Path("data/graphrag.db")

    @model_validator(mode="after")
    def _check_api_key(self) -> "Settings":
        if self.llm_provider == LLMProvider.GEMINI and not self.gemini_api_key:
            warnings.warn(
                "GEMINI_API_KEY is not set. LLM features will fail until configured.",
                stacklevel=2,
            )
        if self.llm_provider == LLMProvider.OPENAI and not self.openai_api_key:
            warnings.warn(
                "OPENAI_API_KEY is not set. LLM features will fail until configured.",
                stacklevel=2,
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
