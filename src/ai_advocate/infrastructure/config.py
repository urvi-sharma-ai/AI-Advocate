from __future__ import annotations

from datetime import date
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    chroma_persist_dir: Path = Field(default=Path(".chroma"), alias="CHROMA_PERSIST_DIR")
    openai_chat_model: str = Field(default="gpt-4o-mini", alias="OPENAI_CHAT_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        alias="OPENAI_EMBEDDING_MODEL",
    )
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    manifest_path: Path | None = Field(default=None, alias="MANIFEST_PATH")
    chroma_collection: str = Field(default="legal_statutes", alias="CHROMA_COLLECTION")

    embedding_batch_size: int = Field(
        default=100,
        ge=1,
        le=2048,
        alias="EMBEDDING_BATCH_SIZE",
        description="Chunks per embeddings request; avoids max_tokens_per_request.",
    )

    # Reference “today” for temporal disambiguation in prompts (ISO date).
    reference_date: date = Field(
        default_factory=date.today,
        alias="REFERENCE_DATE",
    )

    @field_validator("manifest_path", mode="before")
    @classmethod
    def _empty_manifest_path_is_none(cls, value: object) -> object:
        # Empty MANIFEST_PATH= in .env becomes "" → Path which normalizes to "." (cwd), not None.
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("manifest_path", mode="after")
    @classmethod
    def _dot_manifest_path_is_none(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        if value == Path("."):
            return None
        return value

    def resolved_manifest(self) -> Path:
        default = self.data_dir / "manifest.yaml"
        if self.manifest_path is None:
            return default
        return self.manifest_path.expanduser()

    def require_api_key(self) -> str:
        if not self.openai_api_key.strip():
            from ai_advocate.domain.exceptions import ConfigurationError

            raise ConfigurationError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return self.openai_api_key
