"""Načíta a kontroluje nastavenia CLI, providerov a prahov verifikácie."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _bool_env(name: str, default: bool) -> bool:
    """Prijme názov premennej a predvolenú hodnotu; vráti jej booleovské nastavenie."""
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    """Nemenné nastavenia úložísk, modelov, prahov a bezpečnostných volieb aplikácie."""

    files_dir: Path
    data_dir: Path
    qdrant_path: Path
    qdrant_collection: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    max_retrieval_attempts: int
    min_retrieval_score: float
    min_answer_confidence: float
    embedding_provider: str
    openai_embedding_model: str
    local_embedding_model: str
    deepseek_model: str
    deepseek_base_url: str
    typesafe_model: str
    min_jev_confidence: float
    openai_web_model: str
    web_search_enabled: bool
    ocr_mode: str
    tenant_id: str
    user_id: str

    @classmethod
    def load(cls, env_file: str | Path = ".env") -> Settings:
        """Načíta .env a prostredie, validuje ich a vráti pripravené nastavenia."""
        load_dotenv(env_file)
        settings = cls(
            files_dir=Path(os.getenv("FILES_DIR", "files")),
            data_dir=Path(os.getenv("DATA_DIR", "data")),
            qdrant_path=Path(os.getenv("QDRANT_PATH", "data/qdrant")),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "document_chunks"),
            top_k=int(os.getenv("TOP_K", "5")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1400")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "180")),
            max_retrieval_attempts=int(os.getenv("MAX_RETRIEVAL_ATTEMPTS", "2")),
            min_retrieval_score=float(os.getenv("MIN_RETRIEVAL_SCORE", "0.25")),
            min_answer_confidence=float(os.getenv("MIN_ANSWER_CONFIDENCE", "0.72")),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "openai").lower(),
            openai_embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"
            ),
            local_embedding_model=os.getenv(
                "LOCAL_EMBEDDING_MODEL", "intfloat/multilingual-e5-large"
            ),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            typesafe_model=os.getenv("TYPESAFE_MODEL", "jev-1.13.0"),
            min_jev_confidence=float(os.getenv("MIN_JEV_CONFIDENCE", "0.80")),
            openai_web_model=os.getenv("OPENAI_WEB_MODEL", "gpt-5.6-terra"),
            web_search_enabled=_bool_env("WEB_SEARCH_ENABLED", True),
            ocr_mode=os.getenv("OCR_MODE", "reject").lower(),
            tenant_id=os.getenv("TENANT_ID", "local"),
            user_id=os.getenv("USER_ID", "cli-user"),
        )
        settings.validate()
        settings.files_dir.mkdir(parents=True, exist_ok=True)
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.qdrant_path.parent.mkdir(parents=True, exist_ok=True)
        return settings

    def validate(self) -> None:
        """Skontroluje rozsahy a povolené voľby; pri chybe vyvolá ValueError."""
        if self.top_k != 5:
            raise ValueError("TOP_K musí byť podľa návrhu nastavené na 5.")
        if not 300 <= self.chunk_size <= 4000:
            raise ValueError("CHUNK_SIZE musí byť v rozsahu 300 až 4000 znakov.")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP musí byť nezáporný a menší než CHUNK_SIZE.")
        if self.embedding_provider not in {"openai", "local"}:
            raise ValueError("EMBEDDING_PROVIDER musí byť 'openai' alebo 'local'.")
        if self.ocr_mode not in {"reject", "hosted"}:
            raise ValueError("OCR_MODE musí byť 'reject' alebo 'hosted'.")
        if not 0 < self.min_jev_confidence <= 1:
            raise ValueError("MIN_JEV_CONFIDENCE musí byť v intervale (0, 1].")
