# src/schema.py
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, HttpUrl, Field
from datetime import datetime


class SourceMetadata(BaseModel):
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    language: Optional[str] = None
    url: Optional[HttpUrl] = None
    license: Optional[str] = None
    source_type: Optional[str] = None  # gutenberg / html / json_api / txt / unknown
    raw_meta: Optional[Dict[str, Any]] = None


class InputRecord(BaseModel):
    id: str = Field(..., description="Unique id for this document (repo-level)")
    metadata: SourceMetadata
    paragraphs: List[str] = Field(..., description="Paragraph-preserving text segments")
    paragraph_orig_indices: List[int] = Field(..., description="Original paragraph indices")
    ingest_time: datetime = Field(default_factory=datetime.utcnow)
    provenance: Dict[str, Any] = Field(default_factory=dict)
