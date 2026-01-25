from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ClinicalState:
    """
    Shared mutable state for a single pipeline execution.
    """

    # ----------------------------
    # Input
    # ----------------------------
    file_paths: List[str]

    # ----------------------------
    # Ingestion output
    # ----------------------------
    raw_documents: List[Dict] = field(default_factory=list)

    # ----------------------------
    # NLP output (raw)
    # ----------------------------
    nlp_results: List[Dict] = field(default_factory=list)

    # ----------------------------
    # NLP output (date-normalized)
    # ----------------------------
    normalized_nlp_results: List[Dict] = field(default_factory=list)

    # ----------------------------
    # Canonical normalized date
    # ----------------------------
    normalized_date: Optional[str] = None

    # ----------------------------
    # Embedding output
    # ----------------------------
    embedding_results: List[Dict] = field(default_factory=list)

    # ----------------------------
    # Vector store output
    # ----------------------------
    vector_store_results: List[Dict] = field(default_factory=list)

    # ----------------------------
    # FINAL LLM reasoning output (answer.py)
    # ----------------------------
    reasoning_result: Optional[Dict] = None

    # ----------------------------
    # Bookkeeping
    # ----------------------------
    errors: List[str] = field(default_factory=list)
    current_step: Optional[str] = None

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
