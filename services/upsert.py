"""
Pinecone Vector Store Agent
---------------------------
Responsibility:
- Store embeddings in Pinecone
- NO reasoning
- NO interpretation
- Deterministic upsert only
"""

from dotenv import load_dotenv
load_dotenv()
import os
from typing import Dict, List
from pinecone import Pinecone

# =====================================================
# CONFIG (loaded from .env)
# =====================================================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_URL = os.getenv("PINECONE_INDEX_URL")

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not found in environment")

if not PINECONE_INDEX_URL:
    raise RuntimeError("PINECONE_INDEX_URL not found in environment")


# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# NOTE: host= is REQUIRED when using serverless indexes
index = pc.Index(host=PINECONE_INDEX_URL)


# =====================================================
# PUBLIC API
# =====================================================

# ... (Imports and Config) ...

def upsert_embeddings(embedding_output: Dict) -> Dict:
    metadata = embedding_output.get("doc_metadata", {})
    embeddings = embedding_output.get("embeddings", [])
    vectors: List[Dict] = []

    for idx, item in enumerate(embeddings):
        vector_id = _build_vector_id(metadata, item, idx)

        meta = {
            "entity": item.get("entity"),
            "normalized": item.get("normalized"),
            "type": item.get("type"),
            "value": item.get("value"),
            "unit": item.get("unit"),
            "context": item.get("context"), # <--- This will hold the full text notes
            "source": metadata.get("source"),
            "date": metadata.get("date"),
            "doc_type": metadata.get("doc_type"),
        }
        
        # Remove None values
        meta = {k: v for k, v in meta.items() if v is not None}

        vectors.append({
            "id": vector_id,
            "values": item["embedding"],
            "metadata": meta
        })

    if vectors:
        index.upsert(vectors=vectors)

    return {"upserted_vectors": len(vectors), "index": PINECONE_INDEX_URL}

def _build_vector_id(metadata: Dict, item: Dict, idx: int) -> str:
    source = metadata.get("source", "doc")
    # Fallback to "CHUNK" if normalized is missing
    entity = item.get("normalized", "CHUNK") 
    safe_source = source.replace(" ", "_")
    return f"{safe_source}:{entity}:{idx}"

# =====================================================
# STANDALONE TEST
# =====================================================

def _standalone_test():
    """
    Run this file directly to test Pinecone upsert.
    """

    sample_embedding_output = {
        "doc_metadata": {
            "source": "sample_note.txt",
            "date": "2024-08-14",
            "doc_type": "lab_report"
        },
        "embedding_model": "text-embedding-004",
        "embeddings": [
            {
                "entity": "SGOT",
                "type": "lab",
                "normalized": "SGOT",
                "embedding": [0.01] * 768
            },
            {
                "entity": "ALBUMIN",
                "type": "lab",
                "normalized": "ALBUMIN",
                "embedding": [0.02] * 768
            }
        ]
    }

    result = upsert_embeddings(sample_embedding_output)

    print("\n[INFO] Pinecone upsert result:")
    print(result)


if __name__ == "__main__":
    _standalone_test()
