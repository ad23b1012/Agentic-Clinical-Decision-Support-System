"""
Embedding Agent (Full-Text + Entity Support)
--------------------------------------------
Responsibility:
- Convert Clinical Entities -> Embeddings
- Convert Residual Text / Conclusions -> Embeddings
"""

import os
import json
from typing import Dict, List, Union
from dotenv import load_dotenv

load_dotenv()
from google import genai
from google.genai import types

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
EMBEDDING_MODEL = "text-embedding-004"
config = types.EmbedContentConfig(
    output_dimensionality=768,
    task_type="RETRIEVAL_QUERY"  # Or "RETRIEVAL_DOCUMENT" depending on your use case
)

def embed_clinical_json(clinical_output: Union[Dict, str]) -> Dict:
    data = _load_json(clinical_output)
    entities = data.get("entities", [])
    metadata = data.get("doc_metadata", {})
    
    records = []

    # -------------------------------------------------
    # 1. Embed Structured Entities
    # -------------------------------------------------
    for ent in entities:
        text = _entity_to_text(ent)
        if not text.strip(): continue

        vector = _embed_text(text)
        
        records.append({
            "entity": ent.get("entity"),
            "type": ent.get("type"),
            "normalized": ent.get("normalized"),
            "value": ent.get("value"),
            "unit": ent.get("unit"),
            "context": ent.get("context"),
            "embedding": vector
        })

    # -------------------------------------------------
    # 2. Embed Unstructured Text (Residuals & Conclusion)
    # -------------------------------------------------
    
    # Process Residual Text (Doctor's Notes)
    residual = data.get("residual_text", "").strip()
    if residual and len(residual) > 10:
        print("[DEBUG] Embedding residual text...")
        records.append({
            "entity": "Clinical Note",       # Generic Name
            "normalized": "CLINICAL_NOTE",   # ID-safe Name
            "type": "clinical_note",
            "context": residual,             # The full text
            "value": None,
            "unit": None,
            "embedding": _embed_text(residual)
        })

    # Process Conclusion (AI Summary)
    conclusion = data.get("conclusion_text", "").strip()
    if conclusion and len(conclusion) > 10:
        print("[DEBUG] Embedding conclusion text...")
        records.append({
            "entity": "AI Summary",
            "normalized": "AI_SUMMARY",
            "type": "summary",
            "context": conclusion,
            "value": None,
            "unit": None,
            "embedding": _embed_text(conclusion)
        })

    return {
        "doc_metadata": metadata,
        "embedding_model": EMBEDDING_MODEL,
        "embeddings": records
    }


def _entity_to_text(ent: Dict) -> str:
    raw_context = ent.get("context", "").strip()
    tags = [f"Entity: {ent.get('entity')}", f"Type: {ent.get('type')}"]
    
    if ent.get("section"):
        tags.append(f"Section: {ent.get('section')}")

    if raw_context:
        return f"{raw_context} | {' | '.join(tags)}"
    return " | ".join(tags)


def _load_json(inp: Union[Dict, str]) -> Dict:
    if isinstance(inp, dict): return inp
    if isinstance(inp, str):
        with open(inp, "r", encoding="utf-8") as f: return json.load(f)
    raise TypeError("Input must be dict or JSON file path")


def _embed_text(text: str) -> List[float]:
    # Truncate text to avoid token limits (approx 2000 chars safe for basic use)
    safe_text = text[:8000] 
    response = client.models.embed_content(model=EMBEDDING_MODEL, contents=safe_text, config=config)
    return response.embeddings[0].values


# =====================================================
# STANDALONE TEST (WITH EMBEDDING DISPLAY)
# =====================================================

def _standalone_test():
    sample_clinical_output = {
        "doc_metadata": {
            "source": "sample_note.txt",
            "date": "2024-08-14",
            "doc_type": "lab_report"
        },
        "entities": [
            {
                "entity": "SGOT",
                "type": "lab",
                "normalized": "SGOT",
                "value": "162",
                "unit": "U/L",
                "section": "laboratory_results"
            },
            {
                "entity": "ALBUMIN",
                "type": "lab",
                "normalized": "ALBUMIN",
                "value": "3.7",
                "unit": "g/dL",
                "section": "laboratory_results"
            }
        ]
    }

    output = embed_clinical_json(sample_clinical_output)

    print("\n[INFO] Embedding test successful\n")

    for idx, emb in enumerate(output["embeddings"], start=1):
        vector = emb["embedding"]

        print("=" * 70)
        print(f"Entity {idx}: {emb['entity']}")
        print(f"Type    : {emb['type']}")
        print(f"Dim     : {len(vector)}")
        print(f"Preview : {vector[:10]}")  # first 10 values only
        print("=" * 70)

    print("\n[INFO] Total embeddings:", len(output["embeddings"]))


if __name__ == "__main__":
    _standalone_test()
