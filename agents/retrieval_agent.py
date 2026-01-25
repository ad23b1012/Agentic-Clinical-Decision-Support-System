"""
Retriever Agent (Entity-centric, Pinecone)
------------------------------------------
Responsibility:
- Retrieve relevant clinical ENTITIES from Pinecone
- Support:
    1. Summary retrieval (dashboard)
    2. Q/A retrieval (chat)

Rules:
- NO LLM usage
- Deterministic retrieval only
"""

from typing import List, Dict, Optional
from pinecone import Pinecone
import os
from services.embedding import _embed_text

# =====================================================
# PINECONE CONFIG
# =====================================================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_URL = os.getenv("PINECONE_INDEX_URL")

if not PINECONE_API_KEY or not PINECONE_INDEX_URL:
    raise RuntimeError("Pinecone environment variables missing")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_INDEX_URL)


# =====================================================
# RETRIEVAL CONFIG
# =====================================================

QA_TOP_K = 5
SUMMARY_TOP_K = 20


# =====================================================
# RETRIEVER AGENT
# =====================================================

class RetrieverAgent:

    # -------------------------------------------------
    # Mode 1: Q/A Retrieval (High Precision)
    # -------------------------------------------------

    def retrieve_for_qa(
        self,
        query: str,
        filters: Optional[Dict] = None,
        top_k: int = QA_TOP_K
    ) -> List[Dict]:
        
        if not query or not query.strip():
            return []

        # Embed the query
        query_vector = _embed_text(query)
        pinecone_filter = self._build_filter(filters)

        # Query Pinecone
        response = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=pinecone_filter
        )

        return self._postprocess(response.matches)


    # -------------------------------------------------
    # Mode 2: Summary Retrieval (High Recall)
    # -------------------------------------------------

    def retrieve_for_summary(
        self,
        filters: Optional[Dict] = None,
        top_k: int = SUMMARY_TOP_K
    ) -> List[Dict]:

        anchor_query = "clinical summary diagnosis labs medications findings history"
        query_vector = _embed_text(anchor_query)
        pinecone_filter = self._build_filter(filters)

        response = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=pinecone_filter
        )

        return self._postprocess(response.matches)


# =====================================================
# INTERNAL HELPERS
# =====================================================

    def _build_filter(self, filters: Optional[Dict]) -> Optional[Dict]:
        if not filters:
            return None
        return {
            k: {"$eq": v}
            for k, v in filters.items()
            if v is not None
        }


    def _postprocess(self, matches) -> List[Dict]:
        """
        Normalize Pinecone matches.
        UPDATED: Now extracts 'value', 'unit', and 'context'
        """
        results = []

        for m in matches:
            meta = m.metadata or {}

            results.append({
                "id": m.id,
                "score": m.score,
                # Core Fields
                "entity": meta.get("entity"),
                "normalized": meta.get("normalized"),
                "type": meta.get("type"),
                
                # NEW FIELDS (Critical for your Answer)
                "value": meta.get("value"),
                "unit": meta.get("unit"),
                "context": meta.get("context"),
                
                # Metadata
                "source": meta.get("source"),
                "date": meta.get("date"),
                "doc_type": meta.get("doc_type"),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results


# =====================================================
# MANUAL TEST
# =====================================================

if __name__ == "__main__":
    retriever = RetrieverAgent()

    print("\n[TEST] Q/A Retrieval\n")
    print("Query: 'The clinical impression is that the patient's unconjugated bilirubin level is elevated at what level?'")

    qa_results = retriever.retrieve_for_qa(
        query="The clinical impression is that the patient's unconjugated bilirubin level is elevated at what level?",
        # filters={"doc_type": "lab_report"}
    )

    for r in qa_results:
        print("=" * 70)
        print(f"Entity : {r['entity']}")
        print(f"Value  : {r['value']} {r['unit']}")  # <--- Now we print the value!
        print(f"Context: {r['context']}")
        print(f"Score  : {r['score']:.4f}")

    print("\n[TEST] Summary Retrieval\n")
    # ... (summary test code)
    print("\n[TEST] Summary Retrieval\n")

    summary_results = retriever.retrieve_for_summary(
        # filters={"doc_type": "lab_report"}
    )

    for r in summary_results:
        print("=" * 70)
        print("Entity :", r["entity"])
        print("Type   :", r["type"])
        print("Score  :", r["score"])
