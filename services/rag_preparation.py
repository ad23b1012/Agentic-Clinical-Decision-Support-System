"""
RAG Preparation Agent
--------------------
Responsibilities:
- Convert clinical NLP outputs into RAG-ready chunks
- Preserve clinical meaning, section, time, and traceability
- NO embeddings
- NO retrieval
- NO reasoning

Designed for:
- Hybrid RAG
- Explainable answers
- Future GraphRAG upgrade
"""

from typing import List, Dict
from collections import defaultdict
import uuid

MAX_CHARS = 2000


# -----------------------------
# Public API
# -----------------------------

def prepare_rag_chunks(
    entities: List[Dict],
    doc_type: str = "clinical_note"
) -> List[Dict]:
    """ 
    Converts extracted clinical entities into RAG-ready chunks.
    """

    # Step 1: Group entities by (source, date, section)
    grouped = _group_entities(entities)

    # Step 2: Build chunks per group
    chunks: List[Dict] = []

    for (source, date, section), ents in grouped.items():
        chunk = _build_chunk(
            source=source,
            date=date,
            section=section,
            entities=ents,
            doc_type=doc_type
        )

        if chunk:
            chunks.append(chunk)

    return chunks


# -----------------------------
# Grouping logic
# -----------------------------

def _group_entities(entities: List[Dict]) -> Dict:
    """
    Groups entities by document, date, and section.
    """
    grouped = defaultdict(list)

    for ent in entities:
        key = (
            ent.get("source"),
            ent.get("date"),
            ent.get("section", "unspecified")
        )
        grouped[key].append(ent)

    return grouped


# -----------------------------
# Chunk builder
# -----------------------------

def _build_chunk(
    source: str,
    date: str,
    section: str,
    entities: List[Dict],
    doc_type: str
) -> Dict | None:
    """
    Builds a single RAG chunk from grouped entities.
    """

    if not entities:
        return None

    # Collect unique context snippets
    contexts = []
    seen = set()
    for e in entities:
        ctx = e.get("context", "").strip()
        norm = " ".join(ctx.lower().split()) #This avoids duplicate embeddings for trivial whitespace differences.
        if norm not in seen:
            contexts.append(ctx)
            seen.add(norm)

    if not contexts:
        return None

    # Assemble chunk text (light cleanup only)
    contexts = sorted(contexts,key=lambda c: entities[0]["context"].find(c)) #This ensures: symptoms → labs → investigations → treatmen
    chunk_text = "\n".join(contexts)

    if len(chunk_text) > MAX_CHARS:
        chunk_text = chunk_text[:MAX_CHARS]


    # Collect entity metadata
    entity_names = []
    entity_types = []
    negated_entities = []
    labs = []

    for e in entities:
        entity_names.append(e["normalized"])
        entity_types.append(e["type"])

        if e.get("negated"):
            negated_entities.append(e["normalized"])

        if e["type"] == "lab":
            labs.append({
                "lab": e["normalized"],
                "value": e.get("value"),
                "unit": e.get("unit"),
                "context": e.get("context")
            })

    # Build final chunk
    return {
        "chunk_id": _generate_chunk_id(source, date, section),
        "text": chunk_text,
        "entities": sorted(set(entity_names)),
        "entity_types": sorted(set(entity_types)),
        "section": section,
        "date": date,
        "source": source,
        "metadata": {
            "doc_type": doc_type,
            "negated_entities": sorted(set(negated_entities)),
            "labs": labs
        }
    }


# -----------------------------
# Utilities
# -----------------------------

def _generate_chunk_id(source: str, date: str, section: str) -> str:
    """
    Generates a stable but unique chunk ID.
    """
    base = f"{source}_{date}_{section}"
    return f"{base}_{uuid.uuid4().hex[:8]}"


# -----------------------------
# MAIN (Manual Test)
# -----------------------------

def main():
    # Simulated output from clinical_nlp_agent
    extracted_entities = [

    # -------------------------
    # Document 1 – ER Visit
    # -------------------------

    {
        "entity": "chest pain",
        "normalized": "CHEST_PAIN",
        "type": "symptom",
        "negated": False,
        "context": (
            "Patient presents with acute onset chest pain radiating to the left arm, "
            "associated with diaphoresis and nausea."
        ),
        "section": "review_of_systems",
        "date": "2024-08-14",
        "source": "er_note_1.txt",
    },
    {
        "entity": "shortness of breath",
        "normalized": "SHORTNESS_OF_BREATH",
        "type": "symptom",
        "negated": False,
        "context": (
            "Patient presents with acute onset chest pain radiating to the left arm, "
            "associated with diaphoresis and nausea."
        ),
        "section": "review_of_systems",
        "date": "2024-08-14",
        "source": "er_note_1.txt",
    },
    {
        "entity": "fever",
        "normalized": "FEVER",
        "type": "symptom",
        "negated": True,
        "context": (
            "Denies fever, cough, or recent infection."
        ),
        "section": "review_of_systems",
        "date": "2024-08-14",
        "source": "er_note_1.txt",
    },

    # -------------------------
    # Past Medical History
    # -------------------------

    {
        "entity": "diabetes mellitus",
        "normalized": "DIABETES_MELLITUS",
        "type": "condition",
        "negated": False,
        "context": (
            "Past medical history is significant for type 2 diabetes mellitus "
            "and long-standing hypertension."
        ),
        "section": "past_medical_history",
        "date": "2024-08-14",
        "source": "er_note_1.txt",
    },
    {
        "entity": "hypertension",
        "normalized": "HYPERTENSION",
        "type": "condition",
        "negated": False,
        "context": (
            "Past medical history is significant for type 2 diabetes mellitus "
            "and long-standing hypertension."
        ),
        "section": "past_medical_history",
        "date": "2024-08-14",
        "source": "er_note_1.txt",
    },

    # -------------------------
    # Labs (with interpretation)
    # -------------------------

    {
        "entity": "hemoglobin",
        "normalized": "HEMOGLOBIN",
        "type": "lab",
        "negated": False,
        "value": "9.8",
        "unit": "g/dL",
        "context": (
            "Hemoglobin is 9.8 g/dL, which is below the normal reference range, "
            "suggestive of anemia."
        ),
        "section": "investigation",
        "date": "2024-08-14",
        "source": "er_note_1.txt",
    },
    {
        "entity": "troponin",
        "normalized": "TROPONIN",
        "type": "lab",
        "negated": False,
        "value": "1.2",
        "unit": "ng/mL",
        "context": (
            "Cardiac troponin is elevated at 1.2 ng/mL, raising concern for myocardial injury."
        ),
        "section": "investigation",
        "date": "2024-08-14",
        "source": "er_note_1.txt",
    },

    # -------------------------
    # Radiology Report (Narrative)
    # -------------------------

    {
        "entity": "ct angiography",
        "normalized": "CT_ANGIOGRAPHY",
        "type": "procedure",
        "negated": False,
        "context": (
            "CT angiography of the chest demonstrates no evidence of aortic dissection "
            "or pulmonary embolism. Mild coronary artery calcifications are noted."
        ),
        "section": "radiology",
        "date": "2024-08-14",
        "source": "ct_report_1.txt",
    },

    # -------------------------
    # Treatment
    # -------------------------

    {
        "entity": "aspirin",
        "normalized": "ASPIRIN",
        "type": "medication",
        "negated": False,
        "context": (
            "The patient was treated with aspirin and started on high-intensity statin therapy."
        ),
        "section": "treatment",
        "date": "2024-08-14",
        "source": "er_note_1.txt",
    },
    {
        "entity": "atorvastatin",
        "normalized": "ATORVASTATIN",
        "type": "medication",
        "negated": False,
        "context": (
            "The patient was treated with aspirin and started on high-intensity statin therapy."
        ),
        "section": "treatment",
        "date": "2024-08-14",
        "source": "er_note_1.txt",
    },

    # -------------------------
    # Document 2 – Follow-up Visit
    # -------------------------

    {
        "entity": "chest pain",
        "normalized": "CHEST_PAIN",
        "type": "symptom",
        "negated": True,
        "context": (
            "On follow-up visit, the patient denies chest pain or shortness of breath."
        ),
        "section": "review_of_systems",
        "date": "2024-08-21",
        "source": "followup_note.txt",
    },
    {
        "entity": "shortness of breath",
        "normalized": "SHORTNESS_OF_BREATH",
        "type": "symptom",
        "negated": True,
        "context": (
            "On follow-up visit, the patient denies chest pain or shortness of breath."
        ),
        "section": "review_of_systems",
        "date": "2024-08-21",
        "source": "followup_note.txt",
    },
]


    chunks = prepare_rag_chunks(extracted_entities)

    print("\nRAG Chunks:\n")
    for c in chunks:
        print("=" * 80)
        for k, v in c.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
