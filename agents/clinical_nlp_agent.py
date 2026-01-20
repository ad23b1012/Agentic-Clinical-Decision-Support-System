"""
Clinical NLP Agent (Enhanced)
-----------------------------
Responsibilities:
- Extract clinical entities from text
- Detect negation (no / denies / without)
- Extract labs with value + unit
- Preserve context, date, and source
- NO diagnosis or reasoning

Designed for downstream:
- Temporal agent
- GraphRAG
- Explainability
"""

from typing import List, Dict
import re

# -----------------------------
# Public API
# -----------------------------

def extract_entities(documents: List[Dict]) -> List[Dict]:
    extracted: List[Dict] = []

    for doc in documents:
        text = doc["text"]

        for extractor in [
            _extract_symptoms,
            _extract_conditions,
            _extract_labs,
            _extract_medications,
            _extract_procedures,
        ]:
            entities = extractor(text)

            for ent in entities:
                extracted.append({
                    "entity": ent["entity"],
                    "type": ent["type"],
                    "normalized": _normalize(ent["entity"]),
                    "negated": ent.get("negated", False),
                    "value": ent.get("value"),
                    "unit": ent.get("unit"),
                    "context": ent["context"],
                    "date": doc.get("date"),
                    "source": doc.get("source"),
                })

    return extracted


# -----------------------------
# Negation detection (simple, reliable)
# -----------------------------

NEGATION_CUES = [
    "no", "not", "denies", "denied", "without",
    "absence of", "negative for", "free of"
]

def _is_negated(text: str, start: int) -> bool:
    """
    Checks a small window before entity mention
    """
    window = text[max(0, start - 40):start].lower()
    return any(cue in window for cue in NEGATION_CUES)


# -----------------------------
# Entity extractors
# -----------------------------

def _extract_symptoms(text: str) -> List[Dict]:
    symptoms = [
        "chest pain", "shortness of breath", "fever", "fatigue",
        "weight loss", "cough", "headache", "palpitations",
        "nausea", "vomiting", "dizziness"
    ]
    return _keyword_entity_extractor(text, symptoms, "symptom")


def _extract_conditions(text: str) -> List[Dict]:
    conditions = [
        "diabetes", "hypertension", "tuberculosis", "cancer",
        "pneumonia", "asthma", "anemia", "myocardial infarction"
    ]
    return _keyword_entity_extractor(text, conditions, "condition")


def _extract_medications(text: str) -> List[Dict]:
    medications = [
        "paracetamol", "aspirin", "metformin",
        "insulin", "amoxicillin", "atorvastatin"
    ]
    return _keyword_entity_extractor(text, medications, "medication")


def _extract_procedures(text: str) -> List[Dict]:
    procedures = [
        "ct scan", "x-ray", "ecg", "echocardiogram",
        "angiography", "biopsy"
    ]
    return _keyword_entity_extractor(text, procedures, "procedure")


# -----------------------------
# Improved Lab Extraction
# -----------------------------

def _extract_labs(text: str) -> List[Dict]:
    """
    Extracts lab name, numeric value, and unit.
    Example:
        Hemoglobin: 10.2 g/dL
        WBC = 12000 /mm3
    """

    lab_patterns = {
        "hemoglobin": r"(hemoglobin|hb)\s*[:=]?\s*(\d+\.?\d*)\s*(g/dl|gm/dl)?",
        "wbc": r"(wbc|white blood cell)\s*[:=]?\s*(\d+)\s*(/mm3|x10\^3)?",
        "esr": r"(esr)\s*[:=]?\s*(\d+)\s*(mm/hr)?",
        "platelets": r"(platelet[s]?)\s*[:=]?\s*(\d+)\s*(/mm3|x10\^3)?",
    }

    results = []

    for lab, pattern in lab_patterns.items():
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            start = match.start()
            end = match.end()

            results.append({
                "entity": lab,
                "type": "lab",
                "value": match.group(2),
                "unit": match.group(3),
                "negated": _is_negated(text, start),
                "context": text[max(0, start - 40):min(len(text), end + 40)],
            })

    return results


# -----------------------------
# Shared keyword extractor
# -----------------------------

def _keyword_entity_extractor(text: str, keywords: List[str], entity_type: str) -> List[Dict]:
    lowered = text.lower()
    results = []

    for kw in keywords:
        for match in re.finditer(rf"\b{re.escape(kw)}\b", lowered):
            start, end = match.start(), match.end()
            results.append({
                "entity": kw,
                "type": entity_type,
                "negated": _is_negated(text, start),
                "context": text[max(0, start - 40):min(len(text), end + 40)],
            })

    return results


# -----------------------------
# Normalization (placeholder)
# -----------------------------

def _normalize(entity: str) -> str:
    return entity.upper().replace(" ", "_")


# -----------------------------
# MAIN (Manual Test)
# -----------------------------

def main():
    sample_docs = [
        {
            "text": """
            Patient presents with chest pain and shortness of breath.
            Denies fever and no cough.
            History of diabetes and hypertension.
            Hemoglobin: 10.2 g/dL
            WBC = 12000 /mm3
            ESR: 45 mm/hr
            CT scan of chest performed.
            Treated with aspirin and metformin.
            """,
            "doc_type": "clinical_note",
            "date": "2024-08-14",
            "source": "sample_note.txt",
        }
    ]

    entities = extract_entities(sample_docs)

    print("\nExtracted Entities:\n")
    for e in entities:
        print(e)


if __name__ == "__main__":
    main()
