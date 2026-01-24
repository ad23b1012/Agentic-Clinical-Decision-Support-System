"""
Clinical + Lab NLP Pipeline (OCR-Resilient)
-------------------------------------------
Stage 0: Document type detection
Stage 1: Robust extraction (Pivot-around-Number)
Stage 2: LLM reasoning → KEEP / DROP entities
Stage 3: LLM normalization
Stage 4: Residual text reasoning

STRICT: No diagnosis, No interpretation.
"""

import os
import re
import json
from datetime import datetime
from typing import List, Dict, Tuple
from dotenv import load_dotenv

load_dotenv()

# =====================================================
# GROQ CLIENT (Updated Model)
# =====================================================
from groq import Groq
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
# Using Llama 3.3 70B for superior JSON following and OCR correction
MODEL_NAME = "llama-3.3-70b-versatile" 

# =====================================================
# SPACY (Optional)
# =====================================================
try:
    import spacy
    _NLP = spacy.load("en_ner_bc5cdr_md")
    MODEL_NER_AVAILABLE = True
except Exception:
    _NLP = None
    MODEL_NER_AVAILABLE = False


# =====================================================
# PUBLIC PIPELINE
# =====================================================

def extract_and_process(documents: List[Dict]) -> Dict:
    if not documents:
        return {}
        
    extracted = extract_entities(documents)
    print(f"[DEBUG] Extracted {len(extracted)} raw entities via Regex/NLP")

    # LLM KEEP / DROP
    filtered = llm_reason_and_filter_entities(extracted)
    print(f"[DEBUG] {len(filtered)} entities survived LLM validation")

    # Deterministic residual text
    residual_text = build_residual_text(
        documents[0]["text"],
        filtered
    )

    payload = {
        "doc_metadata": {
            "source": documents[0].get("source"),
            "date": documents[0].get("date"),
            "doc_type": documents[0].get("doc_type")
        },
        "entities": filtered,
        "residual_text": residual_text
    }

    normalized = llm_normalize_entities(payload)

    # Residual reasoning (safe)
    normalized["conclusion_text"] = llm_reason_over_residual(residual_text)

    return normalized


# =====================================================
# DOCUMENT TYPE DETECTION
# =====================================================

def detect_doc_type(text: str) -> str:
    # Expanded keywords for better detection
    lab_markers = [
        "bilirubin", "sgot", "sgpt", "alkaline", "albumin", "globulin", 
        "serum", "lft", "hemoglobin", "platelet", "neutrophils", "lymphocytes",
        "investigation", "observed value", "biological ref", "method", "unit"
    ]
    # Simple fuzzy check (allows for slight OCR errors)
    text_lower = text.lower()
    score = 0
    for k in lab_markers:
        if k in text_lower:
            score += 1
            
    return "lab_report" if score >= 2 else "clinical_note"


# =====================================================
# ENTITY EXTRACTION
# =====================================================

def extract_entities(documents: List[Dict]) -> List[Dict]:
    entities = []

    for doc in documents:
        text = doc["text"]
        doc_type = detect_doc_type(text)
        doc["doc_type"] = doc_type

        if doc_type == "lab_report":
            raw = extract_labs_robust(text) # New Robust Function
            source = "lab_parser_regex"
        else:
            raw = extract_clinical_entities(text)
            source = "clinical_nlp_spacy"

        for ent in raw:
            entities.append(build_entity(ent, doc, source))

    return entities


# =====================================================
# ROBUST LAB PARSER (Pivot-around-Number)
# =====================================================

JUNK_PREFIXES = [
    "iso", "i so", "regn", "mci", "hospital",
    "specimen", "facility", "note", "end of report", 
    "doctor", "patient", "company", "sponsor"
]

# We don't filter strict "KNOWN_LABS" anymore because OCR might misspell them.
# Instead, we rely on the line STRUCTURE.

UNIT_CLEANUP = {
    "mgdl": "mg/dL", "mg/dl": "mg/dL", "mgid": "mg/dL",
    "u/l": "U/L", "iul": "IU/L", "u/1": "U/L",
    "gnvdl": "g/dL", "giivdl": "g/dL", "omdt": "g/dL", "g/dl": "g/dL"
}

def extract_labs_robust(text: str) -> List[Dict]:
    results = []
    
    # Pre-clean common OCR artifacts
    lines = text.splitlines()
    
    for line in lines:
        clean_line = line.strip()
        if len(clean_line) < 5: 
            continue
            
        # 1. Skip obvious header/footer junk
        lower_line = clean_line.lower()
        if any(lower_line.startswith(prefix) for prefix in JUNK_PREFIXES):
            continue

        # 2. PIVOT STRATEGY: Find the first *value* (number)
        # Look for a number that might be a decimal (e.g., 7.79, 162, 0.5)
        # We ignore numbers at the very start of line (like list numbers "1.")
        
        # Regex: Look for a number that is NOT part of a date or ID
        # Matches: " 7.79 ", " 162 ", "0.5"
        value_match = re.search(r'\s(\d{1,4}(?:\.\d{1,2})?)\s', " " + clean_line + " ")
        
        if not value_match:
            continue
            
        value_str = value_match.group(1)
        start_idx = value_match.start() - 1 # Adjust for added space
        
        # 3. SPLIT: Left is Name, Right is Unit/Ref Range
        # We assume the name is to the left of the value
        left_part = clean_line[:start_idx].strip()
        right_part = clean_line[start_idx + len(value_str):].strip()
        
        # Filter noise from Name
        # Remove non-alpha chars from end of name (like ":", "-", ".")
        left_part = re.sub(r"[^a-zA-Z0-9\s\(\)]+$", "", left_part).strip()
        
        if len(left_part) < 3 or re.search(r'\d', left_part): 
            # If name is too short or contains numbers, it's likely noise or a date
            continue

        # Extract Unit from Right Part
        # Take the first "word" after the value as the potential unit
        unit_match = re.search(r'^([a-zA-Z/%]+)', right_part)
        unit = unit_match.group(1) if unit_match else ""
        
        # Clean unit
        unit_clean = unit.lower().replace(".", "")
        final_unit = UNIT_CLEANUP.get(unit_clean, unit)

        results.append({
            "entity": left_part,
            "type": "lab",
            "value": value_str,
            "unit": final_unit,
            "negated": False,
            "context": clean_line,
            "span": (text.find(clean_line), text.find(clean_line) + len(clean_line))
        })
        
    return results


# =====================================================
# CLINICAL NLP (Standard)
# =====================================================

def extract_clinical_entities(text: str) -> List[Dict]:
    entities = []
    # Combine heuristic + model
    if MODEL_NER_AVAILABLE:
        entities.extend(extract_model_entities(text))
    
    # Fallback/Augment with keywords
    entities.extend(extract_symptoms(text))
    
    return entities

def extract_model_entities(text: str) -> List[Dict]:
    results = []
    if not _NLP: return []
    
    doc = _NLP(text)
    for ent in doc.ents:
        if ent.label_ in ["DISEASE", "CHEMICAL"]:
            results.append({
                "entity": ent.text,
                "type": "condition" if ent.label_ == "DISEASE" else "medication",
                "negated": False, # Basic logic
                "context": extract_sentence(text, ent.start_char),
                "span": (ent.start_char, ent.end_char)
            })
    return results

def extract_symptoms(text):
    # Simple fallback
    return []

# =====================================================
# ENTITY BUILDER & UTILS
# =====================================================

def build_entity(ent: Dict, doc: Dict, source: str) -> Dict:
    return {
        "entity": ent["entity"],
        "type": ent["type"],
        "normalized": normalize(ent["entity"]),
        "value": ent.get("value"),
        "unit": ent.get("unit"),
        "negated": ent.get("negated", False),
        "context": ent["context"],
        "section": "lab_results" if doc["doc_type"] == "lab_report" else "clinical",
        "date": doc.get("date"),
        "source": doc.get("source"),
        "extraction_source": source,
        "span": ent.get("span")
    }

def extract_sentence(text, start):
    # Simple context window
    start_clamp = max(0, start - 30)
    end_clamp = min(len(text), start + 80)
    return text[start_clamp:end_clamp].replace("\n", " ")

def normalize(e):
    return re.sub(r"\s+", "_", e.strip().upper())

# =====================================================
# LLM: ENTITY VALIDATION (Using Llama 3 70B)
# =====================================================

def llm_reason_and_filter_entities(entities: List[Dict]) -> List[Dict]:
    if not entities:
        return []
        
    # We send the entities to Llama to clean up the OCR mess
    prompt = {
        "task": "Validate and Correct OCR Lab Data",
        "rules": [
            "Fix misspelled entity names (e.g. 'BILIRUB1N' -> 'BILIRUBIN')",
            "Fix common unit errors (e.g. 'omdt' -> 'g/dL')",
            "Remove entities that are clearly noise/headers",
            "KEEP valid lab results",
            "DROP purely administrative fields (Bed No, PatientID)"
        ],
        "input_entities": [
            {"id": i, "entity": e["entity"], "value": e["value"], "unit": e["unit"]}
            for i, e in enumerate(entities)
        ]
    }

    completion = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a Clinical Data Validator. Return JSON with 'validated_entities' list containing {id, decision: 'KEEP'|'DROP', corrected_name, corrected_unit}."
            },
            {
                "role": "user",
                "content": json.dumps(prompt)
            }
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    try:
        content = completion.choices[0].message.content
        verdict = json.loads(content)
        
        kept_entities = []
        valid_map = {item["id"]: item for item in verdict.get("validated_entities", [])}
        
        for i, original in enumerate(entities):
            decision = valid_map.get(i)
            if decision and decision.get("decision") == "KEEP":
                # Apply corrections from LLM
                if "corrected_name" in decision:
                    original["entity"] = decision["corrected_name"]
                    original["normalized"] = normalize(decision["corrected_name"])
                if "corrected_unit" in decision:
                    original["unit"] = decision["corrected_unit"]
                    
                kept_entities.append(original)
                
        return kept_entities

    except Exception as e:
        print(f"[ERROR] LLM Validation failed: {e}")
        return entities # Fallback: return everything if LLM fails (Fail-Open for debug, Fail-Closed for prod)


# =====================================================
# LLM: NORMALIZATION
# =====================================================

def llm_normalize_entities(payload: Dict) -> Dict:
    # Pass-through for now, can add specific LOINC mapping here later
    return payload


# =====================================================
# RESIDUAL & CONCLUSION
# =====================================================

def build_residual_text(original_text: str, kept_entities: List[Dict]) -> str:
    # Remove the parts of text that were extracted as entities
    # This leaves "Unstructured Comments" or "Doctor's Notes"
    # Simple implementation: Return lines that didn't generate an entity
    
    extracted_contexts = {e["context"] for e in kept_entities}
    lines = original_text.splitlines()
    residual = []
    
    for line in lines:
        if line.strip() and line.strip() not in extracted_contexts:
            # Simple heuristic: if the line looks like a lab result but wasn't extracted, skip it
            # If it looks like text, keep it.
            if len(line) > 20: 
                residual.append(line.strip())
                
    return "\n".join(residual)

def llm_reason_over_residual(residual_text: str) -> str:
    if len(residual_text) < 10: return "No significant comments."
    
    completion = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Summarize the clinical impression from this text. No new facts."},
            {"role": "user", "content": residual_text}
        ]
    )
    return completion.choices[0].message.content

# =====================================================
# SAVE OUTPUT
# =====================================================

def save_result_json(output: Dict, base_dir: str = "results") -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, base_dir)
    os.makedirs(results_dir, exist_ok=True)

    source = output["doc_metadata"].get("source", "document")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"nlp_output_{timestamp}.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return path

# =====================================================
# TEST
# =====================================================
if __name__ == "__main__":
    docs = [{
        "text": """
i SO : 9001-2008

SARVODAYA HOSPITAL Dr. (Capt) Atul per (Rott) MB Path
KJ-7, Kavi Nagar, Ghaziabad (U-P.) Regn No. MCI 3426

mi

[10 Investigation Observed Value Unit Biological Ref interval
IOCHEMISTRY

- VER FUNCTION TEST (LFT)

BILIRUBIN TOTAL es mg/dl 0,30- 1,20

CONJUGATED (D. BILIRUBIN) 7.79 H mg/dl 0.00 - 0.30
UNCONJUGATED (1.0.BILIRUBIN) 1.63 H mg/dl 0.00 - 0.70
SGOT 162 H WAL 0.00 - 46.00
SGPT 86 H WWAL 0.00 - 49.00
ALKALINE PHOSPHATASE 396 H U/L 42.00 - 128.00
TOTAL PROTEIN 6.2 gnvdl 6.20 - 3.00
ALBUMIN 3.7 L omdt 3.80 - 5.40
GLOBULIN 2.5 giivdl 1,50 - 3.60
AWG RATIO — 1.48 1.0-2.0
GAMMAT-GT 263 H IU/L 11,00 - 50.00
wo,

Specimen : SERUM
™ END OF REPORT **

FACILITIES : FOR HORMONES ASSAYS, FNAC, HISTOPATHOLOGY, BONE MARROW ASPIRATION & BIOPSY WITH MICRO PHOTOGRAPHS.
NOTE : ABOVE MENTIONED FINDINGS ARE A PROFESSIONAL J AD NOT A FINAL DIAGNGSIS, ALL LABORATORY TESTS & OTHER
INVESTIGATION RESULTS ARE TO BE CORELATED CLINIC-PATHOLOGICALLY, DISCREPANCIES, IF ANY, NECESSITATE REVIEW/REPEAT OF THE TESTS.

CLINICAL CORELATION IS MANDATORY
""",
        "date": "2024-08-14",
        "source": "sample_note.txt"
    }]

    final_output = extract_and_process(docs)
    path = save_result_json(final_output)

    print(json.dumps(final_output, indent=2))
    print(f"\n[INFO] Saved to {path}")
