"""
Temporal Agent
--------------
Responsibilities:
- Build a chronological patient timeline
- Group events by entity
- Detect progression patterns
- Detect contradictions over time

NO diagnosis.
NO reasoning.
Pure temporal intelligence.
"""

from typing import List, Dict
from collections import defaultdict
from datetime import datetime


# -----------------------------
# Public API
# -----------------------------

def build_timeline(entities: List[Dict]) -> Dict:
    """
    Entry point used by orchestrator.

    Input:
        Output of clinical_nlp_agent.extract_entities()

    Output:
        {
          "timeline": [...],
          "entity_history": {...},
          "progressions": [...],
          "conflicts": [...]
        }
    """

    ordered_events = _order_by_time(entities)
    entity_history = _group_by_entity(ordered_events)
    progressions = _detect_progression(entity_history)
    conflicts = _detect_conflicts(entity_history)

    return {
        "timeline": ordered_events,
        "entity_history": entity_history,
        "progressions": progressions,
        "conflicts": conflicts,
    }


# -----------------------------
# Timeline construction
# -----------------------------

def _order_by_time(entities: List[Dict]) -> List[Dict]:
    """
    Sort events chronologically.
    Unknown dates go last.
    """

    def parse_date(e):
        if e.get("date"):
            try:
                return datetime.strptime(e["date"], "%Y-%m-%d")
            except Exception:
                pass
        return datetime.max

    return sorted(entities, key=parse_date)


def _group_by_entity(events: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Groups events by normalized entity.
    """

    history = defaultdict(list)

    for event in events:
        key = event["normalized"]
        history[key].append(event)

    return dict(history)


# -----------------------------
# Progression detection
# -----------------------------

def _detect_progression(entity_history: Dict[str, List[Dict]]) -> List[Dict]:
    """
    Detects:
    - Repeated symptoms over time
    - Worsening lab trends
    """

    progressions = []

    for entity, events in entity_history.items():
        if len(events) < 2:
            continue

        # Case 1: Recurrent symptoms / conditions
        if events[0]["type"] in {"symptom", "condition"}:
            progressions.append({
                "entity": entity,
                "type": events[0]["type"],
                "pattern": "recurrent",
                "occurrences": len(events),
                "dates": [e["date"] for e in events],
            })

        # Case 2: Lab value trend
        if events[0]["type"] == "lab":
            values = []
            for e in events:
                try:
                    values.append(float(e["value"]))
                except Exception:
                    continue

            if len(values) >= 2:
                trend = "stable"
                if values[-1] > values[0]:
                    trend = "increasing"
                elif values[-1] < values[0]:
                    trend = "decreasing"

                progressions.append({
                    "entity": entity,
                    "type": "lab",
                    "pattern": trend,
                    "values": values,
                    "dates": [e["date"] for e in events],
                })

    return progressions


# -----------------------------
# Conflict detection
# -----------------------------

def _detect_conflicts(entity_history: Dict[str, List[Dict]]) -> List[Dict]:
    """
    Detects contradictions:
    - Entity marked negated and later affirmed (or vice versa)
    """

    conflicts = []

    for entity, events in entity_history.items():
        negation_states = set()

        for e in events:
            negation_states.add(e.get("negated", False))

        if len(negation_states) > 1:
            conflicts.append({
                "entity": entity,
                "issue": "negation_conflict",
                "events": [
                    {
                        "date": e.get("date"),
                        "negated": e.get("negated"),
                        "context": e.get("context"),
                        "source": e.get("source"),
                    }
                    for e in events
                ],
            })

    return conflicts


# -----------------------------
# MAIN (Manual Test)
# -----------------------------

def main():
    sample_entities = [
        {
            "entity": "chest pain",
            "normalized": "CHEST_PAIN",
            "type": "symptom",
            "negated": False,
            "date": "2024-06-01",
            "context": "Patient presents with chest pain",
            "source": "visit1.txt",
        },
        {
            "entity": "chest pain",
            "normalized": "CHEST_PAIN",
            "type": "symptom",
            "negated": False,
            "date": "2024-07-10",
            "context": "Chest pain persists",
            "source": "visit2.txt",
        },
        {
            "entity": "fever",
            "normalized": "FEVER",
            "type": "symptom",
            "negated": True,
            "date": "2024-07-10",
            "context": "Denies fever",
            "source": "visit2.txt",
        },
        {
            "entity": "fever",
            "normalized": "FEVER",
            "type": "symptom",
            "negated": False,
            "date": "2024-08-01",
            "context": "Patient has fever",
            "source": "visit3.txt",
        },
        {
            "entity": "hemoglobin",
            "normalized": "HEMOGLOBIN",
            "type": "lab",
            "value": "12.0",
            "unit": "g/dL",
            "negated": False,
            "date": "2024-06-01",
            "context": "Hemoglobin 12.0 g/dL",
            "source": "lab1.txt",
        },
        {
            "entity": "hemoglobin",
            "normalized": "HEMOGLOBIN",
            "type": "lab",
            "value": "10.2",
            "unit": "g/dL",
            "negated": False,
            "date": "2024-08-01",
            "context": "Hemoglobin 10.2 g/dL",
            "source": "lab2.txt",
        },
    ]

    timeline = build_timeline(sample_entities)

    print("\n=== TIMELINE ===")
    for e in timeline["timeline"]:
        print(e)

    print("\n=== PROGRESSIONS ===")
    for p in timeline["progressions"]:
        print(p)

    print("\n=== CONFLICTS ===")
    for c in timeline["conflicts"]:
        print(c)


if __name__ == "__main__":
    main()
