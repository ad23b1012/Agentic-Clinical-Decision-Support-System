from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from app.state import ClinicalState
from app.orchestrator import ClinicalOrchestrator


def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    file_paths = [
        PROJECT_ROOT / "samples" / "report_1_medic.png",
        # PROJECT_ROOT / "samples" / "report_2_medic.png",
        # PROJECT_ROOT / "samples" / "report_3.jpeg",
        # PROJECT_ROOT / "samples" / "report_4.jpeg",
        # PROJECT_ROOT / "samples" / "report_5.jpeg",
    ]

    file_paths = [str(p) for p in file_paths if p.exists()]
    if not file_paths:
        print("[ERROR] No valid input files")
        return

    state = ClinicalState(file_paths=file_paths)
    orchestrator = ClinicalOrchestrator()

    state = orchestrator.run_ingestion(state)
    state = orchestrator.run_clinical_nlp(state)
    state = orchestrator.run_embedding(state)
    state = orchestrator.run_vector_upsert(state)

    # 🔥 FINAL REASONING STEP
    state = orchestrator.run_reasoning(state)

    print("\n================ FINAL CLINICAL REASONING ================\n")
    if state.reasoning_result:
        print(state.reasoning_result)
    else:
        print("[ERROR] No reasoning output produced")

    if state.errors:
        print("\n[WARNINGS]")
        for e in state.errors:
            print("-", e)


if __name__ == "__main__":
    main()
