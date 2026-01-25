import shutil
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.state import ClinicalState
from app.orchestrator import ClinicalOrchestrator

router = APIRouter()


@router.post("/analyze")
async def analyze_documents(files: List[UploadFile] = File(...)):
    """
    Full clinical pipeline:
    - Ingestion
    - NLP
    - Embedding
    - Vector store
    - Reasoning
    """

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []

        for file in files:
            path = Path(tmpdir) / file.filename
            with open(path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            paths.append(str(path))

        # -----------------------------
        # Run pipeline
        # -----------------------------
        state = ClinicalState(file_paths=paths)
        orchestrator = ClinicalOrchestrator()

        state = orchestrator.run_ingestion(state)
        state = orchestrator.run_clinical_nlp(state)
        state = orchestrator.run_embedding(state)
        state = orchestrator.run_vector_upsert(state)
        state = orchestrator.run_reasoning(state)

        if not state.reasoning_result:
            raise HTTPException(
                status_code=500,
                detail="Clinical reasoning failed"
            )

        return {
            "status": "success",
            "normalized_date": state.normalized_date,
            "reasoning": state.reasoning_result,
            "errors": state.errors
        }
