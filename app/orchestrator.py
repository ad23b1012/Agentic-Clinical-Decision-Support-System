import json
from app.state import ClinicalState
from typing import List, Dict
from services.ingestion import ingest_files
from services.clinical_nlp import extract_and_process, save_result_json
from services.embedding import embed_clinical_json
from services.upsert import upsert_embeddings
from services.date_normalizer import update_dates_consistently
from services.answer import run_clinical_reasoning, SYSTEM_PROMPT
from app.orchestrator_mind import query_clinical_system

class ClinicalOrchestrator:
    """
    Coordinates pipeline execution.
    """

    # ----------------------------
    # STEP 1: INGESTION
    # ----------------------------
    def run_ingestion(self, state: ClinicalState) -> ClinicalState:
        state.current_step = "ingestion"

        try:
            docs = ingest_files(state.file_paths)
            if not docs:
                state.add_error("Ingestion produced no documents")
            else:
                state.raw_documents = docs
        except Exception as e:
            state.add_error(f"Ingestion failed: {e}")

        return state

    # ----------------------------
    # STEP 2: CLINICAL NLP + DATE NORMALIZATION
    # ----------------------------
    def run_clinical_nlp(self, state: ClinicalState) -> ClinicalState:
        state.current_step = "clinical_nlp"

        if not state.raw_documents:
            state.add_error("No documents for NLP")
            return state

        for doc in state.raw_documents:
            try:
                result = extract_and_process([doc])
                save_result_json(result)
                state.nlp_results.append(result)
            except Exception as e:
                state.add_error(f"NLP failed for {doc.get('source')}: {e}")

        if not state.nlp_results:
            return state

        # ---- date normalization (global) ----
        canonical = update_dates_consistently(state.nlp_results[0])
        state.normalized_date = canonical["doc_metadata"]["date"]

        for item in state.nlp_results:
            state.normalized_nlp_results.append(
                update_dates_consistently(item)
            )

        return state

    # ----------------------------
    # STEP 3: EMBEDDING
    # ----------------------------
    def run_embedding(self, state: ClinicalState) -> ClinicalState:
        state.current_step = "embedding"

        for doc in state.normalized_nlp_results:
            try:
                emb = embed_clinical_json(doc)
                state.embedding_results.append(emb)
            except Exception as e:
                state.add_error(f"Embedding failed: {e}")

        return state

    # ----------------------------
    # STEP 4: VECTOR STORE
    # ----------------------------
    def run_vector_upsert(self, state: ClinicalState) -> ClinicalState:
        state.current_step = "vector_store"

        for emb in state.embedding_results:
            try:
                res = upsert_embeddings(emb)
                state.vector_store_results.append(res)
            except Exception as e:
                state.add_error(f"Vector upsert failed: {e}")

        return state

    # ----------------------------
    # STEP 5: FINAL REASONING (OSS-120B)
    # ----------------------------
    def run_reasoning(self, state: ClinicalState) -> ClinicalState:
        state.current_step = "clinical_reasoning"

        if not state.normalized_nlp_results:
            state.add_error("No normalized NLP results for reasoning")
            return state

        try:
            # 🔥 PASS FULL PATIENT HISTORY
            output_path = run_clinical_reasoning(
                SYSTEM_PROMPT,
                state.normalized_nlp_results
            )

            with open(output_path, "r", encoding="utf-8") as f:
                state.reasoning_result = json.load(f)

        except Exception as e:
            state.add_error(f"Reasoning failed: {e}")

        return state
    
    # ==========================================================
    # PIPELINE: ONLINE CHAT / QUERY
    # ==========================================================

    async def answer_user_query(self, query: str, chat_history: List = None) -> Dict:
        """
        Wraps the LangGraph reasoning engine.
        call this from your chat interface or API.
        """
        if chat_history is None:
            chat_history = []
            
        # Delegate to the specialized Mind Orchestrator
        return await query_clinical_system(query, chat_history)
