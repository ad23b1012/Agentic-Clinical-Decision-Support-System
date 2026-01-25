import streamlit as st
import tempfile
from pathlib import Path
import json
from dotenv import load_dotenv

# -------------------------------------------------
# ENV (MUST be first)
# -------------------------------------------------
load_dotenv()

# -------------------------------------------------
# PIPELINE IMPORTS
# -------------------------------------------------
from app.state import ClinicalState
from app.orchestrator import ClinicalOrchestrator
from services.answer import SYSTEM_PROMPT, run_clinical_reasoning


# -------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="MediMind AI – Clinical Decision Support",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 MediMind AI")
    st.caption("Agentic Clinical Decision Support")

    st.divider()

    st.markdown("### 📌 Capabilities")
    st.markdown(
        """
        • Multi-report ingestion  
        • Deterministic clinical NLP  
        • Date-normalized patient history  
        • LLM-based longitudinal reasoning  
        • Differential diagnosis generation  
        """
    )

    st.divider()

    st.markdown("### ⚠️ Safety Guardrails")
    st.markdown(
        """
        • No diagnosis confirmation  
        • No hallucination  
        • Evidence-based reasoning only  
        • Structured & auditable outputs  
        """
    )

    st.divider()
    st.caption("Version: v1.0.0")


# -------------------------------------------------
# MAIN HEADER
# -------------------------------------------------
st.title("🧠 MediMind AI")
st.subheader("Clinical Decision Support System")
st.caption("Upload clinical reports to generate a factual longitudinal summary and differential diagnoses.")

st.divider()


# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------
uploaded_files = st.file_uploader(
    "📤 Upload Clinical Reports (PDF / Images)",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Please upload one or more clinical reports to begin analysis.")
    st.stop()


# -------------------------------------------------
# SAVE FILES TEMPORARILY
# -------------------------------------------------
temp_dir = tempfile.TemporaryDirectory()
file_paths = []

for file in uploaded_files:
    temp_path = Path(temp_dir.name) / file.name
    with open(temp_path, "wb") as f:
        f.write(file.read())
    file_paths.append(str(temp_path))

st.success(f"Uploaded {len(file_paths)} file(s)")


# -------------------------------------------------
# RUN PIPELINE
# -------------------------------------------------
if st.button("🚀 Run Clinical Analysis", use_container_width=True):

    with st.spinner("Running full agentic clinical pipeline…"):
        orchestrator = ClinicalOrchestrator()
        state = ClinicalState(file_paths=file_paths)

        # -----------------------------
        # STEP 1: INGESTION
        # -----------------------------
        state = orchestrator.run_ingestion(state)

        if not state.raw_documents:
            st.error("Ingestion failed. No documents extracted.")
            st.stop()

        # -----------------------------
        # STEP 2: CLINICAL NLP
        # -----------------------------
        state = orchestrator.run_clinical_nlp(state)

        if not state.normalized_nlp_results:
            st.error("Clinical NLP produced no usable results.")
            st.stop()

        # -----------------------------
        # STEP 3: LLM REASONING (OSS-120B)
        # -----------------------------
        reasoning_output_path = run_clinical_reasoning(
            SYSTEM_PROMPT,
            state.normalized_nlp_results
        )

        with open(reasoning_output_path, "r", encoding="utf-8") as f:
            reasoning_output = f.read()

    st.success("Clinical analysis completed successfully")

    # -------------------------------------------------
    # PARSE REASONING OUTPUT
    # -------------------------------------------------
    try:
        parsed = json.loads(reasoning_output)
    except json.JSONDecodeError:
        st.error("Reasoning agent returned invalid JSON.")
        st.code(reasoning_output)
        st.stop()

    # -------------------------------------------------
    # RESULTS UI
    # -------------------------------------------------
    st.divider()
    st.header("📄 Clinical Reasoning Results")

    # -----------------------------
    # CLINICAL SUMMARY
    # -----------------------------
    st.markdown("### 🧾 Clinical Summary")

    with st.container(border=True):
        st.markdown(
            parsed.get("clinical_summary", "_No clinical summary generated._")
        )

    # -----------------------------
    # DIFFERENTIAL DIAGNOSES
    # -----------------------------
    st.markdown("### 🧪 Differential Diagnoses (Prioritized)")

    diffs = parsed.get("differential_diagnoses", [])

    if not diffs:
        st.info("No differential diagnoses were generated due to limited evidence.")
    else:
        for idx, dx in enumerate(diffs, start=1):
            with st.container(border=True):
                col1, col2 = st.columns([1, 12])

                with col1:
                    st.markdown(f"**#{idx}**")

                with col2:
                    st.markdown(f"**{dx.get('name', 'Unknown Diagnosis')}**")
                    st.markdown(
                        dx.get("justification", "_No justification provided._")
                    )

    # -----------------------------
    # RAW JSON (AUDIT MODE)
    # -----------------------------
    with st.expander("🔍 View Raw JSON Output (Audit / Debug)"):
        st.code(parsed, language="json")

    # -------------------------------------------------
    # WARNINGS
    # -------------------------------------------------
    if state.errors:
        st.divider()
        st.subheader("⚠️ Pipeline Warnings")
        for err in state.errors:
            st.warning(err)
