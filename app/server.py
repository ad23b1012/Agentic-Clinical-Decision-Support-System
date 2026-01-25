from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="Agentic Clinical Decision Support System",
    description="Processes clinical documents into structured summaries and differential diagnoses",
    version="1.0.0"
)

app.include_router(router, prefix="/api")
