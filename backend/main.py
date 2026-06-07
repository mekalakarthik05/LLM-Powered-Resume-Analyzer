from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
import asyncio
import os
import tempfile
import json
import logging
import uuid
import time
from typing import Any, Awaitable, Callable, Dict, Optional

from pydantic import BaseModel

from parser import extract_text_from_file
from scorer import rank_resumes
from extractor import enrich_resume_data
from llm_analysis import generate_rag_insight, generate_optimized_resume
from resume_pdf import pdf_download_filename, render_resume_pdf

# Configure robust logging for debugability and monitoring
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title='Aura API', description='Next-Gen Candidate Optimization Engine')

# CORS configuration from environment
CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_MIME_TYPES = {
    'application/pdf', 
    'text/plain', 
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
}

PROCESSING_STEPS = [
    {"key": "validate_input", "label": "Validate request"},
    {"key": "save_upload", "label": "Read uploaded resume"},
    {"key": "extract_text", "label": "Extract resume text"},
    {"key": "enrich_resume", "label": "Structure resume data"},
    {"key": "score_resume", "label": "Score ATS match"},
    {"key": "rag_chunk", "label": "Chunk resume for RAG"},
    {"key": "rag_jd_embedding", "label": "Embed job description"},
    {"key": "rag_resume_embeddings", "label": "Embed resume chunks"},
    {"key": "rag_rank", "label": "Rank relevant evidence"},
    {"key": "llm_insight", "label": "Run recruiter LLM"},
    {"key": "llm_parse", "label": "Parse AI insight"},
    {"key": "draft_resume", "label": "Generate optimized draft"},
    {"key": "package_report", "label": "Package final report"},
]
STEP_LABELS = {step["key"]: step["label"] for step in PROCESSING_STEPS}
ProgressEmitter = Optional[Callable[[Dict[str, Any]], Awaitable[None]]]


class ResumePdfRequest(BaseModel):
    optimized_resume: Dict[str, Any]


def _json_line(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, default=str) + "\n"


async def _run_optimization_pipeline(
    job_description: str,
    file: UploadFile,
    request_id: str,
    progress: ProgressEmitter = None,
) -> Dict[str, Any]:
    trace = []
    phase_started_at = {}

    async def emit(
        key: str,
        status: str,
        detail: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = time.perf_counter()
        if status == "active":
            phase_started_at[key] = now

        event = {
            "type": "progress",
            "request_id": request_id,
            "key": key,
            "label": STEP_LABELS.get(key, key.replace("_", " ").title()),
            "status": status,
            "detail": detail,
            "meta": meta or {},
        }

        if status in {"done", "warning", "error"} and key in phase_started_at:
            event["duration_ms"] = int((now - phase_started_at[key]) * 1000)

        trace.append({k: v for k, v in event.items() if k != "type"})

        if progress:
            await progress(event)

    await emit("validate_input", "active", "Checking the job description and uploaded file.")

    if not job_description or not job_description.strip():
        await emit("validate_input", "error", "Job description is required.")
        raise HTTPException(status_code=400, detail='Job description is required and cannot be empty.')

    if not file or not file.filename:
        await emit("validate_input", "error", "No valid resume file was uploaded.")
        raise HTTPException(status_code=400, detail='No valid file uploaded.')

    if file.content_type not in ALLOWED_MIME_TYPES and not file.filename.lower().endswith(('.pdf', '.txt', '.docx')):
        await emit("validate_input", "error", "Unsupported file type.")
        raise HTTPException(status_code=400, detail='Invalid file type. Only PDF, TXT, and DOCX are supported.')

    await emit(
        "validate_input",
        "done",
        "Inputs validated.",
        {"filename": file.filename, "content_type": file.content_type},
    )

    logger.info(f"[{request_id}] Processing optimization request for JD length: {len(job_description)} and file: {file.filename}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, file.filename)

        await emit("save_upload", "active", "Reading the uploaded resume into a temporary workspace.")
        size = 0
        with open(temp_file_path, 'wb') as buffer:
            while chunk := await file.read(8192):
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    await emit("save_upload", "error", "Resume file is larger than the 5 MB limit.", {"bytes": size})
                    raise HTTPException(status_code=413, detail='File too large. Maximum size is 5MB.')
                buffer.write(chunk)

        await emit(
            "save_upload",
            "done",
            f"Stored {(size / 1024):.0f} KB for parsing.",
            {"bytes": size},
        )

        try:
            await emit("extract_text", "active", "Extracting text from the resume file.")
            extracted_text = extract_text_from_file(temp_file_path)
            if not extracted_text or not str(extracted_text).strip():
                await emit("extract_text", "error", "No meaningful text could be extracted from the resume.")
                raise HTTPException(status_code=422, detail='Could not extract meaningful text from the uploaded file. Please ensure it is a valid text-based document.')

            await emit(
                "extract_text",
                "done",
                f"Extracted {len(extracted_text)} characters from the resume.",
                {"characters": len(extracted_text)},
            )

            raw_data = {'filename': file.filename, 'text': extracted_text}
            await emit("enrich_resume", "active", "Structuring contact, skills, experience, and education signals.")
            try:
                enriched_data = enrich_resume_data(raw_data)
                await emit("enrich_resume", "done", "Resume structure enrichment complete.")
            except Exception as e:
                logger.error(f"Enrichment Failed: {e}")
                enriched_data = raw_data
                await emit(
                    "enrich_resume",
                    "warning",
                    "Resume enrichment failed; continuing with raw extracted text.",
                    {"error": str(e)},
                )

            await emit("score_resume", "active", "Computing ATS, keyword, semantic, and evidence scores.")
            try:
                ranked = rank_resumes(job_description, [enriched_data])
                if not ranked:
                    raise ValueError("Scoring returned empty results.")
                result_data = ranked[0]
                await emit(
                    "score_resume",
                    "done",
                    f"Score calculated: {result_data.get('score', 0)}.",
                    {"score": result_data.get("score", 0)},
                )
            except Exception as e:
                logger.error(f"Scoring Failed: {e}")
                await emit("score_resume", "error", "Scoring failed.", {"error": str(e)})
                raise HTTPException(status_code=500, detail='Error ranking resume against Job Description.')

            try:
                insight_response = await generate_rag_insight(job_description, extracted_text, progress=emit)
                if isinstance(insight_response, dict):
                    result_data['ai_insight'] = insight_response.get('analysis', {})
                    result_data['recruiter_verdict'] = insight_response.get('recruiter_verdict')
                    result_data['weighted_score'] = insight_response.get('weighted_score')
                    result_data['confidence_score'] = insight_response.get('confidence_score')
                    result_data['seniority_inference'] = insight_response.get('seniority_inference')
                    result_data['pipeline_warnings'] = insight_response.get('pipeline_warnings', [])
                else:
                    logger.error(f"[{request_id}] Unexpected insight response type: {type(insight_response)}")
                    result_data['ai_insight'] = {'summary': 'Analysis complete.'}
                    result_data['pipeline_warnings'] = ['Unexpected LLM response shape; fallback summary was used.']
                    await emit("llm_parse", "warning", "Unexpected LLM response shape; fallback summary was used.")
            except Exception as e:
                logger.error(f"[{request_id}] LLM Insight Generation Failed: {e}", exc_info=True)
                await emit(
                    "llm_insight",
                    "warning",
                    "LLM insight generation failed; continuing with deterministic scoring output.",
                    {"error": str(e)},
                )
                result_data['ai_insight'] = {
                    'summary': 'Analysis complete. Refer to the score breakdown and recommendations above for resume improvement guidance.',
                    'missing_skills': [],
                    'suggestions': [],
                    'breakdown': [],
                    'resume_signals': [],
                    'rewrite_strategy': []
                }
                result_data['recruiter_verdict'] = None
                result_data['weighted_score'] = None
                result_data['confidence_score'] = None
                result_data['seniority_inference'] = None
                result_data['pipeline_warnings'] = [f"LLM insight generation failed: {str(e)}"]

            await emit("draft_resume", "active", "Creating the fact-preserving optimized draft.")
            try:
                result_data['optimized_resume'] = generate_optimized_resume(
                    job_description,
                    extracted_text,
                    result_data
                )
                draft_status = "done" if result_data['optimized_resume'].get("can_generate") else "warning"
                draft_detail = (
                    "Optimized resume draft generated."
                    if result_data['optimized_resume'].get("can_generate")
                    else "Draft generation was gated by the evidence policy."
                )
                await emit(
                    "draft_resume",
                    draft_status,
                    draft_detail,
                    {"can_generate": result_data['optimized_resume'].get("can_generate", False)},
                )
            except Exception as e:
                logger.error(f"Optimized Resume Generation Failed: {e}", exc_info=True)
                result_data['optimized_resume'] = {
                    'can_generate': False,
                    'reason': 'Could not build a fact-preserving resume draft.',
                    'draft': '',
                    'format': 'text',
                    'integrity_rules': ['No new claims were generated.'],
                    'blocked_missing_terms': []
                }
                await emit("draft_resume", "warning", "Optimized resume draft generation failed.", {"error": str(e)})

            await emit("package_report", "active", "Preparing the final response payload.")
            result_data.setdefault('recruiter_verdict', None)
            result_data.setdefault('weighted_score', None)
            result_data.setdefault('confidence_score', None)
            result_data.setdefault('seniority_inference', None)
            result_data.setdefault('ai_insight', {})
            result_data.setdefault('ats_signals', {})
            result_data.setdefault('score_breakdown', [])
            result_data.setdefault('pipeline_warnings', [])
            result_data['original_resume_text'] = extracted_text
            result_data['request_id'] = request_id

            await emit("package_report", "done", "Report payload ready.")
            result_data['processing_trace'] = trace.copy()

            logger.info(f"[{request_id}] Analysis complete. Score: {result_data.get('score', 0)}")
            return result_data

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f'[{request_id}] Internal Processing Error: {e}', exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to process the document: {str(e)}")

@app.get('/api/health')
async def health_check():
    """Health check endpoint for monitoring and deployment orchestration"""
    return {
        "status": "healthy",
        "version": "4.0",
        "service": "Aura AI Resume Analysis Engine"
    }


@app.post('/api/resume/pdf')
async def download_resume_pdf(request: ResumePdfRequest):
    optimized_resume = request.optimized_resume or {}
    draft = str(optimized_resume.get("draft") or "").strip()
    if not draft:
        raise HTTPException(status_code=400, detail="No optimized resume draft is available for PDF export.")

    try:
        pdf_bytes = render_resume_pdf(optimized_resume)
    except Exception as exc:
        logger.error("PDF generation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate resume PDF.") from exc

    filename = pdf_download_filename(optimized_resume)
    return Response(
        content=pdf_bytes,
        media_type='application/pdf',
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

@app.post('/api/optimize')
async def optimize_resume(
    job_description: str = Form(..., description="The target job description"),
    file: UploadFile = File(..., description="The candidate's resume file")
):
    request_id = str(uuid.uuid4())[:8]
    try:
        return await _run_optimization_pipeline(job_description, file, request_id)
    except HTTPException:
        raise
    except Exception as e:
         logger.critical(f"[{request_id}] Unhandled system error: {e}", exc_info=True)
         raise HTTPException(status_code=500, detail="A critical system error occurred while processing the request.")


@app.post('/api/optimize/stream')
async def optimize_resume_stream(
    job_description: str = Form(..., description="The target job description"),
    file: UploadFile = File(..., description="The candidate's resume file")
):
    request_id = str(uuid.uuid4())[:8]

    async def stream_events():
        queue: asyncio.Queue = asyncio.Queue()

        async def enqueue(event: Dict[str, Any]) -> None:
            await queue.put(event)
            await asyncio.sleep(0)

        async def run_pipeline() -> None:
            try:
                await enqueue({"type": "steps", "request_id": request_id, "steps": PROCESSING_STEPS})
                result = await _run_optimization_pipeline(job_description, file, request_id, progress=enqueue)
                await enqueue({"type": "result", "request_id": request_id, "data": result})
            except HTTPException as exc:
                await enqueue({
                    "type": "error",
                    "request_id": request_id,
                    "status_code": exc.status_code,
                    "detail": exc.detail,
                })
            except Exception as exc:
                logger.critical(f"[{request_id}] Unhandled streaming system error: {exc}", exc_info=True)
                await enqueue({
                    "type": "error",
                    "request_id": request_id,
                    "status_code": 500,
                    "detail": "A critical system error occurred while processing the request.",
                })
            finally:
                await queue.put(None)

        task = asyncio.create_task(run_pipeline())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield _json_line(event)
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(stream_events(), media_type="application/x-ndjson")
