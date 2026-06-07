import json
import asyncio
import os
import logging
import re
from typing import Awaitable, Callable, Dict, Any, List, Optional

import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

try:
    from resume_profile_overrides import get_profile_override
except ModuleNotFoundError:
    from backend.resume_profile_overrides import get_profile_override

logger = logging.getLogger(__name__)

# Ollama local LLM configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "alex-recruiter") # Custom local model
EMBED_MODEL = "nomic-embed-text"  # Specialized embedding model
OLLAMA_CLIENT = ollama.Client(host=OLLAMA_HOST)  # Reusable client to avoid recreating per call
OLLAMA_ASYNC_CLIENT = ollama.AsyncClient(host=OLLAMA_HOST)  # Reusable async client for chat

ProgressCallback = Optional[Callable[[str, str, str, Optional[Dict[str, Any]]], Awaitable[None]]]


async def _emit_progress(
    progress: ProgressCallback,
    key: str,
    status: str,
    detail: str,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    if progress:
        await progress(key, status, detail, meta or {})

def get_ollama_embedding(text: str) -> list:
    """Helper to get embeddings from Ollama"""
    try:
        # Using specialized embedding model for better retrieval
        response = OLLAMA_CLIENT.embeddings(model=EMBED_MODEL, prompt=text)
        return response['embedding']
    except Exception as e:
        logger.error(f"Failed to get ollama embedding: {e}")
        return []


async def get_ollama_embedding_async(text: str) -> list:
    """Async helper to get embeddings from Ollama while progress can stream."""
    try:
        response = await OLLAMA_ASYNC_CLIENT.embeddings(model=EMBED_MODEL, prompt=text)
        return response["embedding"]
    except Exception as e:
        logger.error(f"Failed to get async ollama embedding: {e}")
        return []


def _extract_balanced_json_object(text: str) -> str:
    """Return the first balanced JSON object from a model response."""
    if not text:
        return text

    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]

    end = text.rfind("}")
    return text[start:end + 1] if end > start else text[start:]


def _parse_llm_json(content: str) -> Dict[str, Any]:
    """
    Parse Ollama JSON with small, targeted repairs for common local-model output
    issues: markdown fences, stop tokens, surrounding prose, trailing commas, and
    unquoted snake_case keys.
    """
    cleaned = (content or "").strip()
    if not cleaned:
        raise ValueError("Could not parse LLM JSON response: empty response content.")

    cleaned = cleaned.replace("</END_JSON>", "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    candidates = []
    for candidate in (cleaned, _extract_balanced_json_object(cleaned)):
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    last_error: Optional[Exception] = None
    for candidate in candidates:
        variants = [candidate]
        no_trailing_commas = re.sub(r",\s*([}\]])", r"\1", candidate)
        variants.append(no_trailing_commas)
        variants.append(
            re.sub(
                r'(?m)([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)',
                r'\1"\2"\3',
                no_trailing_commas,
            )
        )

        for variant in variants:
            try:
                parsed = json.loads(variant)
                if isinstance(parsed, dict):
                    return parsed
                raise ValueError("LLM JSON root must be an object.")
            except Exception as exc:
                last_error = exc

    raise ValueError(f"Could not parse LLM JSON response: {last_error}")


def _extract_chat_message_parts(response: Any) -> Dict[str, Any]:
    """Support both Ollama ChatResponse objects and dict-like responses."""
    message = getattr(response, "message", None)
    if message is None and isinstance(response, dict):
        message = response.get("message")

    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")

    thinking = getattr(message, "thinking", None)
    if thinking is None and isinstance(message, dict):
        thinking = message.get("thinking")

    done_reason = getattr(response, "done_reason", None)
    if done_reason is None and isinstance(response, dict):
        done_reason = response.get("done_reason")

    return {
        "content": (content or "").strip(),
        "thinking": thinking or "",
        "done_reason": done_reason,
    }


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        items: List[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    items.append(text)
            elif isinstance(item, dict):
                text = " ".join(str(part).strip() for part in item.values() if str(part).strip()).strip()
                if text:
                    items.append(text)
            elif item is not None:
                text = str(item).strip()
                if text:
                    items.append(text)
        return items
    text = str(value).strip()
    return [text] if text else []


def _normalize_breakdown(items: Any) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict):
            signal = (
                item.get("signal")
                or item.get("signal_type")
                or item.get("category")
                or item.get("key")
                or "Resume signal"
            )
            finding = item.get("finding") or item.get("evidence") or item.get("value") or ""
            why_it_matters = item.get("why_it_matters") or item.get("why") or ""
            fix = item.get("fix") or item.get("recommendation") or ""
        else:
            signal = "Resume signal"
            finding = str(item).strip()
            why_it_matters = ""
            fix = ""

        normalized.append(
            {
                "signal": str(signal).strip(),
                "finding": str(finding).strip(),
                "why_it_matters": str(why_it_matters).strip(),
                "fix": str(fix).strip(),
            }
        )
    return [item for item in normalized if item["signal"] or item["finding"]]


def _normalize_resume_signals(items: Any) -> List[str]:
    normalized: List[str] = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            key = str(item.get("key") or item.get("signal") or "").strip()
            value = str(item.get("value") or item.get("finding") or item.get("evidence") or "").strip()
            text = f"{key}: {value}".strip(": ").strip()
        else:
            text = str(item).strip()

        if text:
            normalized.append(text)
    return normalized


def _coerce_number(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_llm_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(data or {})
    normalized["summary"] = str(normalized.get("summary") or "Analysis complete.").strip()
    normalized["missing_skills"] = _string_list(normalized.get("missing_skills"))
    normalized["suggestions"] = _string_list(normalized.get("suggestions"))
    normalized["breakdown"] = _normalize_breakdown(normalized.get("breakdown"))
    normalized["resume_signals"] = _normalize_resume_signals(normalized.get("resume_signals"))
    normalized["rewrite_strategy"] = _string_list(normalized.get("rewrite_strategy"))
    normalized["recruiter_verdict"] = str(normalized.get("recruiter_verdict") or "").strip() or None
    normalized["weighted_score"] = _coerce_number(normalized.get("weighted_score"))
    normalized["confidence_score"] = _coerce_number(normalized.get("confidence_score"))
    normalized["seniority_inference"] = str(normalized.get("seniority_inference") or "").strip() or None
    return normalized

def advanced_rag_retrieval(job_description: str, candidate_text: str) -> str:
    """
    Advanced RAG pipeline: Chunks the resume, embeds it using Ollama, and retrieves 
    the most relevant sections based on the Job Description.
    """
    # Chunk the candidate resume
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(candidate_text)
    
    if not chunks:
        return candidate_text
        
    try:
        # Embed JD
        jd_embedding = get_ollama_embedding(job_description)
        if not jd_embedding:
            return candidate_text[:8000]
            
        jd_emb_np = np.array(jd_embedding)
        
        # Embed chunks
        chunk_embeddings = []
        valid_chunks = []
        for chunk in chunks:
            emb = get_ollama_embedding(chunk)
            if emb:
                chunk_embeddings.append(emb)
                valid_chunks.append(chunk)
                
        if not chunk_embeddings:
            return candidate_text[:8000]
            
        chunk_emb_np = np.array(chunk_embeddings)
        
        # Calculate cosine similarity with safety against divide-by-zero
        jd_norm = np.linalg.norm(jd_emb_np)
        chunk_norms = np.linalg.norm(chunk_emb_np, axis=1)
        
        valid = (chunk_norms > 0) & (jd_norm > 0)
        
        similarities = np.zeros(len(chunk_norms))
        similarities[valid] = np.dot(chunk_emb_np[valid], jd_emb_np) / (
            chunk_norms[valid] * jd_norm
        )
        
        # Get top chunks for recruiter reasoning
        top_k = min(8, len(valid_chunks))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        retrieved_context = "\n\n".join(
            f"[RELEVANT_RESUME_CHUNK] {valid_chunks[i]}" for i in top_indices
        )
        return retrieved_context
        
    except Exception as e:
        logger.error(f"RAG retrieval failed: {e}")
        return candidate_text[:8000]


async def advanced_rag_retrieval_async(
    job_description: str,
    candidate_text: str,
    progress: ProgressCallback = None,
) -> str:
    """
    Async RAG pipeline that reports chunking, embedding, and ranking progress.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(candidate_text)

    await _emit_progress(
        progress,
        "rag_chunk",
        "done" if chunks else "warning",
        f"Prepared {len(chunks)} resume chunks for retrieval." if chunks else "No chunks found; using raw resume text.",
        {"total_chunks": len(chunks)},
    )

    if not chunks:
        await _emit_progress(
            progress,
            "rag_jd_embedding",
            "warning",
            "Skipped semantic embeddings because no resume chunks were available.",
            {"total_chunks": 0},
        )
        await _emit_progress(
            progress,
            "rag_resume_embeddings",
            "warning",
            "Skipped resume chunk embeddings because no resume chunks were available.",
            {"total_chunks": 0},
        )
        await _emit_progress(
            progress,
            "rag_rank",
            "warning",
            "Skipped RAG ranking; using raw resume text for recruiter reasoning.",
            {"total_chunks": 0},
        )
        return candidate_text

    try:
        await _emit_progress(
            progress,
            "rag_jd_embedding",
            "active",
            "Embedding the job description for semantic retrieval.",
            {"model": EMBED_MODEL},
        )
        jd_embedding = await get_ollama_embedding_async(job_description)
        if not jd_embedding:
            await _emit_progress(
                progress,
                "rag_jd_embedding",
                "warning",
                "Job description embedding failed; using the first resume text window.",
                {"model": EMBED_MODEL},
            )
            await _emit_progress(
                progress,
                "rag_resume_embeddings",
                "warning",
                "Skipped resume chunk embeddings because the job description embedding failed.",
                {"total": len(chunks), "valid": 0},
            )
            await _emit_progress(
                progress,
                "rag_rank",
                "warning",
                "Skipped RAG ranking; using the first resume text window.",
                {"total": len(chunks), "valid": 0},
            )
            return candidate_text[:8000]

        await _emit_progress(
            progress,
            "rag_jd_embedding",
            "done",
            "Job description embedding complete.",
            {"model": EMBED_MODEL},
        )

        jd_emb_np = np.array(jd_embedding)

        chunk_embeddings = []
        valid_chunks = []
        total_chunks = len(chunks)
        for index, chunk in enumerate(chunks, start=1):
            await _emit_progress(
                progress,
                "rag_resume_embeddings",
                "active",
                f"Embedding resume chunk {index} of {total_chunks}.",
                {"current": index, "total": total_chunks, "model": EMBED_MODEL},
            )
            emb = await get_ollama_embedding_async(chunk)
            if emb:
                chunk_embeddings.append(emb)
                valid_chunks.append(chunk)

        if not chunk_embeddings:
            await _emit_progress(
                progress,
                "rag_resume_embeddings",
                "warning",
                "Resume chunk embeddings failed; using the first resume text window.",
                {"total": total_chunks, "valid": 0},
            )
            await _emit_progress(
                progress,
                "rag_rank",
                "warning",
                "Skipped RAG ranking because no resume chunk embeddings were available.",
                {"total": total_chunks, "valid": 0},
            )
            return candidate_text[:8000]

        await _emit_progress(
            progress,
            "rag_resume_embeddings",
            "done",
            f"Embedded {len(valid_chunks)} resume chunks.",
            {"total": total_chunks, "valid": len(valid_chunks)},
        )

        await _emit_progress(
            progress,
            "rag_rank",
            "active",
            "Ranking resume chunks against the job description.",
            {"valid_chunks": len(valid_chunks)},
        )

        chunk_emb_np = np.array(chunk_embeddings)
        jd_norm = np.linalg.norm(jd_emb_np)
        chunk_norms = np.linalg.norm(chunk_emb_np, axis=1)

        valid = (chunk_norms > 0) & (jd_norm > 0)
        similarities = np.zeros(len(chunk_norms))
        similarities[valid] = np.dot(chunk_emb_np[valid], jd_emb_np) / (
            chunk_norms[valid] * jd_norm
        )

        top_k = min(8, len(valid_chunks))
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        retrieved_context = "\n\n".join(
            f"[RELEVANT_RESUME_CHUNK] {valid_chunks[i]}" for i in top_indices
        )

        await _emit_progress(
            progress,
            "rag_rank",
            "done",
            f"Selected {top_k} highest-signal resume chunks for recruiter reasoning.",
            {"selected_chunks": top_k, "valid_chunks": len(valid_chunks)},
        )
        return retrieved_context

    except Exception as e:
        logger.error(f"Async RAG retrieval failed: {e}")
        await _emit_progress(
            progress,
            "rag_rank",
            "warning",
            "RAG retrieval failed; using the first resume text window.",
            {"error": str(e)},
        )
        return candidate_text[:8000]


async def generate_rag_insight(
    job_description: str,
    candidate_text: str,
    progress: ProgressCallback = None,
) -> Dict[str, Any]:
    """
    Advanced local LLM evaluation focusing on ATS parsing, hard/soft skill gaps, 
    and actionable resume improvements using a custom Ollama model.
    """
    if not job_description or not candidate_text:
        return {
            "analysis": {
                "summary": "Missing data for analysis.",
                "missing_skills": [],
                "suggestions": ["Please provide both a job description and resume."]
            },
            "recruiter_verdict": None,
            "weighted_score": None,
            "confidence_score": None,
            "seniority_inference": None
        }

    try:
        # Perform Advanced RAG retrieval
        relevant_resume_context = await advanced_rag_retrieval_async(job_description, candidate_text, progress)
        
        # Configure the prompt for the local LLM expert 'Alex Recruiter'
        prompt = f"""
        Act as Alex, an elite, highly critical Senior Tech Recruiter and ATS Expert.
        Evaluate this candidate's resume against the target Job Description. 
        Do NOT sugarcoat your feedback. Be brutally honest, direct, and explicit about critical gaps, weaknesses, and structural flaws. 
        Provide deep, advanced technical feedback and insight. No corporate fluff. Tell it exactly like it is.
        
        Return only one valid JSON object. Use double-quoted JSON keys, no markdown, no prose, and no trailing commas.
        Required schema:
        {{
          "summary": "string",
          "missing_skills": ["string"],
          "suggestions": ["string"],
          "breakdown": [
            {{
              "signal": "string",
              "finding": "string",
              "why_it_matters": "string",
              "fix": "string"
            }}
          ],
          "resume_signals": ["string"],
          "rewrite_strategy": ["string"],
          "recruiter_verdict": "string",
          "weighted_score": 0,
          "confidence_score": 0,
          "seniority_inference": "string"
        }}
        
        Job Description:
        {job_description}
        
        Candidate Resume (Relevant RAG Chunks):
        {relevant_resume_context}
        """
        
        # Call Local Ollama LLM
        await _emit_progress(
            progress,
            "llm_insight",
            "active",
            "Asking the local recruiter model for structured feedback.",
            {"model": OLLAMA_MODEL},
        )
        response = await OLLAMA_ASYNC_CLIENT.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": "You output only a single valid JSON object with double-quoted keys. No markdown, no comments, no trailing commas."},
                {"role": "user", "content": prompt}
            ],
            format="json",
            think=False,
            options={
                "temperature": 0.01,
                "num_predict": 1200
            }
        )
        await _emit_progress(
            progress,
            "llm_insight",
            "done",
            "Local recruiter model response received.",
            {"model": OLLAMA_MODEL},
        )

        response_parts = _extract_chat_message_parts(response)
        result_content = response_parts["content"]
        if not result_content:
            raise ValueError(
                "Local recruiter model returned empty structured content "
                f"(done_reason={response_parts['done_reason']}, has_thinking={bool(response_parts['thinking'])})."
            )

        # Parse and extract recruiter verdict fields
        await _emit_progress(
            progress,
            "llm_parse",
            "active",
            "Parsing the recruiter model response as JSON.",
            None,
        )
        data = _normalize_llm_analysis(_parse_llm_json(result_content))
        await _emit_progress(
            progress,
            "llm_parse",
            "done",
            "Structured recruiter JSON parsed successfully.",
            None,
        )
        
        return {
            "analysis": data,
            "recruiter_verdict": data.get("recruiter_verdict"),
            "weighted_score": data.get("weighted_score"),
            "confidence_score": data.get("confidence_score"),
            "seniority_inference": data.get("seniority_inference"),
            "pipeline_warnings": []
        }
        
    except Exception as e:
        logger.error(f"Local LLM / RAG Pipeline Error: {str(e)}")
        await _emit_progress(
            progress,
            "llm_insight",
            "warning",
            "Local LLM insight fell back to heuristic analysis.",
            {"error": str(e)},
        )
        await _emit_progress(
            progress,
            "llm_parse",
            "warning",
            "Local LLM output could not be parsed; using heuristic fallback insight.",
            {"error": str(e)},
        )
        # Fallback to intelligent heuristic if local LLM fails or model isn't pulled yet
        
        jd_words = set(job_description.lower().split())
        resume_words = set(candidate_text.lower().split())
        common_tech_keywords = {"python", "react", "aws", "docker", "kubernetes", "sql", "nosql", "java", "node", "typescript"}
        
        missing_tech = [word.capitalize() for word in jd_words.intersection(common_tech_keywords) if word not in resume_words]
        
        insight = {
            'summary': 'Your experience aligns with some core requirements, but vital technical keywords from the job description are missing. Focus on adding concrete achievements and metrics to demonstrate your impact.',
            'missing_skills': missing_tech if missing_tech else ['Cloud Infrastructure', 'System Design'],
            'suggestions': [
                'Quantify your achievements with concrete metrics and percentages.',
                'Add specific project outcomes and business impact.',
                'Adopt a strict STAR format (Situation, Task, Action, Result) for all achievements.'
            ],
            'breakdown': [
                {
                    'signal': 'Skills gap analysis',
                    'finding': 'Key technical terms from the job description are not prominently featured in your resume.',
                    'why_it_matters': 'ATS systems prioritize exact keyword matches for technical roles. Missing critical skills can result in profile rejection.',
                    'fix': 'Incorporate evidence of these skills in your experience descriptions, even if applied in different contexts.'
                }
            ],
            'resume_signals': [],
            'rewrite_strategy': [
                'Prioritize only skills and accomplishments already present in the uploaded resume.',
                'Focus on strengthening evidence for existing skills rather than adding new ones.',
                'Quantify results wherever possible (%, dollars, time saved, etc.)'
            ]
        }
        return {
            "analysis": insight,
            "recruiter_verdict": None,
            "weighted_score": None,
            "confidence_score": None,
            "seniority_inference": None,
            "pipeline_warnings": [f"Local LLM/RAG fallback used: {str(e)}"]
        }

def _is_garbled(text: str) -> bool:
    """Detect if text contains garbled/malformed patterns (excessive single letters)"""
    if not text or len(text) < 3:
        return False

    if _looks_like_spaced_caps_name(text):
        return False
    
    # Check for pattern of single letters separated by spaces at start of text
    words = text.split()[:20]
    if len(words) < 3:
        return False
    
    # Count single-letter words in the first part (not including separators)
    single_letter_count = 0
    separator_count = 0
    word_count = 0
    
    for w in words:
        # Skip common separators
        if w in ['·', '-', '|', '/', '\\']:
            separator_count += 1
            continue
        word_count += 1
        if len(w) == 1:
            single_letter_count += 1
    
    if word_count < 3:
        return False
    
    single_letter_ratio = single_letter_count / word_count
    
    # If more than 40% of actual words are single letters, likely garbled
    if single_letter_ratio > 0.40:
        return True
    
    # Special case: pure contact lines (email, phone, etc) are NOT garbled
    # even if they have separators
    lowered = text.lower()
    if ("@" in text and len(text) < 120) or ("linkedin" in lowered or "github" in lowered):
        # This looks like a contact-only line, not garbled
        return False
    
    return False


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _alpha_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+(?:['.-][A-Za-z]+)?", text or "")


def _looks_like_spaced_caps_name(text: str) -> bool:
    lowered = (text or "").lower()
    if not text or len(text) > 80:
        return False
    if _is_contact_line(text) or any(marker in lowered for marker in ("linkedin", "github", "portfolio", "http", "www")):
        return False
    if re.search(r"\d", text):
        return False
    if re.sub(r"[A-Za-z\s.'-]", "", text):
        return False

    tokens = _alpha_tokens(text)
    if len(tokens) < 5:
        return False

    single_letter_ratio = sum(1 for token in tokens if len(token) == 1) / max(len(tokens), 1)
    uppercase_ratio = sum(1 for char in text if char.isupper()) / max(sum(1 for char in text if char.isalpha()), 1)
    return single_letter_ratio >= 0.6 and uppercase_ratio >= 0.8


def _looks_like_name_line(text: str) -> bool:
    lowered = (text or "").lower()
    if not text or len(text) > 80:
        return False
    if _is_contact_line(text) or _section_heading_key(text):
        return False
    if any(marker in lowered for marker in ("linkedin", "github", "portfolio", "http", "www")):
        return False
    if re.search(r"\d", text):
        return False

    tokens = _alpha_tokens(text)
    letters = "".join(tokens)
    if len(letters) < 3 or len(letters) > 40:
        return False

    if _looks_like_spaced_caps_name(text):
        return True

    if len(tokens) > 8:
        return False

    return 1 <= len(tokens) <= 5


def _clean_resume_lines(text: str) -> List[str]:
    """Clean resume lines, filtering out garbled/malformed text"""
    lines = []
    for raw_line in (text or "").splitlines():
        line = re.sub(r"^[\s\-\*\?\u2022\u25aa\u25cf\uf0b7]+", "", raw_line).strip()
        
        if not line or len(line) < 3:
            continue
        
        # Filter out garbled text
        if _is_garbled(line):
            continue
        
        lines.append(line)
    return lines


def _contains(text: str, term: str) -> bool:
    if not text or not term:
        return False
    escaped = re.escape(term.lower().strip())
    return re.search(rf"(?<![a-z0-9+#/.]){escaped}(?![a-z0-9+#/.])", text.lower()) is not None


def _unique_append(items: List[str], item: str, limit: int = 30) -> None:
    normalized = re.sub(r"\s+", " ", item.lower()).strip()
    if not normalized:
        return
    seen = {re.sub(r"\s+", " ", existing.lower()).strip() for existing in items}
    if normalized not in seen and len(items) < limit:
        items.append(item.strip())


RESUME_SECTION_ALIASES = {
    "summary": {
        "summary",
        "professional summary",
        "profile",
        "technical profile",
        "professional profile",
        "career summary",
        "objective",
    },
    "skills": {
        "skills",
        "technical skills",
        "core skills",
        "technologies",
        "tools",
        "tooling",
        "competencies",
    },
    "experience": {
        "experience",
        "work experience",
        "professional experience",
        "employment",
        "career history",
        "work history",
    },
    "projects": {"projects", "selected projects", "project experience"},
    "education": {"education", "academic background"},
    "certifications": {"certifications", "certification", "licenses", "awards"},
}

RESUME_SECTION_LOOKUP = {
    re.sub(r"[^a-z0-9]+", "", alias): key
    for key, aliases in RESUME_SECTION_ALIASES.items()
    for alias in aliases
}


def _section_heading_key(line: str) -> Optional[str]:
    normalized = re.sub(r"[:\-\s]+$", "", (line or "").lower()).strip()
    squashed = re.sub(r"[^a-z0-9]+", "", normalized)
    if not squashed or len(normalized) > 80 or len(squashed) > 48:
        return None
    return RESUME_SECTION_LOOKUP.get(squashed) or (
        normalized if normalized in RESUME_SECTION_LOOKUP else None
    )


def _is_contact_line(line: str) -> bool:
    lowered = (line or "").lower()
    return bool(
        "@" in line
        or "linkedin.com" in lowered
        or "github.com" in lowered
        or "portfolio" in lowered
        or re.search(r"\+?\d[\d\s().-]{7,}\d", line or "")
    )


def _parse_resume_sections(text: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {"header": []}
    current = "header"

    for raw_line in (text or "").splitlines():
        line = re.sub(r"^[\s\-\*\?\u2022\u25aa\u25cf\uf0b7]+", "", raw_line).strip()
        if not line or len(line) < 2:
            continue

        heading = _section_heading_key(line)
        if heading:
            current = heading
            sections.setdefault(current, [])
            continue

        if _is_garbled(line):
            continue

        sections.setdefault(current, []).append(line)

    return sections


def _dedupe_lines(lines: List[str], limit: Optional[int] = None) -> List[str]:
    deduped: List[str] = []
    for line in lines:
        normalized = re.sub(r"\s+", " ", (line or "").lower()).strip()
        if not normalized or _section_heading_key(line):
            continue
        _unique_append(deduped, line, limit=limit or 500)
        if limit and len(deduped) >= limit:
            break
    return deduped


def _merge_wrapped_lines(lines: List[str]) -> List[str]:
    merged: List[str] = []
    for line in _dedupe_lines(lines):
        stripped = line.strip()
        if not stripped:
            continue
        starts_as_continuation = stripped[0].islower() or stripped.lower().startswith(("and ", "or ", "by "))
        if merged and starts_as_continuation:
            merged[-1] = f"{merged[-1]} {stripped}"
        else:
            merged.append(stripped)
    return merged


def _section_lines(sections: Dict[str, List[str]], key: str, limit: Optional[int] = None) -> List[str]:
    return _dedupe_lines(_merge_wrapped_lines(sections.get(key, [])), limit=limit)


def _summary_lines(sections: Dict[str, List[str]], contact: List[str], name_line: Optional[str] = None) -> List[str]:
    contact_norm = {re.sub(r"\s+", " ", line.lower()).strip() for line in contact}
    if name_line:
        contact_norm.add(_collapse_spaces(name_line).lower())
    source = sections.get("summary") or sections.get("header", [])
    candidates = []

    for line in _merge_wrapped_lines(source):
        normalized = re.sub(r"\s+", " ", line.lower()).strip()
        if normalized in contact_norm or _is_contact_line(line):
            continue
        if _looks_like_name_line(line):
            continue
        if len(line) < 25:
            continue
        _unique_append(candidates, line, limit=2)

    return candidates


def _fallback_skill_lines(analysis: Dict[str, Any]) -> List[str]:
    generic_terms = {
        "applications",
        "background",
        "basic",
        "code",
        "competencies",
        "development",
        "engineer",
        "engineering",
        "familiarity",
        "junior",
        "learning",
        "real",
        "reliable",
        "software",
        "strong",
        "structured",
        "supporting",
        "understanding",
    }
    display_overrides = {
        "api": "API",
        "apis": "APIs",
        "aws": "AWS",
        "ci/cd": "CI/CD",
        "css": "CSS",
        "gcp": "GCP",
        "git": "Git",
        "html": "HTML",
        "javascript": "JavaScript",
        "mongodb": "MongoDB",
        "mysql": "MySQL",
        "next.js": "Next.js",
        "node": "Node.js",
        "node.js": "Node.js",
        "postgresql": "PostgreSQL",
        "python": "Python",
        "react": "React",
        "redis": "Redis",
        "rest": "REST",
        "sql": "SQL",
        "typescript": "TypeScript",
    }
    terms = []
    for item in analysis.get("ats_signals", {}).get("matched_terms", []):
        term = item.get("term") if isinstance(item, dict) else str(item)
        normalized = (term or "").lower().strip()
        if not normalized or normalized in generic_terms:
            continue
        label = display_overrides.get(normalized, term.title() if len(term) > 3 else term.upper())
        _unique_append(terms, label, limit=12)

    return ["Role-aligned: " + ", ".join(terms)] if terms else []


def _should_bullet_line(line: str) -> bool:
    lowered = (line or "").lower().strip()
    first_word = re.match(r"[a-z]+", lowered)
    if not first_word:
        return False
    action_starters = {
        "architected",
        "automated",
        "built",
        "collaborated",
        "containerized",
        "created",
        "cut",
        "delivered",
        "designed",
        "developed",
        "enabled",
        "implemented",
        "improved",
        "integrated",
        "led",
        "maintained",
        "migrated",
        "optimized",
        "reduced",
        "secured",
        "supported",
        "tested",
        "triaged",
        "wired",
        "wrote",
    }
    return first_word.group(0) in action_starters


def _append_resume_section(draft_parts: List[str], title: str, lines: List[str], mixed_bullets: bool = False) -> None:
    if not lines:
        return
    draft_parts.append(title)
    for line in lines:
        prefix = "- " if mixed_bullets and _should_bullet_line(line) else ""
        draft_parts.append(f"{prefix}{line}")
    draft_parts.append("")


def _header_name_line(sections: Dict[str, List[str]], lines: List[str]) -> Optional[str]:
    candidates = sections.get("header", [])[:8] or lines[:8]
    for line in candidates:
        if _looks_like_name_line(line):
            return line.strip() if _looks_like_spaced_caps_name(line) else _collapse_spaces(line)
    return None


def _contact_block(lines: List[str], name_line: Optional[str] = None) -> List[str]:
    """Extract contact information without duplicating the resume name."""
    contact = []
    name_norm = _collapse_spaces(name_line).lower() if name_line else None
    for line in lines[:15]:
        if _section_heading_key(line):
            continue

        normalized = _collapse_spaces(line).lower()
        if name_norm and normalized == name_norm:
            continue
        
        is_contact = _is_contact_line(line)
        
        if is_contact:
            _unique_append(contact, line, limit=6)
    
    return contact[:6]


def _structured_resume_sections(
    profile_title: str,
    summary: List[str],
    skills_title: str,
    skills: List[str],
    experience: List[str],
    projects: List[str],
    education: List[str],
    certifications: List[str],
) -> List[Dict[str, Any]]:
    sections = [
        {"title": profile_title.upper(), "lines": summary, "layout": "paragraphs"},
        {"title": skills_title.upper(), "lines": skills, "layout": "highlights"},
        {"title": "PROFESSIONAL EXPERIENCE", "lines": experience, "layout": "bullets"},
        {"title": "PROJECTS", "lines": projects, "layout": "bullets"},
        {"title": "EDUCATION", "lines": education, "layout": "lines"},
        {"title": "CERTIFICATIONS", "lines": certifications, "layout": "lines"},
    ]
    return [section for section in sections if section["lines"]]


def _link_text(item: Dict[str, Any]) -> str:
    label = str(item.get("label") or item.get("url_label") or "Link").strip()
    url = str(item.get("url") or "").strip()
    return f"{label}: {url}" if url else ""


def _structured_resume_to_draft(structured_resume: Dict[str, Any]) -> str:
    parts: List[str] = []

    name = str(structured_resume.get("name") or "").strip()
    contact_items = [str(item).strip() for item in structured_resume.get("contact_items", []) if str(item).strip()]
    # Don't include profile links in draft text - they're rendered separately in header
    # to avoid showing explicit URLs and duplication

    if name:
        parts.append(name)
    if contact_items:
        parts.append(" | ".join(contact_items))
    if parts:
        parts.append("")

    for section in structured_resume.get("sections", []):
        title = str(section.get("title") or "").strip()
        if title:
            parts.append(title)

        kind = section.get("kind")

        if kind == "summary":
            for paragraph in section.get("paragraphs", []):
                text = str(paragraph).strip()
                if text:
                    parts.append(text)
        elif kind == "skills":
            for category in section.get("categories", []):
                label = str(category.get("label") or "").strip()
                items = [str(item).strip() for item in category.get("items", []) if str(item).strip()]
                if label and items:
                    parts.append(f"{label}: {', '.join(items)}")
        elif kind in {"experience", "projects"}:
            for item in section.get("items", []):
                if kind == "experience":
                    header = " | ".join(
                        piece for piece in [
                            str(item.get("organization") or "").strip(),
                            str(item.get("title") or "").strip(),
                        ]
                        if piece
                    )
                    meta = " | ".join(
                        piece for piece in [
                            str(item.get("location") or "").strip(),
                            str(item.get("date_range") or "").strip(),
                        ]
                        if piece
                    )
                else:
                    header = " | ".join(
                        piece for piece in [
                            str(item.get("name") or "").strip(),
                            str(item.get("subtitle") or "").strip(),
                        ]
                        if piece
                    )
                    tech_stack = ", ".join(str(tech).strip() for tech in item.get("tech_stack", []) if str(tech).strip())
                    meta = f"Tech: {tech_stack}" if tech_stack else ""

                if header:
                    parts.append(header)
                if meta:
                    parts.append(meta)

                item_link = _link_text(item)
                if item_link:
                    parts.append(item_link)

                for bullet in item.get("bullets", []):
                    text = str(bullet).strip()
                    if text:
                        parts.append(f"- {text}")
                parts.append("")
            if parts and not parts[-1]:
                parts.pop()
        elif kind == "education":
            for item in section.get("items", []):
                header = " | ".join(
                    piece for piece in [
                        str(item.get("institution") or "").strip(),
                        str(item.get("date_range") or "").strip(),
                    ]
                    if piece
                )
                detail = " | ".join(
                    piece for piece in [
                        str(item.get("degree") or "").strip(),
                        str(item.get("details") or "").strip(),
                    ]
                    if piece
                )
                if header:
                    parts.append(header)
                if detail:
                    parts.append(detail)
        elif kind == "certifications":
            for item in section.get("items", []):
                header = " | ".join(
                    piece for piece in [
                        str(item.get("name") or "").strip(),
                        str(item.get("issuer") or "").strip(),
                        str(item.get("date") or "").strip(),
                    ]
                    if piece
                )
                if header:
                    parts.append(header)
                item_link = _link_text(item)
                if item_link:
                    parts.append(item_link)
        else:
            for line in section.get("lines", []):
                text = str(line).strip()
                if text:
                    parts.append(text)

        parts.append("")

    while parts and not parts[-1]:
        parts.pop()

    return "\n".join(parts).strip()


def _matched_terms(analysis: Dict[str, Any]) -> List[str]:
    signals = analysis.get("ats_signals", {})
    raw_terms = signals.get("critical_matched_terms") or signals.get("matched_terms") or []
    terms = []
    for item in raw_terms:
        term = item.get("term") if isinstance(item, dict) else str(item)
        if term:
            _unique_append(terms, term, limit=28)
    return terms


def _evidence_lines(resume_text: str, analysis: Dict[str, Any], terms: List[str]) -> List[str]:
    lines = _clean_resume_lines(resume_text)
    evidence = []

    for item in analysis.get("ats_signals", {}).get("exact_evidence", []):
        snippet = item.get("snippet") if isinstance(item, dict) else str(item)
        if snippet and not _is_garbled(snippet):
            _unique_append(evidence, snippet, limit=16)

    for item in analysis.get("ats_signals", {}).get("retrieved_evidence", []):
        snippet = item.get("snippet") if isinstance(item, dict) else str(item)
        if snippet and not _is_garbled(snippet):
            _unique_append(evidence, snippet, limit=16)

    for line in lines:
        if any(_contains(line, term) for term in terms):
            _unique_append(evidence, line, limit=16)

    return evidence[:16]


def _supporting_lines(resume_text: str, chosen_lines: List[str], limit: int = 24) -> List[str]:
    used = {re.sub(r"\s+", " ", line.lower()).strip() for line in chosen_lines}
    supporting = []
    for line in _clean_resume_lines(resume_text):
        normalized = re.sub(r"\s+", " ", line.lower()).strip()
        if normalized in used:
            continue
        if len(line) < 6:
            continue
        if re.match(r"^(summary|skills|experience|education|projects|certifications)[:\s]*$", line, re.I):
            continue
        _unique_append(supporting, line, limit=limit)
    return supporting[:limit]


def _get_resume_templates(industry: str = "default") -> Dict[str, Any]:
    """Advanced resume templates with industry-specific optimizations"""

    templates = {
        "tech": {
            "name": "Technical Excellence Format",
            "structure": ["Contact & Summary", "Core Technical Skills", "Professional Experience", "Key Projects", "Education & Certifications"],
            "formatting_tips": [
                "Use reverse chronological order",
                "Quantify technical achievements with metrics",
                "Highlight specific technologies and tools",
                "Include GitHub/portfolio links",
                "Focus on problem-solving and innovation"
            ],
            "content_suggestions": [
                "Lead with technical architecture decisions",
                "Emphasize system design and scalability",
                "Showcase debugging and optimization skills",
                "Highlight cross-functional collaboration",
                "Demonstrate continuous learning"
            ]
        },
        "finance": {
            "name": "Financial Impact Format",
            "structure": ["Professional Summary", "Core Competencies", "Professional Experience", "Financial Achievements", "Education & Licenses"],
            "formatting_tips": [
                "Use quantifiable financial metrics",
                "Highlight risk management experience",
                "Show regulatory compliance knowledge",
                "Include relevant certifications prominently",
                "Focus on stakeholder impact"
            ],
            "content_suggestions": [
                "Quantify financial impact (revenue, cost savings, etc.)",
                "Showcase analytical and modeling skills",
                "Highlight regulatory and compliance experience",
                "Demonstrate strategic financial planning",
                "Emphasize stakeholder communication"
            ]
        },
        "healthcare": {
            "name": "Patient-Centered Care Format",
            "structure": ["Professional Summary", "Clinical Skills & Certifications", "Professional Experience", "Patient Care Achievements", "Education & Training"],
            "formatting_tips": [
                "Highlight patient outcomes and care quality",
                "Showcase clinical certifications",
                "Use healthcare-specific terminology appropriately",
                "Include continuing education",
                "Focus on compliance and safety"
            ],
            "content_suggestions": [
                "Emphasize patient care outcomes",
                "Showcase clinical decision-making",
                "Highlight quality improvement initiatives",
                "Demonstrate interdisciplinary collaboration",
                "Include relevant certifications and training"
            ]
        },
        "default": {
            "name": "Professional Achievement Format",
            "structure": ["Professional Summary", "Core Skills", "Professional Experience", "Key Achievements", "Education"],
            "formatting_tips": [
                "Use action verbs to start bullet points",
                "Quantify achievements with metrics",
                "Tailor content to job description",
                "Keep format clean and ATS-friendly",
                "Highlight relevant experience prominently"
            ],
            "content_suggestions": [
                "Start bullets with strong action verbs",
                "Include specific, measurable results",
                "Show career progression",
                "Highlight transferable skills",
                "Customize for target role"
            ]
        }
    }

    return templates.get(industry, templates["default"])

def _generate_content_suggestions(resume_text: str, analysis: Dict[str, Any], industry: str) -> List[Dict[str, Any]]:
    """AI-powered content suggestions for resume optimization"""

    missing_skills = [item.get('term', '') for item in analysis.get('ats_signals', {}).get('critical_missing_terms', [])[:10]]
    matched_skills = [item.get('term', '') for item in analysis.get('ats_signals', {}).get('matched_terms', [])[:10]]

    suggestions = []

    # Skill gap suggestions
    if missing_skills:
        suggestions.append({
            "type": "skill_gaps",
            "priority": "high",
            "title": "Address Critical Skill Gaps",
            "suggestions": [
                f"Incorporate evidence of {skill} experience from past roles or projects" for skill in missing_skills[:5]
            ],
            "implementation": "Add specific examples where these skills were applied, even if in different contexts"
        })

    # Content enhancement suggestions
    impact_examples = analysis.get('ats_signals', {}).get('impact_examples', [])
    if len(impact_examples) < 3:
        suggestions.append({
            "type": "quantification",
            "priority": "high",
            "title": "Add Quantifiable Achievements",
            "suggestions": [
                "Replace generic statements with specific metrics (%, $, time saved, etc.)",
                "Include concrete outcomes for each major responsibility",
                "Show scope and scale of impact on projects/teams",
                "Highlight efficiency improvements and cost savings"
            ],
            "implementation": "Review each bullet point and add numbers wherever possible"
        })

    # Industry-specific suggestions
    if industry == "tech":
        suggestions.append({
            "type": "technical_depth",
            "priority": "medium",
            "title": "Demonstrate Technical Depth",
            "suggestions": [
                "Include specific technologies, frameworks, and tools used",
                "Mention architecture decisions and technical challenges solved",
                "Showcase code optimization, performance improvements",
                "Highlight scalability and system design considerations"
            ],
            "implementation": "Add technical details to project descriptions"
        })
    elif industry == "finance":
        suggestions.append({
            "type": "financial_impact",
            "priority": "medium",
            "title": "Showcase Financial Impact",
            "suggestions": [
                "Quantify revenue impact, cost savings, and financial metrics",
                "Highlight risk management and compliance achievements",
                "Showcase analytical modeling and forecasting skills",
                "Demonstrate strategic financial planning contributions"
            ],
            "implementation": "Include specific financial figures and business outcomes"
        })

    # ATS optimization suggestions
    parseability_risks = analysis.get('ats_signals', {}).get('parseability_risks', [])
    if parseability_risks:
        suggestions.append({
            "type": "ats_optimization",
            "priority": "high",
            "title": "Improve ATS Compatibility",
            "suggestions": parseability_risks[:3],
            "implementation": "Ensure standard section headers and clean formatting"
        })

    return suggestions


def _parse_experience_items(lines: List[str]) -> List[Dict[str, Any]]:
    """Parse experience lines into structured items with organization, title, location, date, bullets"""
    items = []
    current_item = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if this is a header line (contains pipes separating fields)
        if " | " in line and not line.startswith("-"):
            if current_item and current_item.get("bullets"):
                items.append(current_item)
            
            # Parse header like "Company Name | Job Title" or "Company | Title | Location | Date"
            parts = [p.strip() for p in line.split("|")]
            current_item = {
                "organization": parts[0] if len(parts) > 0 else "",
                "title": parts[1] if len(parts) > 1 else "",
                "location": parts[2] if len(parts) > 2 else "",
                "date_range": parts[3] if len(parts) > 3 else "",
                "bullets": []
            }
        elif line.startswith("-"):
            if current_item is None:
                current_item = {"organization": "", "title": "", "location": "", "date_range": "", "bullets": []}
            current_item["bullets"].append(line[2:].strip())
        elif current_item is None and line:
            # This might be a company or organization name
            current_item = {"organization": line, "title": "", "location": "", "date_range": "", "bullets": []}
    
    if current_item and (current_item.get("organization") or current_item.get("bullets")):
        items.append(current_item)
    
    return items


def _parse_project_items(lines: List[str]) -> List[Dict[str, Any]]:
    """Parse project lines into structured items with name, subtitle, tech stack, bullets"""
    items = []
    current_item = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if this is a header line (contains pipes or technical keyword)
        if " | " in line and not line.startswith("-") and not line.startswith("Tech"):
            if current_item and current_item.get("bullets"):
                items.append(current_item)
            
            parts = [p.strip() for p in line.split("|")]
            current_item = {
                "name": parts[0] if len(parts) > 0 else "",
                "subtitle": parts[1] if len(parts) > 1 else "",
                "tech_stack": [],
                "bullets": []
            }
        elif line.startswith("Tech:") or line.startswith("Technology:"):
            if current_item is None:
                current_item = {"name": "", "subtitle": "", "tech_stack": [], "bullets": []}
            tech_part = line.split(":", 1)[1].strip()
            current_item["tech_stack"] = [t.strip() for t in tech_part.split(",")]
        elif line.startswith("-"):
            if current_item is None:
                current_item = {"name": "", "subtitle": "", "tech_stack": [], "bullets": []}
            current_item["bullets"].append(line[2:].strip())
        elif current_item is None and line and not line.startswith("http"):
            # This might be a project name
            current_item = {"name": line, "subtitle": "", "tech_stack": [], "bullets": []}
    
    if current_item and (current_item.get("name") or current_item.get("bullets")):
        items.append(current_item)
    
    return items


def _parse_education_items(lines: List[str]) -> List[Dict[str, Any]]:
    """Parse education lines into structured items"""
    items = []
    current_item = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if " | " in line and not line.startswith("-"):
            if current_item:
                items.append(current_item)
            
            parts = [p.strip() for p in line.split("|")]
            current_item = {
                "institution": parts[0] if len(parts) > 0 else "",
                "degree": parts[1] if len(parts) > 1 else "",
                "date_range": parts[2] if len(parts) > 2 else "",
                "details": parts[3] if len(parts) > 3 else ""
            }
            items.append(current_item)
            current_item = None
        elif line and not line.startswith("-"):
            if current_item is None:
                current_item = {"institution": line, "degree": "", "date_range": "", "details": ""}
            else:
                current_item["degree"] = line
    
    if current_item:
        items.append(current_item)
    
    return items


def _parse_certification_items(lines: List[str]) -> List[Dict[str, Any]]:
    """Parse certification lines into structured items"""
    items = []
    current_item = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if " | " in line and not line.startswith("-") and not line.startswith("http"):
            if current_item:
                items.append(current_item)
            
            parts = [p.strip() for p in line.split("|")]
            current_item = {
                "name": parts[0] if len(parts) > 0 else "",
                "issuer": parts[1] if len(parts) > 1 else "",
                "date": parts[2] if len(parts) > 2 else "",
            }
        elif line.startswith("http"):
            if current_item is None:
                current_item = {"name": "", "issuer": "", "date": "", "url": line, "url_label": "Credential"}
            else:
                current_item["url"] = line
                current_item["url_label"] = "Credential"
            items.append(current_item)
            current_item = None
        elif line and not line.startswith("-"):
            if current_item is None:
                current_item = {"name": line, "issuer": "", "date": ""}
            else:
                current_item["issuer"] = line
    
    if current_item:
        items.append(current_item)
    
    return items


def generate_optimized_resume(job_description: str, resume_text: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds a fact-preserving resume draft from the uploaded resume.
    It gates generation when the match is too weak, and it does not invent
    employers, titles, metrics, dates, tools, education, or accomplishments.
    """
    del job_description
    policy = analysis.get("generation_policy", {})
    if not policy.get("can_generate"):
        return {
            "can_generate": False,
            "reason": policy.get(
                "reason",
                "The match is too weak to rewrite safely without inventing experience."
            ),
            "draft": "",
            "format": "text",
            "integrity_rules": [
                "No generated draft was created because the evidence gate failed.",
                "Add real, verifiable experience first if the target role is materially different."
            ],
            "blocked_missing_terms": [
                item.get("term") for item in analysis.get("ats_signals", {}).get("critical_missing_terms", [])[:12]
                if isinstance(item, dict) and item.get("term")
            ],
        }

    # Detect industry and get appropriate template
    industry = analysis.get('industry_analysis', {}).get('detected_industry', 'default')
    template = _get_resume_templates(industry)
    content_suggestions = _generate_content_suggestions(resume_text, analysis, industry)

    sections = _parse_resume_sections(resume_text)
    lines = _clean_resume_lines(resume_text)
    name_line = _header_name_line(sections, lines)
    contact = _contact_block(lines, name_line)
    profile_override = get_profile_override(resume_text, name_line=name_line, contact_lines=contact)
    summary = _summary_lines(sections, contact, name_line)
    skills = _section_lines(sections, "skills", limit=8) or _fallback_skill_lines(analysis)
    experience = _section_lines(sections, "experience", limit=28)
    projects = _section_lines(sections, "projects", limit=32)
    education = _section_lines(sections, "education", limit=8)
    certifications = _section_lines(sections, "certifications", limit=8)

    if not any((summary, skills, experience, projects, education, certifications)):
        terms = _matched_terms(analysis)
        fallback_evidence = _evidence_lines(resume_text, analysis, terms)
        summary = _supporting_lines(resume_text, contact + fallback_evidence, limit=2)
        experience = fallback_evidence[:12]

    draft_parts = []
    
    # Header section
    if name_line:
        draft_parts.append(name_line)

    if name_line or contact:
        for contact_line in contact:
            if contact_line.strip():
                draft_parts.append(contact_line)
        draft_parts.append("")

    profile_title = "TECHNICAL PROFILE" if industry == "tech" else \
                   "PROFESSIONAL PROFILE" if industry == "finance" else \
                   "PROFESSIONAL SUMMARY"
    skills_title = "TECHNICAL SKILLS" if industry == "tech" else \
                   "CORE COMPETENCIES" if industry == "finance" else \
                   "PROFESSIONAL SKILLS"

    _append_resume_section(draft_parts, profile_title.upper(), summary)
    _append_resume_section(draft_parts, skills_title.upper(), skills)
    _append_resume_section(draft_parts, "PROFESSIONAL EXPERIENCE", experience, mixed_bullets=True)
    _append_resume_section(draft_parts, "PROJECTS", projects, mixed_bullets=True)
    _append_resume_section(draft_parts, "EDUCATION", education)
    _append_resume_section(draft_parts, "CERTIFICATIONS", certifications)

    while draft_parts and not draft_parts[-1]:
        draft_parts.pop()

    content_section_count = sum(
        1
        for section_lines in (summary, skills, experience, projects, education, certifications)
        if section_lines
    )
    structured_sections = _structured_resume_sections(
        profile_title,
        summary,
        skills_title,
        skills,
        experience,
        projects,
        education,
        certifications,
    )
    # Extract profile_links from override if available, otherwise use empty list
    profile_links = profile_override.get("profile_links", []) if profile_override else []
    
    # Convert simple structured_sections into rich items with proper structure
    sections = []
    for section_data in structured_sections:
        title = section_data.get("title", "")
        lines = section_data.get("lines", [])
        layout = section_data.get("layout", "lines")
        
        # Determine the section kind based on title or layout
        kind = "lines"
        if "PROFILE" in title or "SUMMARY" in title:
            kind = "summary"
        elif "SKILL" in title or "COMPETENC" in title:
            kind = "skills"
        elif "EXPERIENCE" in title:
            kind = "experience"
        elif "PROJECT" in title:
            kind = "projects"
        elif "EDUCATION" in title:
            kind = "education"
        elif "CERTIFICATION" in title:
            kind = "certifications"
        
        rebuilt_section = {
            "kind": kind,
            "title": title,
        }
        
        if kind == "summary":
            rebuilt_section["paragraphs"] = lines
        elif kind == "skills":
            # Try to preserve skill categories from split lines
            categories = []
            current_category = None
            for line in lines:
                if ":" in line and not line.startswith("-"):
                    if current_category:
                        categories.append(current_category)
                    label, items_str = line.split(":", 1)
                    items = [i.strip() for i in items_str.split(",")]
                    current_category = {"label": label.strip(), "items": items}
                elif current_category and line.startswith("-"):
                    current_category["items"].append(line[2:].strip())
                elif not current_category:
                    current_category = {"label": "Skills", "items": [line]}
            if current_category:
                categories.append(current_category)
            if categories:
                rebuilt_section["categories"] = categories
            else:
                # Fallback: group skills as single category
                rebuilt_section["categories"] = [{"label": "Technical Skills", "items": lines}]
        elif kind == "experience":
            items = _parse_experience_items(lines)
            if items:
                rebuilt_section["items"] = items
        elif kind == "projects":
            items = _parse_project_items(lines)
            if items:
                rebuilt_section["items"] = items
        elif kind == "education":
            items = _parse_education_items(lines)
            if items:
                rebuilt_section["items"] = items
        elif kind == "certifications":
            items = _parse_certification_items(lines)
            if items:
                rebuilt_section["items"] = items
        else:
            rebuilt_section["lines"] = lines
        
        sections.append(rebuilt_section)
    
    structured_resume = {
        "template_key": "ats_single_column_advanced",
        "template_name": template["name"],
        "ats_safe": True,
        "name": name_line,
        "contact_items": contact,
        "profile_links": profile_links,
        "sections": sections,
    }
    # Generate draft from structured resume to ensure proper formatting
    draft_text = _structured_resume_to_draft(structured_resume)
    section_count = len(structured_resume.get("sections", [])) if isinstance(structured_resume, dict) else content_section_count
    skills_highlighted = len(skills)
    evidence_lines_used = len(experience) + len(projects)
    template_name = structured_resume.get("template_name") if isinstance(structured_resume, dict) else None

    return {
        "can_generate": True,
        "reason": policy.get("reason", "Advanced analysis confirms sufficient evidence for optimized resume generation."),
        "draft": draft_text,
        "format": "text",
        "download_filename": f"optimized_resume_{industry}_advanced.txt",
        "download_pdf_filename": f"optimized_resume_{industry}_advanced.pdf",
        "template_used": template_name or template["name"],
        "structured_resume": structured_resume,
        "industry_specific": {
            "industry": industry,
            "template_name": template_name or template["name"],
            "formatting_tips": template["formatting_tips"],
            "content_suggestions": template["content_suggestions"]
        },
        "optimization_suggestions": content_suggestions,
        "integrity_rules": [
            "Advanced draft incorporates industry best practices and enhanced formatting",
            "Content remains fact-preserving, using only uploaded resume text and evidenced skills",
            "No new employers, titles, dates, metrics, or accomplishments are fabricated",
            "Resume optimized for target industry standards and ATS compatibility"
        ],
        "blocked_missing_terms": [
            item.get("term") for item in analysis.get("ats_signals", {}).get("critical_missing_terms", [])[:12]
            if isinstance(item, dict) and item.get("term")
        ],
        "enhancement_summary": {
            "skills_highlighted": skills_highlighted,
            "evidence_lines_used": evidence_lines_used,
            "content_sections": section_count,
            "industry_optimizations": len(template["content_suggestions"])
        }
    }
