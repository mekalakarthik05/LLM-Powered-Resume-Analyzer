import logging
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import ollama
except ImportError:  # pragma: no cover - dependency is optional at runtime
    ollama = None

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "alex-recruiter")
EMBED_MODEL = "nomic-embed-text"  # Use dedicated embedding model for semantic similarity

# Create reusable client for scorer embeddings (if available)
OLLAMA_CLIENT = None
try:
    if ollama is not None:
        OLLAMA_CLIENT = ollama.Client(host=OLLAMA_HOST)
except Exception as e:
    logger.warning(f"Failed to initialize OLLAMA client for scorer: {e}")

MIN_GENERATION_SCORE = 35
MIN_CRITICAL_COVERAGE = 0.22
MIN_MATCHED_CRITICAL_TERMS = 2

# Enhanced thresholds for aggressive analysis
AGGRESSIVE_THRESHOLDS = {
    'excellent': 85,
    'good': 70,
    'moderate': 55,
    'poor': 40,
    'critical': 25
}

# Industry-specific scoring weights and benchmarks
INDUSTRY_BENCHMARKS = {
    'tech': {
        'semantic_weight': 0.25,
        'technical_weight': 0.35,
        'experience_weight': 0.25,
        'soft_skills_weight': 0.15,
        'critical_tech_terms': ['python', 'javascript', 'react', 'aws', 'docker', 'kubernetes', 'sql', 'api', 'git'],
        'seniority_multipliers': {'entry': 0.7, 'mid': 1.0, 'senior': 1.2, 'lead': 1.3}
    },
    'finance': {
        'semantic_weight': 0.20,
        'technical_weight': 0.40,
        'experience_weight': 0.30,
        'soft_skills_weight': 0.10,
        'critical_tech_terms': ['excel', 'sql', 'python', 'tableau', 'powerbi', 'bloomberg', 'financial modeling'],
        'seniority_multipliers': {'entry': 0.6, 'mid': 1.0, 'senior': 1.3, 'lead': 1.4}
    },
    'healthcare': {
        'semantic_weight': 0.30,
        'technical_weight': 0.25,
        'experience_weight': 0.35,
        'soft_skills_weight': 0.10,
        'critical_tech_terms': ['ehr', 'hipaa', 'patient care', 'clinical', 'medical terminology'],
        'seniority_multipliers': {'entry': 0.8, 'mid': 1.0, 'senior': 1.1, 'lead': 1.2}
    },
    'default': {
        'semantic_weight': 0.20,
        'technical_weight': 0.30,
        'experience_weight': 0.30,
        'soft_skills_weight': 0.20,
        'critical_tech_terms': [],
        'seniority_multipliers': {'entry': 0.7, 'mid': 1.0, 'senior': 1.2, 'lead': 1.3}
    }
}

STOP_WORDS = set(ENGLISH_STOP_WORDS).union(
    {
        "role",
        "candidate",
        "responsibilities",
        "requirements",
        "required",
        "preferred",
        "job",
        "description",
        "team",
        "work",
        "working",
        "experience",
        "years",
        "ability",
        "skills",
    }
)

TECH_ONTOLOGY = {
    "a/b testing",
    "accessibility",
    "adobe analytics",
    "agile",
    "airflow",
    "algorithm",
    "analytics",
    "android",
    "angular",
    "ansible",
    "api",
    "apis",
    "artificial intelligence",
    "aws",
    "azure",
    "bigquery",
    "c",
    "c#",
    "c++",
    "ci/cd",
    "cloud",
    "computer vision",
    "css",
    "data analysis",
    "data engineering",
    "data pipelines",
    "data science",
    "deep learning",
    "devops",
    "django",
    "docker",
    "elasticsearch",
    "etl",
    "express",
    "fastapi",
    "figma",
    "firebase",
    "flask",
    "gcp",
    "git",
    "github",
    "go",
    "golang",
    "graphql",
    "hadoop",
    "html",
    "ios",
    "java",
    "javascript",
    "jira",
    "kafka",
    "kubernetes",
    "langchain",
    "linux",
    "machine learning",
    "microservices",
    "mongodb",
    "mysql",
    "next.js",
    "nlp",
    "node",
    "node.js",
    "nosql",
    "numpy",
    "openai",
    "pandas",
    "postgres",
    "postgresql",
    "power bi",
    "pytorch",
    "python",
    "r",
    "react",
    "redis",
    "rest",
    "rest api",
    "ruby",
    "salesforce",
    "scala",
    "scikit-learn",
    "scrum",
    "snowflake",
    "spark",
    "spring",
    "sql",
    "tableau",
    "tensorflow",
    "terraform",
    "typescript",
    "ux",
    "vue",
}

ACTION_VERBS = {
    "achieved",
    "architected",
    "automated",
    "built",
    "collaborated",
    "created",
    "delivered",
    "designed",
    "developed",
    "drove",
    "enabled",
    "implemented",
    "improved",
    "increased",
    "launched",
    "led",
    "managed",
    "migrated",
    "optimized",
    "owned",
    "reduced",
    "shipped",
    "spearheaded",
    "streamlined",
    "tested",
}

SECTION_ALIASES = {
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

CRITICAL_CUES = (
    "must",
    "required",
    "requirement",
    "minimum",
    "need",
    "needs",
    "proficient",
    "expertise",
    "strong",
    "hands-on",
    "preferred",
    "familiarity",
    "knowledge",
    "essential",
    "mandatory",
    "critical",
    "key",
    "core",
    "fundamental",
    "advanced",
    "senior",
    "lead",
    "principal"
)

# Enhanced critical patterns with context
CRITICAL_PATTERNS = {
    'experience_years': re.compile(r'(\d+\+?)\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience|exp)', re.I),
    'education_level': re.compile(r'(?:bachelor|master|phd|mba|ms|ma|ba|bs)\s+(?:degree|in)', re.I),
    'certification_required': re.compile(r'(?:certified|certification|license)\s+(?:in|for)', re.I),
    'security_clearance': re.compile(r'(?:clearance|classified|secret|top secret)', re.I),
    'remote_work': re.compile(r'(?:remote|hybrid|onsite|wfh)', re.I)
}

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9.+#/-]*")
METRIC_RE = re.compile(r"(\b\d+(\.\d+)?\s?%|\$\s?\d+|\b\d+(\.\d+)?x\b|\b\d+\+|\b\d{2,}\b)")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _tokenize(text: str) -> List[str]:
    tokens = TOKEN_RE.findall(_normalize(text))
    return [token for token in tokens if token not in STOP_WORDS and len(token) > 1]


def _sentences(text: str) -> List[str]:
    chunks = re.split(r"[\n\r]+|(?<=[.!?])\s+", text or "")
    return [chunk.strip(" \t-*") for chunk in chunks if len(chunk.strip()) > 2]


def _lines(text: str) -> List[str]:
    return [re.sub(r"^[\s\-\*\u2022]+", "", line).strip() for line in (text or "").splitlines() if line.strip()]


def _term_boundary(term: str) -> str:
    escaped = re.escape(_normalize(term))
    return rf"(?<![a-z0-9+#/.]){escaped}(?![a-z0-9+#/.])"


def _contains_term(text: str, term: str) -> bool:
    if not text or not term:
        return False
    return re.search(_term_boundary(term), _normalize(text)) is not None


def _clean_term(term: str) -> str:
    return re.sub(r"\s+", " ", term.lower().strip(" -_/,.")).strip()


def _is_noise_term(term: str) -> bool:
    tokens = term.split()
    if not tokens:
        return True
    if all(token in STOP_WORDS for token in tokens):
        return True
    if len(term) < 3 and term not in {"c", "r", "go"}:
        return True
    if term.isdigit():
        return True
    return False


def _heading_key(line: str) -> Optional[str]:
    normalized = re.sub(r"[:\-\s]+$", "", _normalize(line))
    squashed = re.sub(r"[^a-z0-9]+", "", normalized)
    if len(normalized) > 80 or len(squashed) > 48:
        return None
    for key, aliases in SECTION_ALIASES.items():
        if normalized in aliases:
            return key
        if squashed in {re.sub(r"[^a-z0-9]+", "", alias) for alias in aliases}:
            return key
    return None


def extract_sections(text: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = defaultdict(list)
    current = "header"
    for line in _lines(text):
        heading = _heading_key(line)
        if heading:
            current = heading
            continue
        sections[current].append(line)
    return dict(sections)


def _known_skills(text: str) -> List[str]:
    return sorted({skill for skill in TECH_ONTOLOGY if _contains_term(text, skill)})


def _detect_industry(job_description: str) -> str:
    """Detect industry from job description for tailored analysis"""
    jd_lower = _normalize(job_description)

    industry_keywords = {
        'tech': ['software', 'developer', 'engineer', 'programming', 'code', 'tech', 'it', 'computer', 'data', 'ai', 'ml', 'cloud', 'web', 'mobile', 'api'],
        'finance': ['finance', 'financial', 'banking', 'investment', 'accounting', 'analyst', 'trading', 'portfolio', 'wealth', 'credit', 'loan'],
        'healthcare': ['healthcare', 'medical', 'patient', 'clinical', 'hospital', 'nursing', 'pharma', 'biotech', 'health', 'care'],
        'marketing': ['marketing', 'brand', 'advertising', 'campaign', 'social media', 'content', 'seo', 'growth', 'customer'],
        'sales': ['sales', 'business development', 'account', 'territory', 'revenue', 'pipeline', 'closing', 'negotiation']
    }

    scores = {}
    for industry, keywords in industry_keywords.items():
        score = sum(1 for keyword in keywords if keyword in jd_lower)
        scores[industry] = score

    return max(scores.items(), key=lambda x: x[1])[0] if scores else 'default'

def _critical_context_terms(job_description: str, industry: str = None) -> dict:
    """Enhanced critical terms extraction with industry context and patterns"""
    critical = set()
    pattern_matches = {
        'experience_years': [],
        'education_level': [],
        'certification_required': [],
        'security_clearance': [],
        'remote_work': []
    }

    industry_config = INDUSTRY_BENCHMARKS.get(industry or 'default', INDUSTRY_BENCHMARKS['default'])

    for sentence in _sentences(job_description):
        sentence_norm = _normalize(sentence)
        is_critical = any(cue in sentence_norm for cue in CRITICAL_CUES)

        # Extract pattern-based critical requirements
        for pattern_name, pattern in CRITICAL_PATTERNS.items():
            matches = pattern.findall(sentence)
            if matches:
                pattern_matches[pattern_name].extend(matches)
                is_critical = True

        if is_critical:
            critical.update(_known_skills(sentence))
            # Add industry-specific critical terms
            critical.update(industry_config['critical_tech_terms'])

            for token in _tokenize(sentence):
                if len(token) > 3 and token not in STOP_WORDS:
                    critical.add(token)

    return {
        'terms': critical,
        'patterns': pattern_matches,
        'industry': industry or 'default'
    }


def _extract_key_terms(job_description: str, max_terms: int = 90) -> List[Dict[str, Any]]:
    if not job_description:
        return []

    term_scores: Dict[str, float] = {}
    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=list(STOP_WORDS),
            ngram_range=(1, 3),
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9.+#/-]{1,}\b",
            max_features=700,
        )
        matrix = vectorizer.fit_transform([job_description])
        features = vectorizer.get_feature_names_out()
        weights = matrix.toarray()[0]
        for feature, weight in zip(features, weights):
            term = _clean_term(feature)
            if not _is_noise_term(term):
                term_scores[term] = max(float(weight), term_scores.get(term, 0.0))
    except Exception as exc:
        logger.warning("TF-IDF key-term extraction failed: %s", exc)
        for token, count in Counter(_tokenize(job_description)).items():
            if not _is_noise_term(token):
                term_scores[token] = float(count)

    for skill in _known_skills(job_description):
        term_scores[skill] = max(term_scores.get(skill, 0.0), 1.5)

    critical_context = _critical_context_terms(job_description)
    critical_terms = critical_context['terms']
    terms = []
    for term, score in term_scores.items():
        is_skill = term in TECH_ONTOLOGY
        is_critical = term in critical_terms or is_skill
        terms.append(
            {
                "term": term,
                "weight": round(score * (1.8 if is_critical else 1.0), 4),
                "category": "hard_skill" if is_skill else "keyword",
                "critical": is_critical,
            }
        )

    return sorted(terms, key=lambda item: (item["critical"], item["weight"], len(item["term"])), reverse=True)[:max_terms]


def _term_coverage(terms: List[Dict[str, Any]], resume_text: str) -> Dict[str, Any]:
    matched = []
    missing = []
    matched_weight = 0.0
    total_weight = 0.0

    seen = set()
    for item in terms:
        term = item["term"]
        if term in seen:
            continue
        seen.add(term)
        weight = float(item.get("weight", 1.0))
        total_weight += weight
        target = matched if _contains_term(resume_text, term) else missing
        target.append({**item})
        if target is matched:
            matched_weight += weight

    coverage = matched_weight / total_weight if total_weight else 0.0
    critical_terms = [item for item in terms if item.get("critical")]
    critical_total = sum(float(item.get("weight", 1.0)) for item in critical_terms)
    critical_matched = [item for item in critical_terms if _contains_term(resume_text, item["term"])]
    critical_missing = [item for item in critical_terms if not _contains_term(resume_text, item["term"])]
    critical_weight = sum(float(item.get("weight", 1.0)) for item in critical_matched)

    return {
        "coverage": coverage,
        "critical_coverage": critical_weight / critical_total if critical_total else coverage,
        "matched": matched,
        "missing": missing,
        "critical_matched": critical_matched,
        "critical_missing": critical_missing,
    }


def _tfidf_similarity(job_description: str, resume_text: str) -> float:
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=7000, lowercase=True)
        matrix = vectorizer.fit_transform([job_description, resume_text])
        return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    except Exception as exc:
        logger.warning("TF-IDF similarity failure: %s", exc)
        return 0.0


def _char_similarity(job_description: str, resume_text: str) -> float:
    try:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=9000, lowercase=True)
        matrix = vectorizer.fit_transform([job_description, resume_text])
        return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    except Exception as exc:
        logger.warning("Character ngram similarity failure: %s", exc)
        return 0.0


def _semantic_similarity(job_description: str, resume_text: str) -> Optional[float]:
    if ollama is None or OLLAMA_CLIENT is None:
        return None
    try:
        # Use dedicated embedding model for semantic analysis
        jd_emb = OLLAMA_CLIENT.embeddings(model=EMBED_MODEL, prompt=job_description[:4500])["embedding"]
        resume_emb = OLLAMA_CLIENT.embeddings(model=EMBED_MODEL, prompt=resume_text[:4500])["embedding"]
        jd_vector = np.array(jd_emb)
        resume_vector = np.array(resume_emb)
        score = np.dot(jd_vector, resume_vector) / (np.linalg.norm(jd_vector) * np.linalg.norm(resume_vector) + 1e-10)
        return float(max(0.0, min(score, 1.0)))
    except Exception as exc:
        logger.warning("Semantic embedding failure: %s", exc)
        return None


def _resume_chunks(text: str, sections: Dict[str, List[str]]) -> List[Dict[str, str]]:
    chunks: List[Dict[str, str]] = []
    for section, section_lines in sections.items():
        if not section_lines:
            continue
        for line in section_lines:
            if len(line) >= 25:
                chunks.append({"section": section, "text": line})
        joined = " ".join(section_lines)
        for sentence in _sentences(joined):
            if len(sentence) >= 40:
                chunks.append({"section": section, "text": sentence})

    if not chunks:
        chunks = [{"section": "resume", "text": sentence} for sentence in _sentences(text) if len(sentence) >= 25]
    return chunks[:120]


def _bm25(job_description: str, chunks: List[Dict[str, str]]) -> Dict[str, Any]:
    if not chunks:
        return {"score": 0.0, "top_chunks": []}

    query_tokens = _tokenize(job_description)
    if not query_tokens:
        return {"score": 0.0, "top_chunks": []}

    query = Counter(query_tokens)
    chunk_tokens = [_tokenize(chunk["text"]) for chunk in chunks]
    avgdl = sum(len(tokens) for tokens in chunk_tokens) / max(len(chunk_tokens), 1)
    doc_freq = Counter()
    for tokens in chunk_tokens:
        for token in set(tokens):
            doc_freq[token] += 1

    k1 = 1.5
    b = 0.75
    scored = []
    for idx, tokens in enumerate(chunk_tokens):
        if not tokens:
            continue
        term_freq = Counter(tokens)
        dl = len(tokens)
        score = 0.0
        for token, qf in query.items():
            if token not in term_freq:
                continue
            df = doc_freq[token]
            idf = math.log(1 + (len(chunks) - df + 0.5) / (df + 0.5))
            numerator = term_freq[token] * (k1 + 1)
            denominator = term_freq[token] + k1 * (1 - b + b * dl / max(avgdl, 1))
            score += idf * numerator / denominator * min(qf, 3)
        if score > 0:
            scored.append((score, chunks[idx]))

    scored.sort(key=lambda item: item[0], reverse=True)
    best = scored[0][0] if scored else 0.0
    normalized = 1 - math.exp(-best / 12.0)
    return {
        "score": max(0.0, min(normalized, 1.0)),
        "top_chunks": [
            {
                "section": chunk["section"],
                "snippet": chunk["text"][:260],
                "bm25": round(score, 3),
            }
            for score, chunk in scored[:5]
        ],
    }


def _responsibility_alignment(job_description: str, sections: Dict[str, List[str]], terms: List[Dict[str, Any]]) -> Dict[str, Any]:
    experience_text = " ".join(sections.get("experience", []) + sections.get("projects", []))
    if not experience_text:
        experience_text = " ".join(sections.get("header", []) + sections.get("summary", []))

    jd_verbs = sorted({token for token in _tokenize(job_description) if token in ACTION_VERBS})
    resume_verbs = sorted({token for token in _tokenize(experience_text) if token in ACTION_VERBS})
    verb_overlap = sorted(set(jd_verbs).intersection(resume_verbs))
    verb_score = len(verb_overlap) / len(jd_verbs) if jd_verbs else len(resume_verbs) / 8 if resume_verbs else 0

    critical_terms = [item for item in terms if item.get("critical")]
    experience_hits = [item for item in critical_terms if _contains_term(experience_text, item["term"])]
    term_score = len(experience_hits) / len(critical_terms) if critical_terms else 0

    score = max(0.0, min((verb_score * 0.45) + (term_score * 0.55), 1.0))
    return {
        "score": score,
        "matched_action_verbs": verb_overlap[:12],
        "missing_action_verbs": [verb for verb in jd_verbs if verb not in verb_overlap][:12],
        "experience_term_hits": experience_hits[:12],
    }


def _accomplishment_signal(text: str, sections: Dict[str, List[str]]) -> Dict[str, Any]:
    candidate_lines = sections.get("experience", []) + sections.get("projects", [])
    if not candidate_lines:
        candidate_lines = _lines(text)

    if not candidate_lines:
        return {"score": 0.0, "metric_lines": [], "action_lines": []}

    metric_lines = [line for line in candidate_lines if METRIC_RE.search(line)]
    action_lines = []
    for line in candidate_lines:
        first_token = _tokenize(line[:45])
        if first_token and first_token[0] in ACTION_VERBS:
            action_lines.append(line)

    metric_score = min(len(metric_lines) / max(3, len(candidate_lines) * 0.35), 1.0)
    action_score = min(len(action_lines) / max(4, len(candidate_lines) * 0.4), 1.0)
    score = (metric_score * 0.6) + (action_score * 0.4)
    return {
        "score": max(0.0, min(score, 1.0)),
        "metric_lines": [line[:220] for line in metric_lines[:5]],
        "action_lines": [line[:220] for line in action_lines[:5]],
    }


def _parseability_signal(text: str, sections: Dict[str, List[str]]) -> Dict[str, Any]:
    expected_sections = {"skills", "experience", "education"}
    found_sections = expected_sections.intersection(sections.keys())
    section_score = len(found_sections) / len(expected_sections)

    lines = _lines(text)
    if not lines:
        return {"score": 0.0, "found_sections": [], "risks": ["No parseable lines were found."]}

    long_line_ratio = sum(1 for line in lines if len(line) > 170) / len(lines)
    symbol_ratio = len(re.findall(r"[^a-zA-Z0-9\s.,:;/%+#@()&\-]", text or "")) / max(len(text or ""), 1)
    risk_penalty = min((long_line_ratio * 0.5) + (symbol_ratio * 5), 0.45)
    score = max(0.0, min((section_score * 0.75) + 0.25 - risk_penalty, 1.0))

    risks = []
    if "skills" not in sections:
        risks.append("No explicit skills section detected.")
    if "experience" not in sections:
        risks.append("No explicit experience section detected.")
    if long_line_ratio > 0.35:
        risks.append("Many extracted lines are very long; tables or dense formatting may parse poorly.")
    if symbol_ratio > 0.025:
        risks.append("High special-character density can confuse ATS parsing.")

    return {
        "score": score,
        "found_sections": sorted(found_sections),
        "risks": risks,
    }


def _contact_signal(text: str) -> Dict[str, Any]:
    checks = {
        "email": bool(re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text or "")),
        "phone": bool(re.search(r"(?:(?:\+?\d{1,3}[-.\s]?)|(?:\(\+?\d{1,3}\)[-.\s]?))?(?:\d{3}[-.\s]?\d{3}[-.\s]?\d{4})", text or "")),
        "linkedin": "linkedin.com" in _normalize(text),
        "portfolio_or_github": any(marker in _normalize(text) for marker in ("github.com", "portfolio", "behance.net", "dribbble.com")),
    }
    score = (sum(checks.values()) / len(checks)) if checks else 0.0
    return {"score": score, "checks": checks}


def _evidence_for_terms(resume_text: str, matched_terms: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    evidence = []
    seen_lines = set()
    lines = _lines(resume_text)
    for item in matched_terms:
        term = item["term"]
        for line in lines:
            if line in seen_lines:
                continue
            if _contains_term(line, term):
                seen_lines.add(line)
                evidence.append({"term": term, "snippet": line[:260]})
                break
        if len(evidence) >= 10:
            break
    return evidence


def _score_component(name: str, key: str, score: float, weight: int, why: str, signals: List[str]) -> Dict[str, Any]:
    normalized = int(round(max(0.0, min(score, 1.0)) * 100))
    return {
        "name": name,
        "key": key,
        "score": normalized,
        "weight": weight,
        "weighted_points": round(normalized * weight / 100, 1),
        "why": why,
        "signals": [signal for signal in signals if signal],
    }


def _generate_aggressive_feedback(score: int, coverage: Dict[str, Any], industry: str, critical_context: Dict[str, Any]) -> Dict[str, Any]:
    """Generate aggressive, direct feedback with specific improvement recommendations"""

    feedback_level = "excellent" if score >= AGGRESSIVE_THRESHOLDS['excellent'] else \
                    "good" if score >= AGGRESSIVE_THRESHOLDS['good'] else \
                    "moderate" if score >= AGGRESSIVE_THRESHOLDS['moderate'] else \
                    "poor" if score >= AGGRESSIVE_THRESHOLDS['poor'] else "critical"

    critical_missing = len(coverage.get("critical_missing", []))
    total_critical = len(coverage.get("critical_matched", [])) + critical_missing

    aggressive_insights = {
        "overall_assessment": {
            "level": feedback_level,
            "score_range": f"{AGGRESSIVE_THRESHOLDS.get(feedback_level, 'critical')}+",
            "industry_context": f"Evaluated against competitive {industry.title()} market standards"
        },
        "critical_gaps": [],
        "immediate_actions": [],
        "long_term_strategy": []
    }

    # Critical gaps analysis
    if critical_missing > 0:
        aggressive_insights["critical_gaps"].extend([
            f"CRITICAL: {critical_missing}/{total_critical} mandatory requirements missing",
            f"Industry standard: {industry.title()} roles require {total_critical} core competencies",
            "Missing elements are immediate disqualifiers for competitive positions"
        ])

        # Specific missing critical terms
        missing_terms = [item.get('term', '') for item in coverage.get("critical_missing", [])[:5]]
        if missing_terms:
            aggressive_insights["critical_gaps"].append(f"Specifically missing: {', '.join(missing_terms)}")

    # Immediate actions based on score level
    if feedback_level in ["critical", "poor"]:
        aggressive_insights["immediate_actions"].extend([
            "REBUILD: This resume requires fundamental restructuring for the target role",
            "SKILLS AUDIT: Conduct immediate gap analysis against job requirements",
            "CONTENT OVERHAUL: Replace generic content with role-specific evidence",
            "QUANTIFICATION: Add metrics to ALL experience descriptions"
        ])
    elif feedback_level == "moderate":
        aggressive_insights["immediate_actions"].extend([
            "TARGET: Refocus resume content to match job description precisely",
            "ENHANCE: Strengthen weak sections with specific, quantifiable achievements",
            "OPTIMIZE: Reorder content to prioritize most relevant experience",
            "VALIDATE: Ensure all claims are backed by demonstrable evidence"
        ])
    elif feedback_level == "good":
        aggressive_insights["immediate_actions"].extend([
            "POLISH: Fine-tune language and formatting for maximum impact",
            "STRENGTHEN: Add more quantifiable results where possible",
            "TAILOR: Customize remaining generic content for this specific role",
            "VALIDATE: Double-check all technical claims and dates"
        ])

    # Long-term strategy
    aggressive_insights["long_term_strategy"].extend([
        f"DEVELOP: Build expertise in {industry.title()} industry-specific skills and terminology",
        "NETWORK: Connect with professionals in target industry for insights",
        "CERTIFY: Pursue relevant certifications to close competency gaps",
        "SPECIALIZE: Consider focusing on a specific niche within the industry",
        "MENTOR: Seek guidance from successful professionals in target roles"
    ])

    # Industry-specific recommendations
    if industry == "tech":
        aggressive_insights["long_term_strategy"].extend([
            "CONTRIBUTE: Build portfolio through GitHub, open-source projects",
            "CERTIFY: Obtain industry-recognized certifications (AWS, GCP, etc.)",
            "NETWORK: Attend tech meetups, conferences, and hackathons"
        ])
    elif industry == "finance":
        aggressive_insights["long_term_strategy"].extend([
            "CERTIFY: Pursue CFA, CPA, or industry-specific designations",
            "NETWORK: Join professional finance associations and networking events",
            "SPECIALIZE: Focus on specific finance sub-sector expertise"
        ])

    return aggressive_insights

def _generation_policy(score: int, coverage: Dict[str, Any], resume_text: str) -> Dict[str, Any]:
    critical_coverage = float(coverage.get("critical_coverage", 0.0))
    matched_critical = len(coverage.get("critical_matched", []))
    reasons = []

    if len((resume_text or "").strip()) < 300:
        reasons.append("The extracted resume text is too short to safely rewrite.")
    if score < MIN_GENERATION_SCORE:
        reasons.append(f"The match score is below {MIN_GENERATION_SCORE}, which suggests the role may be too different.")
    if critical_coverage < MIN_CRITICAL_COVERAGE and matched_critical < MIN_MATCHED_CRITICAL_TERMS:
        reasons.append(
            "Too few critical job requirements are evidenced in the uploaded resume, so a rewrite would risk inventing experience."
        )

    can_generate = not reasons
    return {
        "can_generate": can_generate,
        "reason": "Resume rewrite is allowed because enough role-relevant evidence exists in the uploaded resume."
        if can_generate
        else " ".join(reasons),
        "thresholds": {
            "minimum_score": MIN_GENERATION_SCORE,
            "minimum_critical_coverage": MIN_CRITICAL_COVERAGE,
            "minimum_matched_critical_terms": MIN_MATCHED_CRITICAL_TERMS,
        },
        "observed": {
            "score": score,
            "critical_coverage": round(critical_coverage, 3),
            "matched_critical_terms": matched_critical,
        },
    }


def analyze_resume_match(job_description: str, resume_text: str) -> Dict[str, Any]:
    """
    Advanced ATS-style analysis with industry-specific benchmarks and aggressive gap detection.
    Enhanced with semantic embeddings, skill coverage, section evidence, and parseability.
    """
    if not job_description or not resume_text:
        policy = _generation_policy(0, {"critical_coverage": 0, "critical_matched": []}, resume_text)
        return {
            "score": 0,
            "score_breakdown": [],
            "ats_signals": {"matched_terms": [], "missing_terms": [], "exact_evidence": []},
            "generation_policy": policy,
            "industry_analysis": {},
            "aggressive_feedback": {}
        }

    # Detect industry for tailored analysis
    detected_industry = _detect_industry(job_description)
    industry_config = INDUSTRY_BENCHMARKS.get(detected_industry, INDUSTRY_BENCHMARKS['default'])

    sections = extract_sections(resume_text)
    key_terms = _extract_key_terms(job_description)
    coverage = _term_coverage(key_terms, resume_text)
    critical_context = _critical_context_terms(job_description, detected_industry)

    tfidf_score = _tfidf_similarity(job_description, resume_text)
    char_score = _char_similarity(job_description, resume_text)
    semantic_score = _semantic_similarity(job_description, resume_text)
    semantic_used = semantic_score is not None
    semantic_proxy = (tfidf_score * 0.65) + (char_score * 0.35)
    semantic_value = semantic_score if semantic_used else semantic_proxy

    chunks = _resume_chunks(resume_text, sections)
    bm25_result = _bm25(job_description, chunks)
    responsibility = _responsibility_alignment(job_description, sections, key_terms)
    accomplishment = _accomplishment_signal(resume_text, sections)
    parseability = _parseability_signal(resume_text, sections)
    contact = _contact_signal(resume_text)

    # Industry-adjusted component weights
    components = [
        _score_component(
            "Industry Semantic Alignment",
            "semantic_context",
            semantic_value * industry_config['semantic_weight'] * 2,  # Weighted by industry
            int(20 * industry_config['semantic_weight'] * 2),
            f"Industry-specific semantic analysis for {detected_industry.title()}. Measures contextual fit beyond keywords."
            if semantic_used
            else f"Industry-tailored lexical analysis for {detected_industry.title()} using advanced ngram techniques.",
            [
                f"Industry: {detected_industry.title()}",
                f"Embedding similarity: {round(semantic_score, 3)}" if semantic_used else "Embedding similarity: unavailable",
                f"Industry-adjusted TF-IDF: {round(tfidf_score * industry_config['semantic_weight'], 3)}",
            ],
        ),
        _score_component(
            "Technical Skills Match",
            "technical_alignment",
            (tfidf_score * 0.75) + (char_score * 0.25) * industry_config['technical_weight'] * 2,
            int(16 * industry_config['technical_weight'] * 2),
            f"Critical technical competency assessment for {detected_industry.title()} roles. Essential skills must be demonstrable.",
            [
                f"{len(coverage['matched'])} technical terms matched",
                f"{len(coverage['critical_missing'])} critical skills missing",
                f"Industry benchmark: {len(industry_config['critical_tech_terms'])} key terms required"
            ],
        ),
        _score_component(
            "Experience Evidence Strength",
            "experience_evidence",
            bm25_result["score"] * industry_config['experience_weight'] * 2,
            int(14 * industry_config['experience_weight'] * 2),
            f"Evaluates the strength and relevance of experience evidence for {detected_industry.title()} positions.",
            [f"Top evidence section: {bm25_result['top_chunks'][0]['section']}" if bm25_result["top_chunks"] else f"Critical: No strong {detected_industry.title()} experience evidence found"],
        ),
        _score_component(
            "Critical Requirements Coverage",
            "critical_requirements",
            coverage["critical_coverage"] * 1.5,  # More aggressive weighting
            25,  # Increased weight for critical requirements
            f"Zero-tolerance assessment of mandatory {detected_industry.title()} requirements. Missing critical elements are disqualifying.",
            [
                f"{len(coverage['critical_matched'])} of {len(coverage['critical_matched']) + len(coverage['critical_missing'])} critical requirements met",
                f"Pattern matches: {sum(len(matches) for matches in critical_context['patterns'].values())} detected",
                "WARNING: Missing critical requirements significantly impact candidacy" if len(coverage['critical_missing']) > 3 else "Critical requirements adequately addressed"
            ],
        ),
        _score_component(
            "Professional Impact Demonstration",
            "impact_demonstration",
            responsibility["score"] * industry_config['experience_weight'] * 1.8,
            int(10 * industry_config['experience_weight'] * 1.8),
            f"Assesses ability to demonstrate value creation in {detected_industry.title()} contexts through action verbs and quantified achievements.",
            [
                f"Action verbs matched: {len(responsibility['matched_action_verbs'])}",
                f"Quantified achievements: {len(accomplishment['metric_lines'])}",
                "CRITICAL: No quantifiable impact demonstrated" if len(accomplishment['metric_lines']) == 0 else "Impact metrics present and compelling"
            ],
        ),
        _score_component(
            "Achievement Quantification",
            "achievement_quantification",
            accomplishment["score"] * 1.3,
            10,
            f"Evaluates the presence of measurable outcomes and results in {detected_industry.title()} work.",
            [
                f"{len(accomplishment['metric_lines'])} achievement metrics found",
                f"{len(accomplishment['action_lines'])} action-oriented statements",
                "MAJOR GAP: Achievements lack quantification" if len(accomplishment['metric_lines']) < 2 else "Strong achievement documentation"
            ],
        ),
        _score_component(
            "ATS Compatibility & Structure",
            "ats_compatibility",
            parseability["score"] * 1.2,
            10,
            f"Critical assessment of resume formatting for {detected_industry.title()} ATS systems and recruiter parsing.",
            [
                f"Structure score: {round(parseability['score'] * 100)}/100",
                f"Issues: {len(parseability['risks'])} formatting problems detected",
                "CRITICAL: ATS parsing issues detected - resume may be filtered out" if parseability["score"] < 0.70 else "ATS-friendly formatting confirmed"
            ],
        ),
        _score_component(
            "Professional Presentation",
            "professional_presentation",
            (contact["score"] * 0.7 + parseability["score"] * 0.3) * industry_config['soft_skills_weight'] * 2,
            int(4 * industry_config['soft_skills_weight'] * 2),
            f"Evaluates professional presentation standards expected in {detected_industry.title()}.",
            [
                f"Contact completeness: {round(contact['score'] * 100)}/100",
                f"Professional sections: {len(parseability['found_sections'])} detected",
                "PROFESSIONALISM CONCERN: Incomplete contact information" if contact["score"] < 0.75 else "Professional presentation standards met"
            ],
        ),
    ]

    score = int(round(sum(component["weighted_points"] for component in components)))
    score = min(max(score, 0), 100)
    policy = _generation_policy(score, coverage, resume_text)

    # Generate aggressive feedback based on score thresholds
    aggressive_feedback = _generate_aggressive_feedback(score, coverage, detected_industry, critical_context)

    return {
        "score": score,
        "scoring_model": {
            "name": "Aura Advanced ATS Signal Engine V4.0",
            "note": (
                "Industry-tailored ATS analysis using advanced NLP, semantic embeddings, and competitive benchmarks. "
                "Provides aggressive gap analysis with specific improvement recommendations for maximum impact."
            ),
            "industry": detected_industry,
            "analysis_version": "4.0-aggressive"
        },
        "score_breakdown": components,
        "ats_signals": {
            "matched_terms": coverage["matched"][:35],
            "missing_terms": coverage["missing"][:35],
            "critical_matched_terms": coverage["critical_matched"][:25],
            "critical_missing_terms": coverage["critical_missing"][:25],
            "exact_evidence": _evidence_for_terms(resume_text, coverage["matched"]),
            "retrieved_evidence": bm25_result["top_chunks"],
            "parseability_risks": parseability["risks"],
            "impact_examples": accomplishment["metric_lines"],
        },
        "generation_policy": policy,
        "industry_analysis": {
            "detected_industry": detected_industry,
            "industry_benchmarks": industry_config,
            "critical_patterns": critical_context['patterns'],
            "industry_score_multiplier": industry_config['seniority_multipliers'].get('mid', 1.0)
        },
        "aggressive_feedback": aggressive_feedback
    }


def get_hybrid_score(jd_text: str, resume_text: str, semantic_weight: float = 0.7, tfidf_weight: float = 0.3) -> float:
    """
    Backward-compatible score wrapper. The old parameters are retained for callers,
    but scoring now comes from the multi-signal ATS-style analysis.
    """
    del semantic_weight, tfidf_weight
    return analyze_resume_match(jd_text, resume_text).get("score", 0)


def rank_resumes(job_description: str, resumes_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ranks parsed resumes using the Aura ATS-style signal engine.
    """
    if not resumes_data:
        return []

    for resume in resumes_data:
        text = resume.get("text", "")
        try:
            analysis = analyze_resume_match(job_description, text)
            resume.update(analysis)
        except Exception as exc:
            logger.error("Failed to score resume: %s", exc, exc_info=True)
            resume["score"] = 0
            resume["score_breakdown"] = []
            resume["ats_signals"] = {"matched_terms": [], "missing_terms": [], "exact_evidence": []}
            resume["generation_policy"] = _generation_policy(0, {"critical_coverage": 0, "critical_matched": []}, text)

    return sorted(resumes_data, key=lambda item: item.get("score", 0), reverse=True)
