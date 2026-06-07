import re
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# Lazy-loaded spaCy to avoid torch DLL issues on Windows
nlp = None
SPACY_AVAILABLE = False

def _load_spacy_model():
    """Lazy load spaCy model only when needed"""
    global nlp, SPACY_AVAILABLE
    if SPACY_AVAILABLE or nlp is not None:
        return nlp
    
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
        return nlp
    except ImportError:
        logging.warning("spaCy not installed. NER features disabled. Run: pip install spacy && python -m spacy download en_core_web_sm")
        SPACY_AVAILABLE = False
        return None
    except OSError:
        logging.warning("spaCy model not found. NER features disabled. Run: python -m spacy download en_core_web_sm")
        SPACY_AVAILABLE = False
        return None
    except Exception as e:
        # Catch torch DLL errors and other runtime issues
        logging.warning(f"Failed to load spaCy (torch/dependency issue): {e}. NER features disabled.")
        SPACY_AVAILABLE = False
        return None

# Extended domain-specific ontology list (can be loaded from external JSON/DB)
TECH_ONTOLOGY = {
    "python", "java", "javascript", "react", "node", "aws", "docker", "fastapi", 
    "postgresql", "c++", "typescript", "sql", "machine learning", "pytorch", 
    "tensorflow", "nlp", "kubernetes", "gcp", "azure", "golang", "ruby", 
    "graphql", "rest api", "ci/cd", "microservices", "mongodb", "redis", "elasticsearch"
}

def extract_entities_spacy(text: str) -> Dict[str, List[str]]:
    """Uses spaCy advanced NER to extract Organizations, Geo-locations, and Dates"""
    model = _load_spacy_model()
    if not model:
        return {"organizations": [], "locations": []}
    
    doc = model(text[:10000]) # Limit length for performance if needed
    organizations = set([ent.text for ent in doc.ents if ent.label_ == "ORG"])
    locations = set([ent.text for ent in doc.ents if ent.label_ == "GPE"])
    
    return {
        "organizations": list(organizations),
        "locations": list(locations)
    }

def enrich_resume_data(raw_data: dict) -> dict:
    """
    Intelligent data enrichment layer. Uses Regex for deterministic patterns
    and NLP (spaCy) for probabilistic Named Entity Recognition (NER).
    Matches an extended taxonomy of tech skills.
    """
    text = raw_data.get("text", "")
    
    # Phone / Email regex matching
    emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    email = emails[0] if emails else None
    
    # Improved phone regex (handles international formats)
    phones = re.findall(r"(?:(?:\+?\d{1,3}[-.\s]?)|(?:\(\+?\d{1,3}\)[-.\s]?))?(?:\d{3}[-.\s]?\d{3}[-.\s]?\d{4})", text)
    phone = phones[0] if phones else None
    
    # Skill matching using token sets
    found_skills = list({skill for skill in TECH_ONTOLOGY if re.search(rf'\b{re.escape(skill)}\b', text, re.IGNORECASE)})
    
    # Advanced NLP Extraction
    nlp_entities = extract_entities_spacy(text)
    
    return {
        "filename": raw_data.get("filename"),
        "text": text,
        "details": {
            "email": email,
            "phone": phone,
            "skills": found_skills,
            "organizations": nlp_entities.get("organizations", []),
            "locations": nlp_entities.get("locations", [])
        }
    }
