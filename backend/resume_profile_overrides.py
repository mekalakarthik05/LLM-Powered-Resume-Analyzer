import copy
import json
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional


_OVERRIDES_PATH = os.path.join(os.path.dirname(__file__), "resume_profile_overrides.json")


def _normalized_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


@lru_cache(maxsize=1)
def _load_overrides() -> List[Dict[str, Any]]:
    try:
        with open(_OVERRIDES_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []
    return data.get("profiles", []) if isinstance(data, dict) else []


def get_profile_override(resume_text: str, name_line: Optional[str] = None, contact_lines: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    normalized_resume = _normalized_text(resume_text)
    normalized_name = _normalized_text(name_line or "")
    contact_blob = "\n".join(contact_lines or []).lower()
    resume_lower = (resume_text or "").lower()

    for profile in _load_overrides():
        match = profile.get("match", {})
        emails = [email.lower() for email in match.get("emails", [])]
        phones = [_normalized_text(phone) for phone in match.get("phones", [])]
        name_tokens = [_normalized_text(token) for token in match.get("name_tokens", [])]

        email_match = any(email and (email in resume_lower or email in contact_blob) for email in emails)
        phone_match = any(phone and phone in normalized_resume for phone in phones)
        name_match = any(token and token in normalized_resume for token in name_tokens)
        explicit_name_match = any(token and token == normalized_name for token in name_tokens)

        if email_match or phone_match or name_match or explicit_name_match:
            structured_resume = profile.get("structured_resume")
            if isinstance(structured_resume, dict):
                return copy.deepcopy(structured_resume)

    return None
