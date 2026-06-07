import re
import unicodedata
from io import BytesIO
from typing import Any, Dict, List, Optional

from fpdf import FPDF


ACCENT = (28, 78, 128)
TEXT = (34, 34, 34)
MUTED = (92, 92, 92)
RULE = (188, 197, 210)
LINK = (28, 78, 128)


def _ascii(text: Any) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    return normalized.encode("ascii", "ignore").decode("ascii").strip()


def _clean_line(text: Any) -> str:
    line = _ascii(text)
    return re.sub(r"\s+", " ", line).strip()


def _safe_filename(name: Optional[str]) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", _clean_line(name or "optimized_resume.pdf")).strip("._")
    if not cleaned:
        cleaned = "optimized_resume.pdf"
    if not cleaned.lower().endswith(".pdf"):
        cleaned = f"{cleaned}.pdf"
    return cleaned


def _draft_to_structured_resume(payload: Dict[str, Any]) -> Dict[str, Any]:
    draft = str(payload.get("draft") or "")
    lines = [line.rstrip() for line in draft.splitlines()]

    name: Optional[str] = None
    contact_items: List[str] = []
    profile_links: List[Dict[str, str]] = []
    sections: List[Dict[str, Any]] = []
    current_section: Optional[Dict[str, Any]] = None
    seen_section = False

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line.isupper() and len(line) <= 48 and re.fullmatch(r"[A-Z &/]+", line):
            current_section = {"title": line, "lines": [], "layout": "lines"}
            if line in {"PROFESSIONAL EXPERIENCE", "PROJECTS"}:
                current_section["layout"] = "bullets"
            elif "SKILL" in line or "COMPETENC" in line:
                current_section["layout"] = "highlights"
            elif "PROFILE" in line or "SUMMARY" in line:
                current_section["layout"] = "paragraphs"
            sections.append(current_section)
            seen_section = True
            continue

        if not seen_section:
            if name is None:
                name = line
            elif "http" in line.lower() and ":" in line:
                label, _, url = line.partition(":")
                profile_links.append({"label": label.strip(), "url": url.strip()})
            else:
                contact_items.append(line)
            continue

        if current_section is not None:
            current_section["lines"].append(line[2:] if line.startswith("- ") else line)

    return {
        "template_key": "ats_single_column_advanced",
        "template_name": payload.get("template_used") or "ATS Resume Template",
        "ats_safe": True,
        "name": name,
        "contact_items": contact_items,
        "profile_links": profile_links,
        "sections": [section for section in sections if section.get("lines")],
    }


def _resume_doc(payload: Dict[str, Any]) -> Dict[str, Any]:
    structured = payload.get("structured_resume")
    if isinstance(structured, dict) and structured.get("sections"):
        return structured
    return _draft_to_structured_resume(payload)


def _line_text(label: str, value: str) -> str:
    clean_value = _clean_line(value)
    return f"{label}: {clean_value}" if clean_value else ""


def _link_text(item: Dict[str, Any]) -> str:
    return _line_text(str(item.get("label") or item.get("url_label") or "Link").strip(), str(item.get("url") or "").strip())


def _joined(items: List[str], sep: str = " | ") -> str:
    return sep.join(item for item in items if item)


def _reset_x(pdf: FPDF) -> None:
    pdf.set_x(pdf.l_margin)


def _render_header(pdf: FPDF, document: Dict[str, Any]) -> None:
    name = _ascii(document.get("name")) or "Optimized Resume"
    canonical_name = _clean_line(document.get("canonical_name") or name)
    contact_items = [_clean_line(item) for item in document.get("contact_items", []) if _clean_line(item)]
    if not contact_items and document.get("contact_lines"):
        contact_items = [_clean_line(item) for item in document.get("contact_lines", []) if _clean_line(item)]
    
    # Get profile link labels only (hide URLs in PDF)
    profile_link_labels = []
    for link in document.get("profile_links", []):
        if isinstance(link, dict):
            label = link.get("label") or link.get("url_label") or "Link"
            url = link.get("url")
            if url:
                profile_link_labels.append(_clean_line(label))

    pdf.set_title(canonical_name or name)
    pdf.set_author("Aura AI")
    pdf.set_creator("Aura AI Resume Analysis Engine")

    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*TEXT)
    _reset_x(pdf)
    pdf.multi_cell(0, 8.2, name)

    # Combine contact items and profile links on one line
    all_contact_info = contact_items + profile_link_labels
    if all_contact_info:
        pdf.ln(1)
        pdf.set_font("Helvetica", "", 10.3)
        pdf.set_text_color(*MUTED)
        _reset_x(pdf)
        pdf.multi_cell(0, 5.1, _joined(all_contact_info))

    pdf.ln(2.8)
    y = pdf.get_y()
    pdf.set_draw_color(*ACCENT)
    pdf.set_line_width(0.9)
    pdf.line(pdf.l_margin, y, 210 - pdf.r_margin, y)
    pdf.ln(4.5)


def _render_section_heading(pdf: FPDF, title: str) -> None:
    pdf.set_font("Helvetica", "B", 10.8)
    pdf.set_text_color(*ACCENT)
    _reset_x(pdf)
    pdf.cell(0, 5.8, _clean_line(title), ln=True)
    y = pdf.get_y()
    pdf.set_draw_color(*RULE)
    pdf.set_line_width(0.28)
    pdf.line(pdf.l_margin, y, 210 - pdf.r_margin, y)
    pdf.ln(2.6)


def _render_bullet(pdf: FPDF, text: str) -> None:
    clean = _clean_line(text)
    if not clean:
        return
    _reset_x(pdf)
    pdf.set_font("Helvetica", "", 10.3)
    pdf.set_text_color(*TEXT)
    pdf.cell(4, 5.1, "-", ln=False)
    pdf.multi_cell(0, 5.1, clean)
    _reset_x(pdf)


def _render_small_link(pdf: FPDF, label: str, url: str) -> None:
    link_text = _line_text(label, url)
    if not link_text:
        return
    pdf.set_font("Helvetica", "", 9.3)
    pdf.set_text_color(*LINK)
    _reset_x(pdf)
    pdf.multi_cell(0, 4.4, link_text)


def _render_generic_section(pdf: FPDF, section: Dict[str, Any]) -> None:
    lines = [_clean_line(line) for line in section.get("lines", []) if _clean_line(line)]
    if not lines:
        return

    _render_section_heading(pdf, section.get("title") or "SECTION")
    layout = section.get("layout") or "lines"
    pdf.set_text_color(*TEXT)

    if layout == "paragraphs":
        pdf.set_font("Helvetica", "", 10.5)
        for line in lines:
            _reset_x(pdf)
            pdf.multi_cell(0, 5.5, line)
            pdf.ln(0.3)
    elif layout in {"highlights", "bullets"}:
        for line in lines:
            _render_bullet(pdf, line)
            pdf.ln(0.4)
    else:
        pdf.set_font("Helvetica", "", 10.4)
        for line in lines:
            _reset_x(pdf)
            pdf.multi_cell(0, 5.2, line)
            pdf.ln(0.2)

    pdf.ln(1.2)


def _render_summary_section(pdf: FPDF, section: Dict[str, Any]) -> None:
    paragraphs = [_clean_line(line) for line in section.get("paragraphs", []) if _clean_line(line)]
    if not paragraphs:
        return
    _render_section_heading(pdf, section.get("title") or "SUMMARY")
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(*TEXT)
    for paragraph in paragraphs:
        _reset_x(pdf)
        pdf.multi_cell(0, 5.5, paragraph)
        pdf.ln(0.5)
    pdf.ln(0.8)


def _render_skills_section(pdf: FPDF, section: Dict[str, Any]) -> None:
    categories = section.get("categories", [])
    if not categories:
        return
    _render_section_heading(pdf, section.get("title") or "SKILLS")
    for category in categories:
        label = _clean_line(category.get("label"))
        items = ", ".join(_clean_line(item) for item in category.get("items", []) if _clean_line(item))
        if not label or not items:
            continue
        pdf.set_font("Helvetica", "B", 10.3)
        pdf.set_text_color(*TEXT)
        _reset_x(pdf)
        pdf.cell(23, 5.2, f"{label}:", ln=False)
        pdf.set_font("Helvetica", "", 10.3)
        pdf.multi_cell(0, 5.2, items)
        pdf.ln(0.4)
    pdf.ln(0.8)


def _render_experience_item(pdf: FPDF, item: Dict[str, Any]) -> None:
    title = _clean_line(item.get("title"))
    organization = _clean_line(item.get("organization"))
    location = _clean_line(item.get("location"))
    date_range = _clean_line(item.get("date_range"))
    meta = _joined([organization, location, date_range])

    if title:
        pdf.set_font("Helvetica", "B", 11.3)
        pdf.set_text_color(*TEXT)
        _reset_x(pdf)
        pdf.multi_cell(0, 5.9, title)

    if meta:
        pdf.set_font("Helvetica", "", 9.9)
        pdf.set_text_color(*MUTED)
        _reset_x(pdf)
        pdf.multi_cell(0, 4.8, meta)

    _render_small_link(pdf, str(item.get("url_label") or "Company website"), str(item.get("url") or ""))

    pdf.ln(0.6)
    for bullet in item.get("bullets", []):
        _render_bullet(pdf, bullet)
        pdf.ln(0.3)
    pdf.ln(1.5)


def _render_project_item(pdf: FPDF, item: Dict[str, Any]) -> None:
    name = _clean_line(item.get("name"))
    subtitle = _clean_line(item.get("subtitle"))
    title = _joined([name, subtitle], " | ")
    tech_stack = ", ".join(_clean_line(tech) for tech in item.get("tech_stack", []) if _clean_line(tech))

    if title:
        pdf.set_font("Helvetica", "B", 11.1)
        pdf.set_text_color(*TEXT)
        _reset_x(pdf)
        pdf.multi_cell(0, 5.8, title)

    if tech_stack:
        pdf.set_font("Helvetica", "", 9.9)
        pdf.set_text_color(*MUTED)
        _reset_x(pdf)
        pdf.multi_cell(0, 4.8, f"Tech Stack: {tech_stack}")

    _render_small_link(pdf, str(item.get("url_label") or "Project URL"), str(item.get("url") or ""))

    pdf.ln(0.6)
    for bullet in item.get("bullets", []):
        _render_bullet(pdf, bullet)
        pdf.ln(0.3)
    pdf.ln(1.5)


def _render_education_item(pdf: FPDF, item: Dict[str, Any]) -> None:
    institution = _clean_line(item.get("institution"))
    date_range = _clean_line(item.get("date_range"))
    degree = _clean_line(item.get("degree"))
    details = _clean_line(item.get("details"))

    header = _joined([institution, date_range])
    if header:
        pdf.set_font("Helvetica", "B", 10.9)
        pdf.set_text_color(*TEXT)
        _reset_x(pdf)
        pdf.multi_cell(0, 5.5, header)
    if degree or details:
        pdf.set_font("Helvetica", "", 10.1)
        pdf.set_text_color(*MUTED)
        _reset_x(pdf)
        pdf.multi_cell(0, 4.9, _joined([degree, details]))
    pdf.ln(1.1)


def _render_certification_item(pdf: FPDF, item: Dict[str, Any]) -> None:
    header = _joined([
        _clean_line(item.get("name")),
        _clean_line(item.get("issuer")),
        _clean_line(item.get("date")),
    ])
    if header:
        pdf.set_font("Helvetica", "B", 10.8)
        pdf.set_text_color(*TEXT)
        _reset_x(pdf)
        pdf.multi_cell(0, 5.4, header)
    _render_small_link(pdf, str(item.get("url_label") or "Credential"), str(item.get("url") or ""))
    pdf.ln(1.1)


def render_resume_pdf(payload: Dict[str, Any]) -> bytes:
    document = _resume_doc(payload or {})
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=13)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()

    _render_header(pdf, document)

    for section in document.get("sections", []) or []:
        kind = section.get("kind")
        if kind == "summary":
            _render_summary_section(pdf, section)
        elif kind == "skills":
            _render_skills_section(pdf, section)
        elif kind == "experience":
            _render_section_heading(pdf, section.get("title") or "PROFESSIONAL EXPERIENCE")
            for item in section.get("items", []):
                _render_experience_item(pdf, item)
        elif kind == "projects":
            _render_section_heading(pdf, section.get("title") or "PROJECTS")
            for item in section.get("items", []):
                _render_project_item(pdf, item)
        elif kind == "education":
            _render_section_heading(pdf, section.get("title") or "EDUCATION")
            for item in section.get("items", []):
                _render_education_item(pdf, item)
        elif kind == "certifications":
            _render_section_heading(pdf, section.get("title") or "CERTIFICATIONS")
            for item in section.get("items", []):
                _render_certification_item(pdf, item)
        else:
            _render_generic_section(pdf, section)

    buffer = BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()


def pdf_download_filename(payload: Dict[str, Any]) -> str:
    return _safe_filename(payload.get("download_pdf_filename") or payload.get("download_filename") or "optimized_resume.pdf")
