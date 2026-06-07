import os
import pdfplumber
import docx
import logging

logger = logging.getLogger(__name__)

def extract_text_from_file(file_path: str) -> str:
    """
    Advanced multi-format document parser.
    Uses pdfplumber for highly accurate PDF text extraction (preserves layout/spacing better),
    python-docx for Word documents, and standard encoding for plain text.
    """
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext == ".pdf":
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    # Extract text while trying to maintain structural integrity
                    extracted = page.extract_text(layout=True)
                    if extracted:
                        text += extracted + "\n"
        elif ext == ".docx":
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        else:
            logger.warning(f"Unsupported file extension for parsing: {ext}")
            
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        
    return text.strip()
