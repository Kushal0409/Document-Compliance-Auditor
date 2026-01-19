import os
from typing import Optional, Tuple

from docx import Document  # python-docx
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def is_supported_file_type(path: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a file path has a supported extension.
    
    Returns:
        (is_valid, warning_message)
        - is_valid: True if file type is supported
        - warning_message: Warning message if not supported, None if supported
    """
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    
    if ext not in SUPPORTED_EXTENSIONS:
        warning = (
            f"⚠️ WARNING: Unsupported file type '{ext}' for file '{os.path.basename(path)}'. "
            f"Only the following file types are supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}. "
            f"This file will be skipped."
        )
        return False, warning
    
    return True, None


def extract_text_from_path(path: str, show_warning: bool = True) -> str:
    """
    Read text from a file path.

    Supports:
    - .txt
    - .pdf
    - .docx
    
    Parameters:
        path: File path to read
        show_warning: If True, raises ValueError with warning for unsupported types.
                     If False, silently raises ValueError (for reference_docs folder).
    
    Raises:
        ValueError: If file type is not supported
    """
    _, ext = os.path.splitext(path)
    ext = ext.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        if show_warning:
            raise ValueError(
                f"⚠️ WARNING: Unsupported file type '{ext}' for file '{os.path.basename(path)}'. "
                f"Only the following file types are supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}. "
                f"Please use a .txt, .pdf, or .docx file instead."
            )
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    if ext == ".docx":
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs).strip()

    if ext == ".pdf":
        reader = PdfReader(path)
        texts = []
        for page in reader.pages:
            page_text: Optional[str] = page.extract_text()
            if page_text:
                texts.append(page_text)
        return "\n".join(texts).strip()

    # Fallback (should never be reached due to ext check)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

