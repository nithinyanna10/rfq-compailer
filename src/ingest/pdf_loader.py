from pathlib import Path
from typing import List, Tuple
from PyPDF2 import PdfReader

def extract_texts_from_pdfs(input_path: str) -> List[Tuple[str, str]]:
    path = Path(input_path)
    files = list(path.glob("**/*.pdf")) if path.is_dir() else [path]
    results: List[Tuple[str, str]] = []
    for f in files:
        try:
            reader = PdfReader(str(f))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            results.append((f.name, text))
        except Exception as e:
            # skip unreadable files
            continue
    return results
