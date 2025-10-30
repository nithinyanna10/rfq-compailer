from docx import Document
from pathlib import Path

def write_proposal_docx(text: str, output_path: str):
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    for line in text.split("\n\n"):
        doc.add_paragraph(line)
    doc.save(str(out))
