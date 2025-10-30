AI RFQ Compiler (Phase 1: Market-Context Agent)

An AI agent that ingests municipal RFQ/RFP PDFs, identifies key requirements, and drafts financing/trading responses as an editable Word document.

Features (Phase 1)
- PDF ingestion and text extraction (PyPDF2/Docling-ready)
- Vector store indexing with ChromaDB for semantic search
- Retrieval-augmented generation (RAG) using LangChain and OpenAI/Anthropic/Gemini
- Word document generation using python-docx
- Simple CLI: index PDFs and generate a draft proposal

Quickstart
1) Create virtual environment and install dependencies
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2) Configure environment
   Copy .env.example to .env
   - For local Gemma via Ollama (no OpenAI key needed):
     PROVIDER=ollama
     OLLAMA_BASE_URL=http://localhost:11434
     MODEL_NAME=gemma2:27b
     (ensure `ollama run gemma2:27b` has been pulled/available)
   - For OpenAI:
     PROVIDER=openai
     OPENAI_API_KEY=...  
     MODEL_NAME=gpt-4o-mini

3) Index your PDFs
   python -m src.main index ./data/samples

4) Generate a draft proposal
   python -m src.main draft --client "City of Example" --rfq_title "Municipal Bond Underwriting RFQ 2025" --output ./outputs/example_proposal.docx

Free sources for RFQ/RFP PDFs
- EMMA (MSRB): https://emma.msrb.org/
- City/County procurement portals
- State procurement portals (BidSync, Bonfire, OpenGov Procure)
- School districts, utilities, transportation agencies procurement pages

Use scripts/fetch_samples.py to download known URLs
   python scripts/fetch_samples.py --urls-file scripts/sample_urls.txt --out-dir ./data/samples

Project Structure
- src/main.py (CLI)
- src/ingest/pdf_loader.py (PDF extraction)
- src/vectorstore/chroma_store.py (index/search)
- src/llm/agent.py (RAG proposal)
- src/generation/word_export.py (DOCX output)
# rfq-compailer
\n\n## Languages\nSupported: English (en), Español (es), Français (fr), Deutsch (de), हिन्दी (hi), 简体中文 (zh).\nSee locale files in `locales/`.\n
