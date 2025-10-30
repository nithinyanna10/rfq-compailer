import argparse
from typing import List
from .config import CHROMA_PERSIST_DIR, COLLECTION_NAME
from .ingest.pdf_loader import extract_texts_from_pdfs
from .vectorstore.chroma_store import ChromaIndex
from .llm.agent import ProposalAgent
from .generation.word_export import write_proposal_docx


def cmd_index(args):
    docs = extract_texts_from_pdfs(args.path)
    idx = ChromaIndex(CHROMA_PERSIST_DIR, COLLECTION_NAME)
    idx.index(docs)
    print(f"Indexed {len(docs)} documents")


def cmd_draft(args):
    idx = ChromaIndex(CHROMA_PERSIST_DIR, COLLECTION_NAME)
    q = idx.query(args.query or args.rfq_title, n=5)
    context_chunks: List[str] = q.get("documents", [[]])[0]
    agent = ProposalAgent()
    proposal = agent.generate(args.client, args.rfq_title, context_chunks)
    write_proposal_docx(proposal, args.output)
    print(f"Draft written to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="AI RFQ Compiler")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Index PDFs from a path (file or dir)")
    p_index.add_argument("path", help="Path to a PDF or directory of PDFs")
    p_index.set_defaults(func=cmd_index)

    p_draft = sub.add_parser("draft", help="Generate a draft proposal")
    p_draft.add_argument("--client", required=True)
    p_draft.add_argument("--rfq_title", required=True)
    p_draft.add_argument("--output", required=True)
    p_draft.add_argument("--query", default="")
    p_draft.set_defaults(func=cmd_draft)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
