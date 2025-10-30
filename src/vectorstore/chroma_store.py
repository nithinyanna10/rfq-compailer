from typing import List, Tuple
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

class ChromaIndex:
    def __init__(self, persist_directory: str, collection_name: str = "rfq_docs"):
        sentence_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=sentence_ef,
        )

    def index(self, docs: List[Tuple[str, str]]):
        if not docs:
            return
        ids = [f"doc-{i}" for i in range(len(docs))]
        texts = [t for _, t in docs]
        metadatas = [{"source": name} for name, _ in docs]
        self.collection.add(ids=ids, documents=texts, metadatas=metadatas)

    def query(self, query_text: str, n: int = 5):
        return self.collection.query(query_texts=[query_text], n_results=n)
