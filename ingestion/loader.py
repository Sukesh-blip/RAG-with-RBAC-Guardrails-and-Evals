"""
Ingestion pipeline: reads data/ folder, parses YAML frontmatter,
chunks documents, embeds them, and upserts into Chroma with
role metadata attached to every chunk.

Uses fastembed (ONNX-based) instead of sentence-transformers, since
sentence-transformers pulls in the full PyTorch library (~300-400MB just
to import), which alone was enough to exceed Render's 512MB free-tier
memory limit and cause the service to crash-loop.
"""

import os
import yaml
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def parse_frontmatter(raw_text: str) -> tuple[dict, str]:
    """Splits a markdown file into (metadata_dict, body_text)."""
    if not raw_text.startswith("---"):
        return {}, raw_text

    parts = raw_text.split("---", 2)
    if len(parts) < 3:
        return {}, raw_text

    metadata = yaml.safe_load(parts[1]) or {}
    body = parts[2].strip()
    return metadata, body


def load_documents() -> list[Document]:
    """Walks data/ recursively, parses each .md file into a Document."""
    docs = []
    for filepath in DATA_DIR.rglob("*.md"):
        raw_text = filepath.read_text(encoding="utf-8")
        metadata, body = parse_frontmatter(raw_text)

        metadata["source"] = filepath.name
        # Chroma metadata values must be scalar - stringify list fields
        if "requires_roles" in metadata and isinstance(metadata["requires_roles"], list):
            metadata["requires_roles"] = ",".join(metadata["requires_roles"])

        docs.append(Document(page_content=body, metadata=metadata))
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=75,
        separators=["\n## ", "\n\n", "\n", ". ", " "],
    )
    return splitter.split_documents(docs)


def run_ingestion() -> dict:
    """Main entry point: load, chunk, embed, upsert. Returns a summary."""
    docs = load_documents()
    chunks = chunk_documents(docs)

    embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL, threads=1)

    # Clear any existing collection first so /ingest is safe to call
    # repeatedly without silently duplicating chunks
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    try:
        client.delete_collection("company_docs")
    except Exception:
        pass  # collection didn't exist yet, nothing to clear

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name="company_docs",
    )
    vectorstore.persist()

    role_counts = {}
    for c in chunks:
        r = c.metadata.get("role", "unknown")
        role_counts[r] = role_counts.get(r, 0) + 1

    return {
        "documents_loaded": len(docs),
        "chunks_created": len(chunks),
        "chunks_by_role": role_counts,
    }


if __name__ == "__main__":
    summary = run_ingestion()
    print(summary)