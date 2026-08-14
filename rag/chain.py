"""
RBAC-filtered retriever + base LCEL RAG chain using Groq.
"""

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from rbac.access_control import get_allowed_doc_roles

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

PROMPT = ChatPromptTemplate.from_template(
    """You are an internal company assistant. Answer the question using ONLY
the context below. If the context doesn't contain the answer, say you don't
have access to that information - do not guess.

Context:
{context}

Question: {question}

Answer:"""
)


def _format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


def get_retriever(user_role: str, k: int = 4):
    allowed_roles = get_allowed_doc_roles(user_role)

    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=_embeddings,
        collection_name="company_docs",
    )

    return vectorstore.as_retriever(
        search_kwargs={
            "k": k,
            "filter": {"role": {"$in": allowed_roles}},
        }
    )


def ask(user_role: str, query: str) -> dict:
    retriever = get_retriever(user_role)
    retrieved_docs = retriever.invoke(query)

    llm = ChatGroq(model=GROQ_MODEL, temperature=0.1)

    chain = (
        {"context": lambda x: _format_docs(retrieved_docs), "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(query)

    sources = [
        {"source": d.metadata.get("source"), "role": d.metadata.get("role")}
        for d in retrieved_docs
    ]

    return {"answer": answer, "sources": sources, "retrieved_count": len(retrieved_docs)}