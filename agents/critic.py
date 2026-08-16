"""
Retrieval-Critic node: judges whether retrieved context is actually
sufficient to answer the question. If not, and a retry budget remains,
the query gets reformulated and sent back through retrieval instead of
generating a low-confidence answer from thin context.
"""

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_RETRIES = 1

CRITIC_PROMPT = ChatPromptTemplate.from_template(
    """Judge whether the context below is sufficient to answer the question.
Respond with exactly one word: SUFFICIENT or INSUFFICIENT.

Question: {question}

Context:
{context}

Judgment:"""
)

REFORMULATE_PROMPT = ChatPromptTemplate.from_template(
    """This question could not be answered from the retrieved context.
Rewrite it as a broader or differently-phrased search query that might
retrieve better matching documents. Return ONLY the rewritten query.

Original question: {question}"""
)


def judge_sufficiency(question: str, context: str, callbacks=None) -> bool:
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    chain = CRITIC_PROMPT | llm | StrOutputParser()
    config = {"callbacks": callbacks} if callbacks else {}
    result = chain.invoke({"question": question, "context": context}, config=config).strip().upper()
    return result.startswith("SUFFICIENT")


def reformulate_query(question: str, callbacks=None) -> str:
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.3)
    chain = REFORMULATE_PROMPT | llm | StrOutputParser()
    config = {"callbacks": callbacks} if callbacks else {}
    return chain.invoke({"question": question}, config=config).strip()