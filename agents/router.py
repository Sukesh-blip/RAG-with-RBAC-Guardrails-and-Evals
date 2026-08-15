"""
Router/Planner node: decomposes a query into one or more sub-questions.
A compound question ("compare marketing spend to payroll costs") gets
split so retrieval runs focused lookups for each piece, instead of one
blended search that blurs together data from different domains.
"""

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

ROUTER_PROMPT = ChatPromptTemplate.from_template(
    """You are a query planner for an internal company chatbot. Break the
user's question into 1-3 focused sub-questions that together would let you
answer it fully. If it's already a single simple question, return it as-is,
unchanged.

Return ONLY the sub-questions, one per line. No numbering, no extra text.

Question: {query}
Sub-questions:"""
)


def plan_sub_queries(query: str) -> list[str]:
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    chain = ROUTER_PROMPT | llm | StrOutputParser()
    result = chain.invoke({"query": query})
    sub_queries = [
        line.strip("-• ").strip() for line in result.strip().split("\n") if line.strip()
    ]
    return sub_queries if sub_queries else [query]