"""
Out-of-scope detection: decides whether a query is even answerable from
company data before retrieval runs. Keeps the bot from confidently
answering things it has no business answering - general trivia, current
events, anything unrelated to the company.
"""

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

SCOPE_PROMPT = ChatPromptTemplate.from_template(
    """You are a scope classifier for an internal company chatbot. The chatbot
only answers questions about company data: finance, HR, payroll, employee
records and contact information, company policies, product roadmap, and
general company information.

Decide if the following question is IN_SCOPE (answerable from internal
company documents) or OUT_OF_SCOPE (general knowledge, personal advice,
current events, or anything unrelated to this company).

Respond with exactly one word: IN_SCOPE or OUT_OF_SCOPE

Examples:
Question: What was Q3 revenue?
Answer: IN_SCOPE

Question: What's the capital of France?
Answer: OUT_OF_SCOPE

Question: Who won the election?
Answer: OUT_OF_SCOPE

Question: What is our remote work policy?
Answer: IN_SCOPE

Question: What is an employee's contact information?
Answer: IN_SCOPE

Question: What is [employee name]'s phone number and email?
Answer: IN_SCOPE

Question: What is my personal phone number?
Answer: OUT_OF_SCOPE

Question: {query}
Answer:"""
)


def is_out_of_scope(query: str, callbacks=None) -> bool:
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    chain = SCOPE_PROMPT | llm | StrOutputParser()
    config = {"callbacks": callbacks} if callbacks else {}
    result = chain.invoke({"query": query}, config=config).strip().upper()
    return "OUT_OF_SCOPE" in result