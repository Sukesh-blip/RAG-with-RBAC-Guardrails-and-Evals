"""
LangGraph state machine wiring together: scope check, query planning,
RBAC-filtered retrieval, retrieval self-correction, generation, and
PII guardrails.
"""

import os
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

from rag.chain import get_retriever, _format_docs, PROMPT
from guardrails.scope_guard import is_out_of_scope
from guardrails.pii_guard import redact_pii
from agents.router import plan_sub_queries
from agents.critic import judge_sufficiency, reformulate_query, MAX_RETRIES

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


class AgentState(TypedDict):
    role: str
    original_query: str
    active_query: str
    sub_queries: List[str]
    retrieved_docs: List[Any]
    sufficient: bool
    retry_count: int
    answer: str
    out_of_scope: bool
    pii_redacted: List[Dict]
    sources: List[Dict]


def scope_node(state: AgentState) -> dict:
    return {"out_of_scope": is_out_of_scope(state["original_query"])}


def router_node(state: AgentState) -> dict:
    return {
        "sub_queries": plan_sub_queries(state["original_query"]),
        "active_query": state["original_query"],
    }


def retrieve_node(state: AgentState) -> dict:
    retriever = get_retriever(state["role"])
    seen, all_docs = set(), []

    queries = state["sub_queries"] if state["retry_count"] == 0 else [state["active_query"]]

    for q in queries:
        for d in retriever.invoke(q):
            key = (d.metadata.get("source"), d.page_content[:50])
            if key not in seen:
                seen.add(key)
                all_docs.append(d)

    return {"retrieved_docs": all_docs}


def critic_node(state: AgentState) -> dict:
    context = _format_docs(state["retrieved_docs"])
    sufficient = judge_sufficiency(state["original_query"], context) if context else False
    return {"sufficient": sufficient}


def critic_router(state: AgentState) -> str:
    if state["sufficient"] or state["retry_count"] >= MAX_RETRIES:
        return "generate"
    return "retry"


def prepare_retry_node(state: AgentState) -> dict:
    return {
        "active_query": reformulate_query(state["original_query"]),
        "retry_count": state["retry_count"] + 1,
    }


def generate_node(state: AgentState) -> dict:
    context = _format_docs(state["retrieved_docs"])
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.1)
    chain = PROMPT | llm | StrOutputParser()

    if not context:
        answer = "I don't have access to information that answers this question."
    else:
        answer = chain.invoke({"context": context, "question": state["original_query"]})

    sources = [
        {"source": d.metadata.get("source"), "role": d.metadata.get("role")}
        for d in state["retrieved_docs"]
    ]
    return {"answer": answer, "sources": sources}


def guardrail_node(state: AgentState) -> dict:
    result = redact_pii(state["answer"])
    return {"answer": result["redacted_text"], "pii_redacted": result["entities_found"]}


def scope_router(state: AgentState) -> str:
    return "blocked" if state["out_of_scope"] else "continue"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("scope_check", scope_node)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("critic", critic_node)
    graph.add_node("prepare_retry", prepare_retry_node)
    graph.add_node("generate", generate_node)
    graph.add_node("guardrail", guardrail_node)

    graph.set_entry_point("scope_check")
    graph.add_conditional_edges(
        "scope_check", scope_router, {"blocked": END, "continue": "router"}
    )
    graph.add_edge("router", "retrieve")
    graph.add_edge("retrieve", "critic")
    graph.add_conditional_edges(
        "critic", critic_router, {"retry": "prepare_retry", "generate": "generate"}
    )
    graph.add_edge("prepare_retry", "retrieve")
    graph.add_edge("generate", "guardrail")
    graph.add_edge("guardrail", END)

    return graph.compile()


_compiled_graph = build_graph()


def run_agent(role: str, query: str) -> dict:
    initial_state = {
        "role": role,
        "original_query": query,
        "active_query": query,
        "sub_queries": [],
        "retrieved_docs": [],
        "sufficient": False,
        "retry_count": 0,
        "answer": "",
        "out_of_scope": False,
        "pii_redacted": [],
        "sources": [],
    }

    final_state = _compiled_graph.invoke(initial_state)

    if final_state.get("out_of_scope"):
        return {
            "answer": "I can only answer questions about company data. That question is outside what I have access to.",
            "sources": [],
            "retrieved_count": 0,
            "out_of_scope": True,
            "pii_redacted": [],
            "sub_queries": [],
        }

    return {
        "answer": final_state["answer"],
        "sources": final_state["sources"],
        "retrieved_count": len(final_state["retrieved_docs"]),
        "out_of_scope": False,
        "pii_redacted": final_state["pii_redacted"],
        "sub_queries": final_state["sub_queries"],
    }