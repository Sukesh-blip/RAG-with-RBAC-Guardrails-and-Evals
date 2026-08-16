"""
LangGraph state machine wiring together: scope check, query planning,
RBAC-filtered retrieval, retrieval self-correction, generation, PII
guardrails, and token/cost tracking across the whole request.
"""

import os
from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

from rag.chain import get_retriever, _format_docs, PROMPT
from guardrails.scope_guard import is_out_of_scope
from guardrails.pii_guard import redact_pii
from agents.router import plan_sub_queries
from agents.critic import judge_sufficiency, reformulate_query, MAX_RETRIES
from monitoring.cost_tracker import TokenUsageCallback, log_usage

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


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
    token_tracker: Optional[Any]


def _cb(state: AgentState):
    tracker = state.get("token_tracker")
    return [tracker] if tracker else None


def scope_node(state: AgentState) -> dict:
    return {"out_of_scope": is_out_of_scope(state["original_query"], callbacks=_cb(state))}


def router_node(state: AgentState) -> dict:
    return {
        "sub_queries": plan_sub_queries(state["original_query"], callbacks=_cb(state)),
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
    sufficient = judge_sufficiency(state["original_query"], context, callbacks=_cb(state)) if context else False
    return {"sufficient": sufficient}


def critic_router(state: AgentState) -> str:
    if state["sufficient"] or state["retry_count"] >= MAX_RETRIES:
        return "generate"
    return "retry"


def prepare_retry_node(state: AgentState) -> dict:
    return {
        "active_query": reformulate_query(state["original_query"], callbacks=_cb(state)),
        "retry_count": state["retry_count"] + 1,
    }


def generate_node(state: AgentState) -> dict:
    context = _format_docs(state["retrieved_docs"])
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.1)
    chain = PROMPT | llm | StrOutputParser()

    if not context:
        answer = "I don't have access to information that answers this question."
    else:
        config = {"callbacks": _cb(state)} if _cb(state) else {}
        answer = chain.invoke({"context": context, "question": state["original_query"]}, config=config)

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


def _initial_state(role: str, query: str, tracker) -> AgentState:
    return {
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
        "token_tracker": tracker,
    }


def run_agent(role: str, query: str) -> dict:
    tracker = TokenUsageCallback()
    final_state = _compiled_graph.invoke(_initial_state(role, query, tracker))

    usage_summary = tracker.summary()
    alerts = log_usage(role, query, usage_summary)

    if final_state.get("out_of_scope"):
        return {
            "answer": "I can only answer questions about company data. That question is outside what I have access to.",
            "sources": [],
            "retrieved_count": 0,
            "out_of_scope": True,
            "pii_redacted": [],
            "sub_queries": [],
            "token_usage": usage_summary,
            "cost_alerts": alerts,
        }

    return {
        "answer": final_state["answer"],
        "sources": final_state["sources"],
        "retrieved_count": len(final_state["retrieved_docs"]),
        "out_of_scope": False,
        "pii_redacted": final_state["pii_redacted"],
        "sub_queries": final_state["sub_queries"],
        "token_usage": usage_summary,
        "cost_alerts": alerts,
    }


def run_agent_with_context(role: str, query: str) -> dict:
    """
    Same as run_agent, but also returns raw retrieved context text and the
    role tag each chunk carries. Used by the eval suite.
    """
    tracker = TokenUsageCallback()
    final_state = _compiled_graph.invoke(_initial_state(role, query, tracker))

    if final_state.get("out_of_scope"):
        return {
            "answer": "I can only answer questions about company data. That question is outside what I have access to.",
            "contexts": [],
            "context_roles": [],
            "out_of_scope": True,
        }

    contexts = [d.page_content for d in final_state["retrieved_docs"]]
    context_roles = [d.metadata.get("role", "unknown") for d in final_state["retrieved_docs"]]

    return {
        "answer": final_state["answer"],
        "contexts": contexts,
        "context_roles": context_roles,
        "out_of_scope": False,
    }