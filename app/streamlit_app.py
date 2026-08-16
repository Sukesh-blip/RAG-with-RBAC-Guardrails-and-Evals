"""
Streamlit frontend for the Agentic RAG RBAC Chatbot.
Calls the FastAPI backend over HTTP - no direct access to the agent/RAG
pipeline from here, matching a real internal-tool architecture.
"""

import os
import requests
import streamlit as st

API_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Company Knowledge Assistant", page_icon="🛡️", layout="centered")

st.title("🛡️ Company Knowledge Assistant")
st.caption("RBAC-filtered RAG with agentic routing, guardrails, and eval-gated deployment")

with st.sidebar:
    st.header("Session")
    role = st.selectbox(
        "Your role",
        options=["finance", "hr", "c_level", "employee"],
        help="Access is enforced at retrieval - you'll only ever see documents your role is allowed to access.",
    )
    st.divider()
    st.caption(f"Backend: `{API_URL}`")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("meta"):
            with st.expander("Details"):
                st.json(msg["meta"])

if prompt := st.chat_input("Ask a question about company data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={"role": role, "query": prompt},
                    timeout=90,
                )
                response.raise_for_status()
                data = response.json()

                st.markdown(data["answer"])

                if data.get("out_of_scope"):
                    st.info("This question was outside the scope of company data.")
                elif data.get("pii_redacted"):
                    st.warning(f"Redacted {len(data['pii_redacted'])} PII item(s) from this answer.")

                meta = {
                    "sources": data.get("sources", []),
                    "sub_queries": data.get("sub_queries", []),
                    "token_usage": data.get("token_usage", {}),
                }
                with st.expander("Details"):
                    st.json(meta)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "meta": meta,
                })

            except requests.exceptions.RequestException as e:
                error_msg = f"Couldn't reach the backend: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})