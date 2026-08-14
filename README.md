<div align="center">

# 🛡️ Agentic RAG Chatbot with RBAC, Guardrails & Eval-Gated Deployment

**A production-grade internal knowledge assistant — where retrieval respects who's asking.**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/Agents-LangGraph-1C3C3C.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Development-orange.svg)]()

[Problem](#-problem-statement) •
[Architecture](#-architecture) •
[Tech Stack](#-tech-stack) •
[Setup](#-setup) •
[Roadmap](#-roadmap) •
[Demo](#-demo)

</div>

---

## 📌 Problem Statement

Internal company chatbots built on standard RAG treat every employee the same —
anyone with bot access can retrieve **any** indexed document, because retrieval has
no concept of *who is asking*. That's a real risk in practice: if payroll records,
financial reports, and marketing spend all live in one vector index, an intern could
ask *"what's the CFO's salary?"* and get an answer — simply because the document was
relevant, not because they were authorized to see it.

**Who this is for:**

| Role | Access |
|---|---|
| 💰 Finance | Financial reports, marketing expense data |
| 👥 HR | Employee records, payroll, HR policy |
| 👔 C-level | Full access across all company data |
| 🙋 Any employee | General company docs — out-of-scope questions are declined, not hallucinated |

**Why naive RAG fails here:**
1. Access control is bolted onto the UI, not enforced at retrieval — a clever prompt can still surface restricted content
2. No guardrails for PII exposure or out-of-scope queries
3. No automated check that a pipeline change hasn't silently degraded answer quality
4. Token cost is invisible until the bill arrives

**What this project builds:** access control enforced *at the retrieval layer*, an
agentic planning + self-correction loop instead of blind single-pass retrieval,
automated PII/guardrail protection, eval-gated CI on every push, and live token
cost tracking — all on a fully open-source, zero-cost stack.

---

## 🏗️ Architecture

```mermaid
flowchart TD
    UI["Streamlit UI<br/>(role-select chat)"] -->|HTTP| API["FastAPI<br/>/chat · /ingest · /eval/status · /cost/usage"]
    API --> Router["Router / Planner Node<br/>(decomposes query, plans sub-queries)"]
    Router --> Retrieve["Retrieve Node<br/>(RBAC-filtered Chroma search)"]
    Retrieve --> Critic["Critic Node<br/>(is context sufficient?)"]
    Critic -->|retry| Retrieve
    Critic -->|sufficient| Guardrail["Guardrail Node<br/>(PII redact, scope check)"]
    Guardrail --> Answer["Final Answer"]

    Retrieve -.-> Chroma[(ChromaDB)]
    Answer -.-> Groq["Groq LLM<br/>(Llama 3.3 / GPT-OSS)"]
    Guardrail -.-> Presidio["Presidio<br/>(PII guard)"]
```

**Key design decision:** RBAC is enforced at the **retriever**, not the prompt.
Every chunk in Chroma carries a `role` metadata tag at ingestion time; the retriever
filters on that tag *before* anything reaches the LLM — so there's no context for
the model to leak in the first place.

**Key design decision:** RBAC is enforced at the **retriever**, not the prompt.
Every chunk in Chroma carries a `role` metadata tag at ingestion time; the retriever
filters on that tag *before* anything reaches the LLM — so there's no context for
the model to leak in the first place.

---

## ⚙️ Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Backend | **FastAPI** | Async, typed, production-realistic API layer |
| Orchestration | **LangChain** + **LangGraph** | LCEL for the base chain, LangGraph for agentic routing/self-correction |
| LLM | **Groq** (Llama 3.3 70B / GPT-OSS) | Free-tier, fast inference, no vendor lock-in |
| Vector DB | **ChromaDB** | Embedded, zero infra, role-metadata filtering at query time |
| Embeddings | **sentence-transformers** | Local, no embedding API cost |
| Guardrails | **Presidio** + LLM-based scope classifier | PII detection/redaction + out-of-scope refusal |
| Eval | **Ragas** | Faithfulness, answer relevancy, context precision |
| Monitoring | **LangSmith** | Full trace visibility into every agent run |
| CI | **GitHub Actions** | Eval suite runs on every push — catches regressions before they ship |
| Frontend | **Streamlit** | Role-select chat interface |
| Deploy | **Render** + **Streamlit Community Cloud** | Free-tier, zero cloud spend |

---

## 📂 Repo Structure

```
rag-rbac-chatbot/
│
├── ingestion/          Chunking, embedding, upsert to Chroma
├── rbac/                Role definitions, access filter logic
├── guardrails/          PII redaction, out-of-scope classifier
├── rag/                 Retriever, prompts, base LCEL chain
├── agents/               LangGraph router + retrieval-critic nodes
├── app/                  FastAPI backend + Streamlit frontend
├── evals/                Ragas test set + eval runner
├── monitoring/           Token/cost logger
├── data/                 Synthetic role-tagged dataset (18 docs)
│   ├── finance/
│   ├── hr/
│   ├── general/
│   └── mixed/
│
├── .github/workflows/    CI eval pipeline
├── requirements.txt
├── .env.example
└── README.md
```

---

## 📊 Dataset

18 synthetic company documents, tagged by role via YAML frontmatter:

| Folder | Docs | Content |
|---|---|---|
| `data/finance/` | 5 | Financial reports, marketing expenses, budget forecast, vendor invoices, revenue breakdown |
| `data/hr/` | 5 | Payroll, leave policy, performance reviews, benefits, hiring policy |
| `data/general/` | 5 | Handbook, product roadmap, security policy, remote work, holiday calendar |
| `data/mixed/` | 3 | Cross-role docs (HR + Finance) used to stress-test agentic query decomposition |

---

## 🚀 Setup

```bash
git clone https://github.com/<your-username>/rag-rbac-chatbot.git
cd rag-rbac-chatbot

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_lg

cp .env.example .env            # add your GROQ_API_KEY
```

---

## 🗺️ Roadmap

- [x] **Phase 0** — Repo scaffold + synthetic dataset
- [ ] **Phase 1** — Core RAG + RBAC-filtered retrieval (FastAPI `/chat`)
- [ ] **Phase 2** — Guardrails (PII redaction, out-of-scope detection)
- [ ] **Phase 3** — Agentic layer (LangGraph router + retrieval-critic)
- [ ] **Phase 4** — Eval suite + CI (Ragas + GitHub Actions)
- [ ] **Phase 5** — Cost monitoring
- [ ] **Phase 6** — Deploy (Render + Streamlit Cloud)
- [ ] **Phase 7** — Demo polish

---

## 🎥 Demo

> Proof moments will be linked here as each phase ships:
> - RBAC enforcement blocking a cross-role query
> - Guardrail catching a PII extraction attempt
> - Agent self-correcting on insufficient retrieval context
> - CI catching an intentionally introduced RBAC regression

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built by <a href="https://github.com/sukesh-blip">Sukesh Biradar</a>
</div>
