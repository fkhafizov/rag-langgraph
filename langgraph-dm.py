"""
╔══════════════════════════════════════════════════════════════╗
║         LangGraph Demo Suite  —  4 Core Patterns            ║
║                                                              ║
║  Demo 1: Simple 3-node linear pipeline                       ║
║  Demo 2: RAG Q&A agent (retrieve → answer → fallback)        ║
║  Demo 3: ReAct agent with tools (calculator + search)        ║
║  Demo 4: Reflection loop (generate → critique → revise)      ║
║                                                              ║
║  Requirements:  pip install langgraph langchain-openai       ║
║  Run:           python langgraph_demo.py                     ║
║  Env:           OPENAI_API_KEY must be set                   ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import operator
from typing import TypedDict, Annotated, List, Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent

# ── Shared LLM (gpt-4o-mini: fast + cheap for demos) ──────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# ── Pretty-print helpers ───────────────────────────────────────────────────────
W = 64

def banner(title: str):
    print(f"\n{'═' * W}")
    print(f"  {title}")
    print(f"{'═' * W}")

def node_log(name: str, text: str, max_chars: int = 400):
    print(f"\n  ▶ [{name}]")
    snippet = text.strip().replace("\n", " ")
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars] + " …"
    print(f"    {snippet}")

def divider():
    print(f"  {'─' * (W - 2)}")

def result(label: str, text: str):
    print(f"\n  ✅ {label}")
    print(f"  {text.strip()}")

# ══════════════════════════════════════════════════════════════════════════════
# DEMO 1 — Simple 3-node Linear Pipeline
# ──────────────────────────────────────────────────────────────────────────────
# Pattern:  START → outline → draft → summarize → END
# Purpose:  Show StateGraph basics — typed state, nodes, linear edges.
# Analogy:  Like a DAG pipeline in scikit-learn or Spark, but nodes are LLM calls.
# ══════════════════════════════════════════════════════════════════════════════

class PipelineState(TypedDict):
    topic:   str
    outline: str
    draft:   str
    summary: str


def node_outline(state: PipelineState) -> PipelineState:
    """Node 1: Generate a 3-bullet outline for the topic."""
    prompt = f"Write a 3-bullet outline for a short blog post about: {state['topic']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    node_log("outline", response.content)
    return {"outline": response.content}


def node_draft(state: PipelineState) -> PipelineState:
    """Node 2: Expand the outline into a short draft (2-3 sentences per bullet)."""
    prompt = (
        f"Expand this outline into a short blog post draft.\n\n"
        f"Topic: {state['topic']}\nOutline:\n{state['outline']}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    node_log("draft", response.content)
    return {"draft": response.content}


def node_summarize(state: PipelineState) -> PipelineState:
    """Node 3: Compress the draft into a single tweet-length summary."""
    prompt = f"Summarize this blog post in one tweet (max 280 chars):\n\n{state['draft']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    node_log("summarize", response.content)
    return {"summary": response.content}


def run_demo_1():
    banner("DEMO 1 — Simple 3-Node Linear Pipeline")
    print("  Pattern:  START → outline → draft → summarize → END")
    print("  Topic:    LangGraph for enterprise AI")
    divider()

    graph = StateGraph(PipelineState)
    graph.add_node("outline",   node_outline)
    graph.add_node("draft",     node_draft)
    graph.add_node("summarize", node_summarize)

    graph.set_entry_point("outline")
    graph.add_edge("outline",   "draft")
    graph.add_edge("draft",     "summarize")
    graph.add_edge("summarize", END)

    app = graph.compile()

    final = app.invoke({"topic": "LangGraph for enterprise AI", "outline": "",
                        "draft": "", "summary": ""})
    result("Final tweet-length summary:", final["summary"])


# ══════════════════════════════════════════════════════════════════════════════
# DEMO 2 — RAG Q&A Agent  (retrieve → answer, with fallback)
# ──────────────────────────────────────────────────────────────────────────────
# Pattern:  START → retrieve → [conditional] → answer | fallback → END
# Purpose:  Show conditional edges — routing based on state inspection.
# Mock KB:  In-memory dict simulates a vector DB retrieval step.
# ══════════════════════════════════════════════════════════════════════════════

# Simulated knowledge base (replaces FAISS / Pinecone for the demo)
KNOWLEDGE_BASE = {
    "langgraph":  "LangGraph is a library for building stateful, multi-actor LLM applications "
                  "using a graph abstraction. Nodes are Python functions; edges define transitions; "
                  "a shared TypedDict holds state across all nodes. It supports cycles, enabling "
                  "agent loops such as plan-act-observe-reflect.",
    "llmops":     "LLMOps covers the full lifecycle of LLM-powered applications: prompt engineering, "
                  "RAG pipeline design, fine-tuning (LoRA/QLoRA), evaluation (RAGAS, DeepEval), "
                  "deployment (Bedrock, Vertex AI), observability (LangSmith), and guardrails.",
    "rag":        "Retrieval-Augmented Generation (RAG) grounds LLM answers in external knowledge. "
                  "Steps: chunk documents → embed → store in vector DB → retrieve top-k chunks → "
                  "pass as context to LLM → generate grounded answer.",
    "infosys":    "Infosys Topaz is Infosys's AI-first suite of services built around generative AI. "
                  "It includes agentic AI workflows, RAG-based enterprise search, and LLMOps tooling "
                  "for large-scale enterprise clients.",
}


def mock_retrieve(question: str) -> str:
    """Keyword-based mock retrieval — simulates a vector DB nearest-neighbor search."""
    question_lower = question.lower()
    for keyword, passage in KNOWLEDGE_BASE.items():
        if keyword in question_lower:
            return passage
    return ""


class RAGState(TypedDict):
    question: str
    context:  str
    answer:   str


def node_retrieve(state: RAGState) -> RAGState:
    """Retrieve relevant context from the mock knowledge base."""
    context = mock_retrieve(state["question"])
    node_log("retrieve", context if context else "(no relevant context found)")
    return {"context": context}


def node_answer(state: RAGState) -> RAGState:
    """Generate a grounded answer using retrieved context."""
    prompt = (
        f"Answer the question using ONLY the provided context. "
        f"Do not add information from outside the context.\n\n"
        f"Context: {state['context']}\n\n"
        f"Question: {state['question']}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    node_log("answer", response.content)
    return {"answer": response.content}


def node_fallback(state: RAGState) -> RAGState:
    """Fallback when no context was retrieved."""
    answer = (
        f"I couldn't find relevant information in the knowledge base to answer: "
        f"'{state['question']}'. Please rephrase or ask about LangGraph, LLMOps, RAG, or Infosys."
    )
    node_log("fallback", answer)
    return {"answer": answer}


def route_rag(state: RAGState) -> Literal["answer", "fallback"]:
    """Conditional edge: route based on whether context was found."""
    return "answer" if state["context"] else "fallback"


def run_demo_2():
    banner("DEMO 2 — RAG Q&A Agent with Conditional Routing")
    print("  Pattern:  START → retrieve → [route] → answer | fallback → END")
    divider()

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("answer",   node_answer)
    graph.add_node("fallback", node_fallback)

    graph.set_entry_point("retrieve")
    graph.add_conditional_edges("retrieve", route_rag)
    graph.add_edge("answer",   END)
    graph.add_edge("fallback", END)

    app = graph.compile()

    questions = [
        "What is LangGraph and how does it work?",   # → hits KB → answer node
        "What are the best restaurants in Dallas?",  # → misses KB → fallback node
    ]
    for q in questions:
        print(f"\n  Question: {q}")
        final = app.invoke({"question": q, "context": "", "answer": ""})
        result("Answer:", final["answer"])


# ══════════════════════════════════════════════════════════════════════════════
# DEMO 3 — ReAct Agent with Tools (calculator + mock web search)
# ──────────────────────────────────────────────────────────────────────────────
# Pattern:  Uses langgraph.prebuilt.create_react_agent
#           LLM decides: call a tool or produce final answer
#           Loop:  think → [tool?] → observe → think → … → answer
# ══════════════════════════════════════════════════════════════════════════════

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely. Input: a valid Python math expression string."""
    try:
        allowed = set("0123456789+-*/.() **eE")
        if not all(c in allowed for c in expression.replace(" ", "")):
            return "Error: only numeric expressions allowed"
        return str(eval(expression))  # noqa: S307 — demo only, expression is sanitised
    except Exception as e:
        return f"Error: {e}"


@tool
def web_search(query: str) -> str:
    """Search the web for current information. Returns a short summary of results."""
    # Mock responses — replace with Tavily / SerpAPI in production
    mock_db = {
        "langgraph latest version": "LangGraph latest stable release is 0.2.x (2025). "
                                    "Key features: streaming, subgraphs, human-in-the-loop interrupts.",
        "infosys revenue 2024":     "Infosys reported annual revenue of approximately $18.6 billion "
                                    "for fiscal year 2024, with continued growth in digital services.",
        "gpt-4o-mini cost":         "GPT-4o-mini costs $0.15 per 1M input tokens and $0.60 per 1M "
                                    "output tokens as of 2025.",
    }
    query_lower = query.lower()
    for key, val in mock_db.items():
        if any(word in query_lower for word in key.split()):
            return val
    return f"No results found for '{query}'. Try a more specific query."


def run_demo_3():
    banner("DEMO 3 — ReAct Agent with Tools")
    print("  Pattern:  think → [tool call?] → observe → think → … → answer")
    print("  Tools:    calculator, web_search (mock)")
    divider()

    tools = [calculator, web_search]
    agent = create_react_agent(llm, tools)

    queries = [
        "What is 2 ** 32 + 17 * 42?",
        "What is the latest version of LangGraph, and what is 1024 * 1024?",
    ]
    for q in queries:
        print(f"\n  Query: {q}")
        response = agent.invoke({"messages": [HumanMessage(content=q)]})
        final_msg = response["messages"][-1].content
        result("Final answer:", final_msg)


# ══════════════════════════════════════════════════════════════════════════════
# DEMO 4 — Reflection Loop (generate → critique → revise)
# ──────────────────────────────────────────────────────────────────────────────
# Pattern:  START → generate → critique → [route] → revise ──┐
#                                                  └──────────┘ (loop)
#                                         └──(done)──→ END
# Purpose:  Show back-edges and loop termination via step counter.
# ══════════════════════════════════════════════════════════════════════════════

class ReflectState(TypedDict):
    task:       str
    draft:      str
    critique:   str
    iteration:  int
    final:      str


MAX_ITERATIONS = 2   # loop at most twice then stop


def node_generate(state: ReflectState) -> ReflectState:
    """Generate or revise a draft based on the task (and prior critique if any)."""
    if state["critique"]:
        prompt = (
            f"You are revising a draft based on critique.\n\n"
            f"Task: {state['task']}\n"
            f"Previous draft:\n{state['draft']}\n\n"
            f"Critique to address:\n{state['critique']}\n\n"
            f"Write an improved version. Be concise (3-4 sentences)."
        )
    else:
        prompt = (
            f"Write a first draft (3-4 sentences) for the following task:\n{state['task']}"
        )
    response = llm.invoke([HumanMessage(content=prompt)])
    iteration = state["iteration"] + 1
    node_log(f"generate (iteration {iteration})", response.content)
    return {"draft": response.content, "iteration": iteration, "critique": ""}


def node_critique(state: ReflectState) -> ReflectState:
    """Critique the current draft — identify specific weaknesses."""
    prompt = (
        f"You are a critical editor. Review this draft and list 2 specific weaknesses "
        f"or improvements needed. Be concise and direct.\n\n"
        f"Task: {state['task']}\n\nDraft:\n{state['draft']}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    node_log(f"critique (iteration {state['iteration']})", response.content)
    return {"critique": response.content}


def node_finalize(state: ReflectState) -> ReflectState:
    """Accept current draft as final output."""
    node_log("finalize", f"Accepting draft after {state['iteration']} iteration(s).")
    return {"final": state["draft"]}


def route_reflect(state: ReflectState) -> Literal["generate", "finalize"]:
    """Loop back to generate (revise) or finalize based on iteration count."""
    if state["iteration"] < MAX_ITERATIONS:
        return "generate"   # revise based on critique
    return "finalize"       # done — accept current draft


def run_demo_4():
    banner("DEMO 4 — Reflection Loop (Generate → Critique → Revise)")
    print(f"  Pattern:  generate ↔ critique (max {MAX_ITERATIONS} iterations) → finalize → END")
    divider()

    graph = StateGraph(ReflectState)
    graph.add_node("generate", node_generate)
    graph.add_node("critique", node_critique)
    graph.add_node("finalize", node_finalize)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "critique")
    graph.add_conditional_edges("critique", route_reflect)
    graph.add_edge("finalize", END)

    app = graph.compile()

    task = (
        "Explain why LangGraph is better than a simple LangChain sequential chain "
        "for building production agentic AI systems."
    )
    print(f"\n  Task: {task}")

    final = app.invoke({
        "task":      task,
        "draft":     "",
        "critique":  "",
        "iteration": 0,
        "final":     "",
    })
    result(f"Final draft (after {final['iteration']} iterations):", final["final"])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("  export OPENAI_API_KEY='sk-...'")
        raise SystemExit(1)

    print("\n" + "█" * W)
    print("█" + " " * (W - 2) + "█")
    print("█" + "  LangGraph Demo Suite — 4 Core Patterns  ".center(W - 2) + "█")
    print("█" + " " * (W - 2) + "█")
    print("█" * W)

    run_demo_1()
    run_demo_2()
    run_demo_3()
    run_demo_4()

    banner("ALL DEMOS COMPLETE")
    print("  Patterns covered:")
    print("   1. Linear pipeline          — StateGraph + linear add_edge")
    print("   2. RAG with conditional routing — add_conditional_edges on retrieval result")
    print("   3. ReAct tool-calling agent — create_react_agent (prebuilt)")
    print("   4. Reflection loop          — back-edges + iteration counter as stop condition")
    print()
