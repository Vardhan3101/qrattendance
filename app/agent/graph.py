from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from app.agent.nodes import (
    AttendanceState,
    compute_context_node,
    fetch_user_node,
    input_node,
    llm_node,
    memory_retrieval_node,
    memory_store_node,
    output_node,
    prompt_builder_node,
)
from app.services.chroma_service import ChromaService
from app.services.openai_service import OpenAIService


def build_attendance_graph(
    db: Session,
    chroma_service: ChromaService,
    openai_service: OpenAIService,
):
    graph = StateGraph(AttendanceState)

    graph.add_node("input", input_node)
    graph.add_node("fetch_user", lambda state: fetch_user_node(state, db=db))
    graph.add_node("compute_context", lambda state: compute_context_node(state, db=db))
    graph.add_node(
        "memory_retrieval",
        lambda state: memory_retrieval_node(state, chroma_service=chroma_service),
    )
    graph.add_node("prompt_builder", prompt_builder_node)
    graph.add_node("llm", lambda state: llm_node(state, openai_service=openai_service))
    graph.add_node(
        "memory_store",
        lambda state: memory_store_node(
            state,
            chroma_service=chroma_service,
            openai_service=openai_service,
        ),
    )
    graph.add_node("output", output_node)

    graph.add_edge(START, "input")
    graph.add_edge("input", "fetch_user")
    graph.add_edge("fetch_user", "compute_context")
    graph.add_edge("compute_context", "memory_retrieval")
    graph.add_edge("memory_retrieval", "prompt_builder")
    graph.add_edge("prompt_builder", "llm")
    graph.add_edge("llm", "memory_store")
    graph.add_edge("memory_store", "output")
    graph.add_edge("output", END)

    return graph.compile()
