from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ai_advocate.infrastructure.graph.nodes import (
    GraphRuntime,
    generate,
    grade_documents,
    retrieve,
    rewrite_query,
    route_after_grade,
)
from ai_advocate.infrastructure.graph.state import RAGGraphState


def build_rag_graph(runtime: GraphRuntime):
    graph = StateGraph(RAGGraphState)

    graph.add_node("rewrite", lambda s: rewrite_query(s, runtime))
    graph.add_node("retrieve", lambda s: retrieve(s, runtime))
    graph.add_node("grade", lambda s: grade_documents(s, runtime))
    graph.add_node("generate", lambda s: generate(s, runtime))

    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "grade")

    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "rewrite": "rewrite",
            "generate": "generate",
        },
    )

    graph.add_edge("generate", END)

    return graph.compile()
