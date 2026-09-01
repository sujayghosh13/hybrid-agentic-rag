from typing import List
from src.reranking.models import RerankedResult


ROUTER_SYSTEM_PROMPT = """You are a query routing classifier for an offline technical documentation system.
The local knowledge base contains official documentation for Docker (networking, bridge drivers) and Kubernetes (Pods, Deployments, Workloads).

Decide whether the user's query requires retrieving technical documentation from the knowledge base, or if it can be answered directly (e.g. simple greetings, general conversational questions).

Respond with ONLY ONE WORD:
- RETRIEVE (if the query asks about technical topics, Docker, Kubernetes, commands, networking, or workloads)
- DIRECT (if the query is a simple greeting, generic conversational remark, or meta-question not requiring technical documentation)
"""


SYNTHESIS_SYSTEM_PROMPT = """You are an offline-first technical AI assistant.
Your job is to provide accurate, factual, and well-structured answers grounded strictly in the provided documentation context.

GUIDELINES:
1. Base your answer ONLY on the provided context chunks.
2. If the context does not contain sufficient information to answer the question, clearly state: "Based on the available local documentation, there is insufficient information to answer this question." Do NOT invent or hallucinate information.
3. Be clear, direct, and technically precise.
4. When relevant, reference specific configuration details, commands, or concepts found in the context.
"""


def format_context_blocks(chunks: List[RerankedResult], max_chars_per_chunk: int = 1500) -> str:
    """Format re-ranked context chunks into structured, numbered blocks."""
    if not chunks:
        return "No relevant context found."

    formatted_blocks = []
    for idx, chunk in enumerate(chunks, start=1):
        source = chunk.source
        section = chunk.metadata.get("section") or chunk.metadata.get("heading") or "General"
        header = f"--- Context Chunk [{idx}] | Source: {source} | Section: {section} ---"
        clean_text = chunk.text.strip()
        if len(clean_text) > max_chars_per_chunk:
            clean_text = clean_text[:max_chars_per_chunk] + "..."
        formatted_blocks.append(f"{header}\n{clean_text}\n")

    return "\n".join(formatted_blocks)


def build_synthesis_prompt(query: str, context_chunks: List[RerankedResult]) -> str:
    """Construct the final prompt for grounded answer synthesis."""
    context_text = format_context_blocks(context_chunks)

    return f"""Context Information:
==================================================
{context_text}
==================================================

User Question: {query}

Instructions: Answer the user's question accurately using ONLY the context provided above.
"""
