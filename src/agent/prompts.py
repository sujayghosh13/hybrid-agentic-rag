from typing import Dict, List, Optional
from src.reranking.models import RerankedResult


ROUTER_SYSTEM_PROMPT = """You are a query routing classifier for an offline technical documentation system.
The local knowledge base contains official documentation for Docker (networking, bridge drivers) and Kubernetes (Pods, Deployments, Workloads).

Decide whether the user's query requires retrieving technical documentation from the knowledge base, or if it can be answered directly (e.g. simple greetings, general conversational questions).

Respond with ONLY ONE WORD:
- RETRIEVE (if the query asks about technical topics, Docker, Kubernetes, commands, networking, or workloads)
- DIRECT (if the query is a simple greeting, generic conversational remark, or meta-question not requiring technical documentation)
"""


REWRITE_SYSTEM_PROMPT = """You are an expert search query optimizer for technical documentation.
Your job is to convert a user question into a concise, high-signal keyword search query suitable for hybrid dense vector and BM25 retrieval.

Rules:
1. Strip out conversational phrases, polite preamble, and filler words.
2. Focus strictly on core technical terms, commands, configurations, and concepts.
3. If specific missing information is requested, produce a query targeting that specific missing aspect.
4. Output ONLY the search query keywords on a single line. Do not provide explanations or quotes.
"""


SUFFICIENCY_SYSTEM_PROMPT = """You are a strict technical verification evaluator.
Evaluate whether the provided context chunks contain sufficient factual information to answer the user question.

Respond in exactly ONE of the following formats:
- SUFFICIENT
- INSUFFICIENT: <concise missing technical concept or entity needed>

Do NOT output any additional text.
"""


SYNTHESIS_SYSTEM_PROMPT = """You are an offline-first technical AI assistant.
Your job is to provide accurate, factual, and well-structured answers grounded strictly in the provided documentation context.

GUIDELINES:
1. Base your answer ONLY on the provided context chunks.
2. If the context does not contain sufficient information to answer the question, clearly state: "Based on the available local documentation, there is insufficient information to answer this question." Do NOT invent or hallucinate information.
3. Be clear, direct, and technically precise.
4. When relevant, reference specific configuration details, commands, or concepts found in the context.
5. Multi-Document Synthesis: When answering questions that require information from multiple documents or sections, synthesize the facts coherently across the sources without making unsupported inferential leaps. Explicitly attribute or connect facts to their respective source documents or components.
6. Provide the factual answer directly and concisely. Do not repeat the question or include chain-of-thought scratchpad text.
"""


CONVERSATIONAL_REWRITE_SYSTEM_PROMPT = """You are an expert search query reformulation assistant for a technical documentation search system.
Your job is to reformulate a user's follow-up question into a complete, self-contained search query suitable for hybrid retrieval, using the recent conversation history.

Rules:
1. Resolve all pronouns, references, and ellipses (e.g., "its lifecycle", "how do I configure it", "what about networking", "show an example of this") using the entities from the conversation history.
2. If the follow-up question is already complete, standalone, or introduces a new topic (e.g. "What is Docker Swarm?"), preserve the user question as-is without adding unrelated context.
3. Strip out conversational filler, polite preambles, and conversational framing.
4. Keep the output focused on core technical terms, concepts, configurations, and commands (applicable across Docker, Kubernetes, and arbitrary uploaded technical documentation).
5. Output ONLY the standalone search query on a single line. Do NOT provide explanations, quotes, or markdown.
"""


def build_rewrite_prompt(query: str, missing_aspect: Optional[str] = None) -> str:
    """Build prompt for optimizing user query into retrieval keywords."""
    if missing_aspect:
        return (
            f"Original Question: {query}\n"
            f"Missing Information: {missing_aspect}\n"
            f"Generate a targeted search query for the missing information:"
        )
    return (
        f"Original Question: {query}\n"
        f"Optimized Search Keywords:"
    )


def build_conversational_rewrite_prompt(
    query: str,
    recent_turns: List[Dict[str, str]],
) -> str:
    """Build prompt for reformulating a follow-up question using bounded conversation history."""
    history_lines = []
    for turn in recent_turns:
        role = turn.get("role", "user").capitalize()
        content = turn.get("content", "").strip()
        history_lines.append(f"{role}: {content}")
    history_text = "\n".join(history_lines)

    return (
        f"Recent Conversation History:\n"
        f"{history_text}\n\n"
        f"Follow-up Question: {query}\n"
        f"Standalone Search Query:"
    )


def build_sufficiency_prompt(query: str, context_chunks: List[RerankedResult]) -> str:
    """Build prompt for evaluating context sufficiency."""
    context_text = format_context_blocks(context_chunks, max_chars_per_chunk=800)
    return (
        f"Context:\n{context_text}\n\n"
        f"User Question: {query}\n\n"
        f"Is the context sufficient to answer the question?"
    )


def format_context_blocks(chunks: List[RerankedResult], max_chars_per_chunk: int = 550) -> str:
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
