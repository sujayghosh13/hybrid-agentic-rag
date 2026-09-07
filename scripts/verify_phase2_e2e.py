"""Live end-to-end verification script for Phase 2: Conversational Memory."""

import sys
import json
import logging

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Phase2Verification")

def run_e2e_verification():
    from src.agent.agent import LocalQwenAgent
    from src.agent.tools import HybridSearchTool, RerankTool
    from src.retrieval.hybrid import HybridRetriever
    from src.reranking.cross_encoder import CrossEncoderReranker
    from src.config import settings

    print("\n" + "=" * 60)
    print("PHASE 2 LIVE END-TO-END VERIFICATION: CONVERSATIONAL MEMORY")
    print("=" * 60)
    print(f"Conversational Memory Enabled: {settings.conversational_memory_enabled}")
    print(f"Conversational Memory Turns:   {settings.conversational_memory_turns}")
    print(f"Query Rewriter Enabled:        {settings.query_rewriter_enabled}")
    print(f"Ollama Model:                  {settings.ollama_model}")

    # Initialize live agent components
    print("\n[1/3] Initializing live HybridRetriever, CrossEncoder, and LocalQwenAgent...")
    retriever = HybridRetriever()
    reranker = CrossEncoderReranker()
    search_tool = HybridSearchTool(retriever=retriever)
    rerank_tool = RerankTool(reranker=reranker)
    agent = LocalQwenAgent(
        hybrid_search_tool=search_tool,
        rerank_tool=rerank_tool,
    )
    print("✅ Live agent successfully initialized.")

    # --- Turn 1: Standalone Question ---
    q1 = "What is a Kubernetes Pod?"
    print(f"\n[2/3] Executing Turn 1 (Standalone): '{q1}'")
    resp1 = agent.run(q1)
    print(f"Turn 1 Query:     {resp1.query}")
    print(f"Turn 1 Answer:    {resp1.answer[:150]}...")
    print(f"Turn 1 Sources:   {len(resp1.sources)} sources attributed")
    for s in resp1.sources[:2]:
        print(f"  - [{s.get('chunk_id')}] {s.get('source')} (Score: {s.get('rerank_score')})")

    # --- Turn 2: Follow-up Question with Conversational History ---
    q2 = "What about its lifecycle?"
    print(f"\n[3/3] Executing Turn 2 (Follow-up): '{q2}' with Turn 1 in history")
    history = [
        {"role": "user", "content": resp1.query},
        {"role": "assistant", "content": resp1.answer[:250]},
    ]
    resp2 = agent.run(q2, chat_history=history)

    print(f"\nTurn 2 Query:            {resp2.query}")
    print(f"Turn 2 Rewritten Queries: {resp2.rewritten_queries}")
    print(f"Turn 2 Retrieval Query:   {resp2.metadata.get('retrieval_query')}")
    print(f"Turn 2 Answer:           {resp2.answer[:180]}...")
    print(f"Turn 2 Sources:          {len(resp2.sources)} sources attributed")
    for s in resp2.sources[:2]:
        print(f"  - [{s.get('chunk_id')}] {s.get('source')} (Score: {s.get('rerank_score')})")

    # Verifications
    assert resp2.query == "What about its lifecycle?", "Response query must match original user input"
    assert len(resp2.rewritten_queries) > 0, "Expected at least one rewritten query"
    rewritten_q = resp2.rewritten_queries[0]
    print(f"\nContextual Reformulation Evaluation:")
    print(f"  Raw Query:        '{q2}'")
    print(f"  Reformulated:     '{rewritten_q}'")
    
    has_pod_context = "pod" in rewritten_q.lower() or "kubernetes" in rewritten_q.lower()
    has_lifecycle = "lifecycle" in rewritten_q.lower()
    print(f"  Contains Pod/K8s: {has_pod_context}")
    print(f"  Contains lifecycle: {has_lifecycle}")
    assert has_pod_context, "Reformulated query must contain context entity ('pod' or 'kubernetes')"
    assert has_lifecycle, "Reformulated query must preserve core concept ('lifecycle')"
    assert len(resp2.sources) > 0, "Response must include source attribution"

    print("\n" + "=" * 60)
    print("✅ PHASE 2 LIVE END-TO-END VERIFICATION PASSED SUCCESSFULLY!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_e2e_verification()
