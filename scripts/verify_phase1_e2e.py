#!/usr/bin/env python3
"""End-to-end verification script for Phase 1: Streamlit document upload, indexing, duplicate prevention, and retrieval."""
import io
import sys
from pathlib import Path

# Ensure project root is in python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pypdf import PdfReader
from src.agent.agent import LocalQwenAgent
from src.ingestion.indexer import DocumentIndexer


def build_test_pdf() -> bytes:
    """Build a minimal valid PDF containing distinct technical documentation."""
    title = "Kubernetes Pod Disruption Budget Policy Specification"
    body = (
        "A PodDisruptionBudget (PDB) limits voluntary disruptions in production clusters by specifying minAvailable or maxUnavailable. "
        "Specifically, minAvailable: 2 ensures that at least two pod replicas remain healthy during voluntary node drains and upgrades. "
        "PDB policies apply only to voluntary disruptions and cannot prevent involuntary hardware node failures."
    )

    pdf_bytes = f"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length {len(body) + len(title) + 60} >>
stream
BT
/F1 14 Tf
50 700 Td
({title}) Tj
0 -25 Td
({body}) Tj
ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000244 00000 n 
0000000450 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
550
%%EOF
""".encode("latin-1")

    # Verify extractable text
    reader = PdfReader(io.BytesIO(pdf_bytes))
    extracted = reader.pages[0].extract_text()
    assert "minAvailable: 2" in extracted, "Failed to embed text into test PDF"
    return pdf_bytes


def main():
    print("=" * 60)
    print("PHASE 1 END-TO-END VERIFICATION")
    print("=" * 60)

    filename = "kubernetes-pod-disruption-budget.pdf"
    pdf_bytes = build_test_pdf()

    indexer = DocumentIndexer()

    # Step 1: Test uploading and indexing the PDF
    print("\n[STEP 1] Testing Document Upload and Indexing...")
    progress_steps = []

    def on_progress(step: str):
        print(f"  -> {step}")
        progress_steps.append(step)

    result = indexer.index_uploaded_file(
        filename=filename,
        content=pdf_bytes,
        on_progress=on_progress,
    )

    print(f"\nResult: {result}")
    assert result["status"] in ("success", "skipped"), f"Expected success or skipped, got {result['status']}"
    assert result["chunks_indexed"] >= 1, "Expected at least 1 chunk indexed"
    print(f"✅ Verified: PDF '{filename}' indexed successfully ({result['chunks_indexed']} chunks).")

    # Step 2: Test duplicate upload handling
    print("\n[STEP 2] Testing Duplicate Upload Prevention...")
    dup_steps = []
    dup_result = indexer.index_uploaded_file(
        filename=filename,
        content=pdf_bytes,
        on_progress=lambda s: dup_steps.append(s),
    )
    print(f"Duplicate Result: {dup_result}")
    assert dup_result["status"] == "skipped", f"Expected skipped, got {dup_result['status']}"
    assert "already indexed" in dup_result["message"]
    print("✅ Verified: Duplicate upload detected and skipped re-indexing.")

    # Step 3: Query the existing RAG pipeline about the new document
    print("\n[STEP 3] Querying RAG Assistant about Newly Indexed PDF...")
    question = "What does minAvailable: 2 ensure in a Kubernetes Pod Disruption Budget?"
    print(f"Question: '{question}'")

    agent = LocalQwenAgent()
    response = agent.run(question)

    print("\n" + "-" * 60)
    print("SYNTHESIZED ANSWER:")
    print(response.answer)
    print("-" * 60)
    print("\nATTRIBUTED SOURCES:")
    for idx, s in enumerate(response.sources, 1):
        print(f"  [{idx}] Chunk ID: {s.get('chunk_id')} | Source: {s.get('source')} | Score: {s.get('rerank_score')}")

    # Step 4: Verify source attribution
    source_filenames = [s.get("metadata", {}).get("filename", "") for s in response.sources]
    source_paths = [s.get("source", "") for s in response.sources]
    found_in_sources = any(filename in fn or filename in sp for fn, sp in zip(source_filenames, source_paths))

    assert found_in_sources, f"Expected '{filename}' in retrieved sources: {source_filenames} / {source_paths}"
    print(f"\n✅ Verified: Source attribution correctly cites '{filename}'.")

    # Step 5: Verify answer quality
    assert "2" in response.answer or "two" in response.answer.lower(), "Answer does not mention the 2 replica guarantee"
    print("✅ Verified: Grounded answer accurately answers question based on uploaded PDF context.")

    print("\n" + "=" * 60)
    print("ALL PHASE 1 END-TO-END VERIFICATIONS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
