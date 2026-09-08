import json
from pathlib import Path
import subprocess

def main():
    repo_root = Path(__file__).resolve().parent.parent
    html_path = repo_root / "frontend" / "index.html"
    content = html_path.read_text(encoding="utf-8")

    real_data = {
        "question": "How does Docker bridge networking work?",
        "answer": "Docker bridge networking works by isolating containers on different networks. Containers on the default bridge network can only communicate with each other by IP addresses, unless using the --link option, which is considered legacy. On a user-defined bridge network, containers can resolve each other by name or alias.\n\nContainers on different user-defined networks cannot communicate directly; they must be on the same network or publish ports to communicate.",
        "sources": [
            {
                "chunk_id": "docker-bridge-network.html_chunk_32",
                "source": "data/raw/docker-bridge-network.html",
                "rerank_score": 3.2266,
                "metadata": {
                    "filename": "docker-bridge-network.html",
                    "doc_type": "html",
                    "section": "Bridge network driver > Usage examples > Use the default bridge network",
                    "heading": "Use the default bridge network",
                    "chunk_index": 32,
                    "total_chunks": 43,
                    "token_count": 982,
                },
                "text": "Containers connected to the same default bridge network can communicate with each other using IP addresses. Docker does not support automatic DNS resolution on the default bridge network. If you want containers to communicate with each other by container name, you must use a user-defined bridge network instead.\n\nTo publish a container's port to the Docker host, use the -p or --publish flag...",
            },
            {
                "chunk_id": "docker-networking.html_chunk_17",
                "source": "data/raw/docker-networking.html",
                "rerank_score": 2.7127,
                "metadata": {
                    "filename": "docker-networking.html",
                    "doc_type": "html",
                    "section": "Networking overview > User-defined networks > Connecting to multiple networks",
                    "heading": "Connecting to multiple networks",
                    "chunk_index": 17,
                    "total_chunks": 25,
                    "token_count": 525,
                },
                "text": "Connecting a container to a network can be compared to connecting an Ethernet cable to a physical host. Just as a host can be connected to multiple Ethernet networks, a container can be connected to multiple Docker networks...\n\nWhen you create or run a container using docker create or docker run, all ports of containers on bridge networks are accessible from the Docker host and other containers connected to the same network...",
            },
        ],
        "orchestration": {
            "retrieval_needed": True,
            "hops_executed": 1,
            "final_evidence_grade": "GOOD",
            "is_corrected": False,
            "rewritten_queries": ["How does Docker bridge networking work?"],
        },
        "performance": {
            "total_latency_ms": 11252.21,
        },
    }

    chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    out_dir = repo_root / "docs" / "screenshots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 01_main_interface.png: Landing view with system status & prompt
    print("Capturing 01_main_interface.png...")
    subprocess.run([
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--window-size=1440,900",
        f"--screenshot={out_dir / '01_main_interface.png'}",
        f"file:///{html_path.resolve()}",
    ], check=True)

    # 2. 02_rag_answer_sources.png: Grounded Answer & Source Attribution Cards
    print("Capturing 02_rag_answer_sources.png...")
    js_2 = """
    <script>
    window.addEventListener('DOMContentLoaded', () => {
        document.getElementById('q').value = %s;
        renderResult(%s, %s);
        const cards = document.querySelectorAll('.source-card');
        cards.forEach(c => c.classList.add('open'));
    });
    </script>
    """ % (json.dumps(real_data["question"]), json.dumps(real_data), json.dumps(real_data["question"]))

    tmp_2 = out_dir / "_tmp_2.html"
    tmp_2.write_text(content.replace("</body>", js_2 + "</body>"), encoding="utf-8")
    subprocess.run([
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--window-size=1440,1400",
        f"--screenshot={out_dir / '02_rag_answer_sources.png'}",
        f"file:///{tmp_2.resolve()}",
    ], check=True)
    tmp_2.unlink()

    # 3. 03_execution_details.png: Focus on Stat Strip, Trace Log & CRAG Grading
    print("Capturing 03_execution_details.png...")
    # Hide hero intro to place execution details & trace front-and-center
    custom_style = """
    <style>
    .hero { display: none !important; }
    .prompt-card { margin-top: 10px !important; }
    </style>
    """
    js_3 = """
    <script>
    window.addEventListener('DOMContentLoaded', () => {
        document.getElementById('q').value = %s;
        renderResult(%s, %s);
        const trace = document.querySelector('details.trace');
        if (trace) trace.open = true;
        const cards = document.querySelectorAll('.source-card');
        if (cards.length > 0) cards[0].classList.add('open');
    });
    </script>
    """ % (json.dumps(real_data["question"]), json.dumps(real_data), json.dumps(real_data["question"]))

    tmp_3 = out_dir / "_tmp_3.html"
    tmp_3.write_text(content.replace("</head>", custom_style + "</head>").replace("</body>", js_3 + "</body>"), encoding="utf-8")
    subprocess.run([
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--window-size=1440,1050",
        f"--screenshot={out_dir / '03_execution_details.png'}",
        f"file:///{tmp_3.resolve()}",
    ], check=True)
    tmp_3.unlink()

    print("Screenshots captured successfully!")

if __name__ == "__main__":
    main()
