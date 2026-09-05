// Runtime configuration for Hybrid-Agentic RAG Frontend
// Override API_BASE_URL for production or Docker deployments.
// For local development: frontend on port 3000, backend on port 8000.
window.__RAG_CONFIG__ = {
    API_BASE_URL: "http://127.0.0.1:8000"
};
