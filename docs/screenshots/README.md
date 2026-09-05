# UI Screenshots & Capture Instructions

This directory holds visual records (`before.png` and `after.png`) documenting the Streamlit interface redesign for the Hybrid Agentic RAG system.

---

## Required Files

- `before.png`: Baseline Streamlit UI (original default design with emoji headers and standard metric widgets).
- `after.png`: Redesigned developer/infrastructure UI (restrained palette `#0B0F14`, technical badges, stat strip, trace log, and evidence cards).

---

## Instructions for Capturing Screenshots

Because this repository enforces strict **offline-first** constraints and does not bundle heavy headless browser drivers (like Playwright or Selenium), follow these steps to capture visual records using your local browser:

### Step 1: Ensure Local Stack is Running
Make sure the backend and frontend services are active:
```powershell
# In terminal 1 (or via docker compose):
docker compose up -d

# Verify services:
curl http://localhost:8000/health
curl http://localhost:8501/_stcore/health
```

### Step 2: Open Application in Browser
1. Open your browser (Google Chrome, Microsoft Edge, Brave, or Firefox) and navigate to:
   `http://localhost:8501`
2. Set the browser window to standard desktop resolution:
   - Width: **1440px** (or 1920px)
   - Height: **900px** (or 1080px)

### Step 3: Run Sample Query
1. In the **ASK THE DOCUMENTATION** area, click on the example chip:
   `How does Docker bridge networking work?`
2. Click **ASK ASSISTANT →** to generate the full answer, metrics strip, orchestration trace, and retrieved sources.

### Step 4: Capture Screenshots
1. **Full Page or Viewport Capture**:
   - In Chrome / Edge: Press `Ctrl + Shift + I` to open Developer Tools.
   - Press `Ctrl + Shift + P` to open the Command Menu.
   - Type `Capture full size screenshot` (or `Capture screenshot`) and press `Enter`.
2. Save the file to this directory:
   - Save the redesigned state as `docs/screenshots/after.png`.
   - If rolling back or comparing with the initial commit baseline, save the original state as `docs/screenshots/before.png`.
