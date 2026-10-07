import os
import sys
import uvicorn
import gradio as gr

# Ensure backend package is in python sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from app.main import app as fastapi_app

# 1. Build an informative landing interface for the Space
with gr.Blocks(title="Tracker Failure Simulator Backend") as demo:
    gr.Markdown("# 🎯 Tracker Failure Simulator Backend")
    gr.Markdown(
        "Stress-test **17 multi-object tracking algorithms** (ByteTrack, BoT-SORT, DeepOC-SORT, StrongSORT, etc.) "
        "against occlusion, crossing paths, motion blur, and camera shake."
    )
    with gr.Accordion("🚀 Active API Endpoints", open=True):
        gr.Markdown("""
        - 📖 **Interactive Swagger Docs**: [/docs](/docs)
        - 🔍 **Trackers Catalog**: [/api/trackers](/api/trackers)
        - 🩺 **Health Check**: [/api/health](/api/health)
        
        *This Hugging Face Space powers the compute engine for the Next.js frontend.*
        """)

# 2. Mount Gradio onto the existing FastAPI application
# This keeps all FastAPI routes (/api/*, /docs, /openapi.json) live and serves Gradio at /
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

# 3. Launch Uvicorn on 0.0.0.0:7860 (Hugging Face Spaces default container port)
# Running unconditionally at module level ensures the Python process remains alive permanently
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
