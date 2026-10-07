import os
import sys

# Ensure backend package is in python sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

import gradio as gr
from app.main import app as fastapi_app

# Define a clean landing view for Hugging Face visitors
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
        
        *This Hugging Face Space powers the backend and compute engine for the Next.js frontend.*
        """)

# Mount the Gradio interface onto the existing FastAPI application
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    # Hugging Face Spaces exposes port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)
