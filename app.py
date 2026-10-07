import os
import sys

# Ensure backend package is in python sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

import gradio as gr
from app.main import app as fastapi_app

# Define the Gradio interface
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

# Mount FastAPI routers
demo.app.include_router(fastapi_app.router)

# In Gradio 5 on Hugging Face Spaces:
# 1. ssr_mode=False disables the background Node.js SSR sidecar process that causes "Stopping Node.js server..."
# 2. block=True keeps the main Python server thread listening and prevents premature process exit.
demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
