import os
import sys

# Ensure backend package is in python sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

# ZeroGPU integration
try:
    import spaces
except ImportError:
    class _MockSpaces:
        def GPU(self, *args, **kwargs):
            def decorator(fn):
                return fn
            return decorator
    spaces = _MockSpaces()

import gradio as gr
from app.main import app as fastapi_app

# Top-level GPU decorated function
@spaces.GPU
def gpu_compute_keepalive(x: str = "") -> str:
    return f"Active: {x}"

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
    
    trigger_btn = gr.Button("Status Check", visible=False)
    output_txt = gr.Textbox(visible=False)
    trigger_btn.click(fn=gpu_compute_keepalive, inputs=output_txt, outputs=output_txt)

# gr.mount_gradio_app mounts the Gradio interface to the FastAPI application (at /),
# preserving all of FastAPI's native endpoints (/api/*, /docs, /openapi.json).
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
