import os
import sys

# 1. spaces MUST be the very first import for ZeroGPU compatibility
try:
    import spaces
except ImportError:
    class _MockSpaces:
        def GPU(self, *args, **kwargs):
            return lambda fn: fn
    spaces = _MockSpaces()

import gradio as gr

# Ensure backend package is in python sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from app.main import app as fastapi_app

# 2. Top-level @spaces.GPU decorated probe function
# This satisfies Hugging Face ZeroGPU's AST scanner and runner checks
@spaces.GPU(duration=60)
def gpu_keepalive_probe(query: str = "") -> str:
    """ZeroGPU probe handler."""
    return f"Status: online. {query}"

# 3. Standard Gradio Interface connected directly to the GPU function
# In Hugging Face Spaces (Gradio SDK), demo.launch() registers the service with HF's supervisor
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
    
    # Visible interactive Gradio component bound to the @spaces.GPU function
    with gr.Row():
        test_btn = gr.Button("Ping Server & GPU Engine")
        test_out = gr.Textbox(label="Status Output", value="Ready")
    test_btn.click(fn=gpu_keepalive_probe, inputs=test_out, outputs=test_out)

# 4. Mount the FastAPI API routes (/api/*, /docs, /openapi.json) onto demo.app
demo.app.include_router(fastapi_app.router)

# 5. Launch native Gradio server with SSR disabled (to avoid Node.js exit)
demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
