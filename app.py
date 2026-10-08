import os
import sys

# 1. spaces MUST be the very first import for Hugging Face ZeroGPU compatibility
try:
    import spaces
except ImportError:
    class _MockSpaces:
        def GPU(self, *args, **kwargs):
            return lambda fn: fn
    spaces = _MockSpaces()

import gradio as gr
from fastapi.responses import RedirectResponse

# Ensure backend package is in python sys.path
backend_dir = os.path.join(os.path.dirname(__file__), "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# If root module "app" is registered without __path__, temporarily remove it
# so the backend 'app' package can be imported reliably
_root_app = sys.modules.get("app")
if _root_app and not hasattr(_root_app, "__path__"):
    del sys.modules["app"]

from app.main import app as fastapi_app

# Convenience redirect so /api/docs routes directly to /docs
@fastapi_app.get("/api/docs", include_in_schema=False)
def api_docs_redirect():
    return RedirectResponse(url="/docs")

# 2. Top-level @spaces.GPU decorated probe function
# Satisfies Hugging Face ZeroGPU AST scanner and supervisor when running on ZeroGPU hardware
@spaces.GPU(duration=60)
def gpu_keepalive_probe(query: str = "") -> str:
    """ZeroGPU probe handler."""
    return f"Status: online. {query}"

# 3. Standard Gradio Interface connected directly to the GPU function
with gr.Blocks(title="Tracker Failure Simulator Backend") as demo:
    gr.Markdown("# 🎯 Tracker Failure Simulator Backend")
    gr.Markdown(
        "Stress-test **17 multi-object tracking algorithms** (ByteTrack, BoT-SORT, DeepOC-SORT, StrongSORT, etc.) "
        "against occlusion, crossing paths, motion blur, and camera shake."
    )
    with gr.Accordion("🚀 Active API Endpoints", open=True):
        gr.Markdown("""
        - 📖 **Interactive Swagger Docs**: [/docs](/docs) or [/api/docs](/api/docs)
        - 🔍 **Trackers Catalog**: [/api/trackers](/api/trackers)
        - 🩺 **Health Check**: [/api/health](/api/health)
        
        *This Hugging Face Space powers the compute engine for the Next.js frontend.*
        """)
    
    # Visible interactive Gradio component bound to the @spaces.GPU function
    with gr.Row():
        test_btn = gr.Button("Ping Server & GPU Engine")
        test_out = gr.Textbox(label="Status Output", value="Ready")
    test_btn.click(fn=gpu_keepalive_probe, inputs=test_out, outputs=test_out)

# 4. Launch Gradio server first with prevent_thread_lock=True
# Hugging Face Spaces detects Gradio running on port 7860 and keeps the container healthy.
# SSR is disabled (ssr_mode=False) to prevent Node.js sidecar crash in single-port sandbox.
demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False, prevent_thread_lock=True)

# 5. Attach the FastAPI routes (/api/*, /docs, /openapi.json) to the active server application.
# In Gradio 5+, demo.launch() creates demo.server_app; including the router here makes all
# FastAPI endpoints directly accessible without being overwritten.
demo.server_app.include_router(fastapi_app.router)

# 6. Block the main thread cleanly while the server is active
demo.block_thread()
