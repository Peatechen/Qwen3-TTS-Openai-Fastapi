# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-TTS OpenAI-Compatible FastAPI Server.

A high-performance TTS API server providing OpenAI-compatible endpoints
for the Qwen3-TTS model.
"""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    gr = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Server configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8880"))
WORKERS = int(os.getenv("WORKERS", "1"))

# Backend configuration
TTS_BACKEND = os.getenv("TTS_BACKEND", "official")
TTS_WARMUP_ON_START = os.getenv("TTS_WARMUP_ON_START", "false").lower() == "true"

# GPU keepalive: periodic matmul prevents AMD DPM from downclocking the GPU
# after an idle period (which would otherwise spike TTFB from ~0.3 s to ~0.85 s).
# Set GPU_KEEPALIVE_INTERVAL to a positive integer (seconds) to enable.
# Recommended: 15 s for AMD ROCm; harmless on NVIDIA.  Default 0 = disabled.
GPU_KEEPALIVE_INTERVAL = int(os.getenv("GPU_KEEPALIVE_INTERVAL", "0"))

# Model auto-unload: release VRAM/RAM when idle, reload on next request.
# Supports human-readable durations: '2m', '30s', '1h', '0' (disabled).
_MODEL_IDLE_TIMEOUT_RAW = os.getenv("MODEL_IDLE_TIMEOUT", "0")
_MODEL_UNLOAD_CHECK_RAW = os.getenv("MODEL_UNLOAD_CHECK_INTERVAL", "30s")

# CORS configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Voice Studio configuration
ENABLE_VOICE_STUDIO = os.getenv("ENABLE_VOICE_STUDIO", "false").lower() == "true"
VOICE_LIBRARY_DIR = Path(os.getenv("VOICE_LIBRARY_DIR", "./voice_library")).resolve()

# Get the directory containing static files
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for model initialization."""
    
    # Print startup banner
    boundary = "░" * 24
    startup_msg = f"""
{boundary}

    ╔═╗┬ ┬┌─┐┌┐┌╔═╗  ╔╦╗╔╦╗╔═╗
    ║═╬╡│││├┤ │││╚═╗───║  ║ ╚═╗
    ╚═╝└┴┘└─┘┘└┘╚═╝   ╩  ╩ ╚═╝
    
    OpenAI-Compatible TTS API
    Backend: {TTS_BACKEND}

{boundary}
"""
    logger.info(startup_msg)
    # Show localhost in logs for user-friendly access URL (server binds to 0.0.0.0)
    display_host = "localhost" if HOST == "0.0.0.0" else HOST
    logger.info(f"Server starting on http://{display_host}:{PORT}")
    logger.info(f"API Documentation: http://{display_host}:{PORT}/docs")
    logger.info(f"Web Interface: http://{display_host}:{PORT}/")
    if ENABLE_VOICE_STUDIO:
        logger.info(f"Voice Studio: http://{display_host}:{PORT}/voice-studio")
    logger.info(boundary)

    # Parse auto-unload durations
    from .backends.auto_unload import parse_duration, get_auto_unload_manager
    try:
        _idle_timeout = parse_duration(_MODEL_IDLE_TIMEOUT_RAW)
    except ValueError as exc:
        logger.warning(f"Invalid MODEL_IDLE_TIMEOUT '{_MODEL_IDLE_TIMEOUT_RAW}': {exc}. Auto-unload disabled.")
        _idle_timeout = 0
    try:
        _check_interval = parse_duration(_MODEL_UNLOAD_CHECK_RAW)
    except ValueError:
        _check_interval = 30

    # Pre-load the TTS backend
    _custom_voices_dir = None
    backend = None
    try:
        from .backends import initialize_backend
        import os as _os
        from pathlib import Path as _Path
        _custom_voices_dir = _os.getenv(
            "TTS_CUSTOM_VOICES",
            str(_Path(__file__).resolve().parent.parent / "custom_voices"),
        )
        logger.info(f"Initializing TTS backend: {TTS_BACKEND}")
        backend = await initialize_backend(warmup=TTS_WARMUP_ON_START)
        logger.info(f"TTS backend '{backend.get_backend_name()}' loaded successfully!")
        logger.info(f"Model: {backend.get_model_id()}")
        
        device_info = backend.get_device_info()
        if device_info.get("gpu_available"):
            logger.info(f"GPU: {device_info.get('gpu_name')}")
            logger.info(f"VRAM: {device_info.get('vram_total')}")

        custom_voice_names = backend.get_custom_voice_names()
        if custom_voice_names:
            logger.info(f"Custom voices ({len(custom_voice_names)}): {custom_voice_names}")
    except Exception as e:
        logger.warning(f"Backend initialization delayed: {e}")
        logger.info("Backend will be loaded on first request.")

    # Start auto-unload manager (works even if backend not yet loaded)
    if backend is None:
        from .backends import get_backend
        try:
            backend = get_backend()
        except Exception:
            pass
    if backend is not None:
        _manager = get_auto_unload_manager()
        _manager.start(
            backend,
            timeout_seconds=_idle_timeout,
            check_interval=_check_interval,
            custom_voices_dir=_custom_voices_dir,
        )

    # GPU keepalive: periodic small matmul prevents AMD DPM from downclocking the GPU
    # after idle, keeping TTFB consistently low (~0.3 s instead of ~0.85 s after idle).
    # Enable with GPU_KEEPALIVE_INTERVAL=15 (seconds). Harmless on NVIDIA.
    keepalive_task = None
    if GPU_KEEPALIVE_INTERVAL > 0:
        try:
            import torch
            if torch.cuda.is_available():
                # Use the backend's device if it's loaded; otherwise fall back to cuda:0.
                try:
                    from .backends import get_backend as _get_backend
                    _be = _get_backend()
                    _ka_device = str(_be.device) if hasattr(_be, "device") and _be.device else "cuda:0"
                except Exception:
                    _ka_device = "cuda:0"
                _ka_tensor = torch.randn(512, 512, device=_ka_device)

                async def _gpu_keepalive():
                    try:
                        while True:
                            await asyncio.sleep(GPU_KEEPALIVE_INTERVAL)
                            with torch.inference_mode():
                                torch.matmul(_ka_tensor, _ka_tensor)
                    except asyncio.CancelledError:
                        pass

                keepalive_task = asyncio.create_task(_gpu_keepalive())
                logger.info(
                    f"GPU keepalive enabled: matmul every {GPU_KEEPALIVE_INTERVAL}s "
                    f"on {_ka_device}"
                )
        except Exception as exc:
            logger.warning(f"GPU keepalive setup failed: {exc}")

    yield

    # Cleanup
    if keepalive_task:
        keepalive_task.cancel()
    # Stop auto-unload manager
    try:
        from .backends.auto_unload import get_auto_unload_manager
        await get_auto_unload_manager().stop()
    except Exception:
        pass
    logger.info("Server shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="Qwen3-TTS API",
    description="""
## Qwen3-TTS OpenAI-Compatible API

A high-performance text-to-speech API server powered by Qwen3-TTS, 
providing full compatibility with OpenAI's TTS API specification.

### Features
- 🎯 OpenAI API compatible endpoints
- 🌍 Multi-language support (10+ languages)
- 🎨 Multiple voice options
- 📊 Multiple audio formats (MP3, Opus, AAC, FLAC, WAV, PCM)
- ⚡ GPU-accelerated inference
- 🔧 Text normalization and sanitization

### Quick Start
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8880/v1", api_key="not-needed")

response = client.audio.speech.create(
    model="qwen3-tts",
    voice="Vivian",
    input="Hello! This is Qwen3-TTS speaking."
)
response.stream_to_file("output.mp3")
```
""",
    version="0.1.0",
    lifespan=lifespan,
    openapi_url="/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from .routers.openai_compatible import router as openai_router
app.include_router(openai_router, prefix="/v1")

# Mount static files if directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Mount Voice Studio if enabled
if ENABLE_VOICE_STUDIO:
    if not GRADIO_AVAILABLE:
        logger.warning("Voice Studio enabled but gradio is not installed. Install with: pip install gradio")
    else:
        try:
            # Import gradio_voice_studio from parent directory
            parent_dir = Path(__file__).parent.parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))
            
            from gradio_voice_studio import build_app
            
            # Build the Voice Studio app with the current server URL
            # Use localhost when server is bound to 0.0.0.0, otherwise use the actual host
            voice_studio_host = "localhost" if HOST == "0.0.0.0" else HOST
            base_url = f"http://{voice_studio_host}:{PORT}"
            voice_studio_app = build_app(base_url, VOICE_LIBRARY_DIR)
            
            # Mount the Gradio app
            app = gr.mount_gradio_app(app, voice_studio_app, path="/voice-studio")
            logger.info(f"Voice Studio mounted at /voice-studio")
        except Exception as e:
            logger.warning(f"Failed to mount Voice Studio: {e}")
            logger.info("Voice Studio can still be run separately with 'qwen-tts-voice-studio'")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main web interface."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    
    # Build links dynamically
    voice_studio_link = ""
    if ENABLE_VOICE_STUDIO and GRADIO_AVAILABLE:
        voice_studio_link = '<li><a href="/voice-studio">🎙️ Voice Studio</a></li>'
    
    # Return a simple HTML page if index.html doesn't exist
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Qwen3-TTS API</title>
    <style>
        body {{ 
            font-family: 'Courier New', monospace; 
            background: #1a1a2e; 
            color: #eee; 
            padding: 40px;
            max-width: 800px;
            margin: 0 auto;
        }}
        pre {{ color: #00ff88; }}
        a {{ color: #00aaff; }}
        h1 {{ color: #fff; }}
    </style>
</head>
<body>
    <pre>
    ╔═╗┬ ┬┌─┐┌┐┌╔═╗  ╔╦╗╔╦╗╔═╗
    ║═╬╡│││├┤ │││╚═╗───║  ║ ╚═╗
    ╚═╝└┴┘└─┘┘└┘╚═╝   ╩  ╩ ╚═╝
    </pre>
    <h1>Qwen3-TTS OpenAI-Compatible API</h1>
    <p>Welcome to the Qwen3-TTS API server!</p>
    <ul>
        <li><a href="/docs">API Documentation (Swagger UI)</a></li>
        <li><a href="/redoc">API Documentation (ReDoc)</a></li>
        <li><a href="/v1/models">List Models</a></li>
        <li><a href="/v1/voices">List Voices</a></li>
        {voice_studio_link}
    </ul>
</body>
</html>
"""


@app.get("/health")
async def health_check():
    """Health check endpoint with backend information."""
    try:
        from .backends import get_backend
        
        backend = get_backend()
        device_info = backend.get_device_info()
        
        return {
            "status": "healthy" if backend.is_ready() else "initializing",
            "backend": {
                "name": backend.get_backend_name(),
                "model_id": backend.get_model_id(),
                "ready": backend.is_ready(),
            },
            "device": {
                "type": device_info.get("device"),
                "gpu_available": device_info.get("gpu_available"),
                "gpu_name": device_info.get("gpu_name"),
                "vram_total": device_info.get("vram_total"),
                "vram_used": device_info.get("vram_used"),
            },
            "version": "0.1.0",
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "backend": {
                "name": TTS_BACKEND,
                "ready": False,
            },
            "version": "0.1.0",
        }


def main():
    """Run the server using uvicorn."""
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=HOST,
        port=PORT,
        workers=WORKERS,
        reload=False,
    )


if __name__ == "__main__":
    main()
