"""Simulation Arena - HTTP application entry point.

    python -m app.main          # serves http://localhost:8000

The API lives under /api. Built artifacts are served from /api/artifacts, and
the built front-end (if it has been compiled) is served at /.
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.api import jobs, store
from app.core import registry

BACKEND_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIST = BACKEND_ROOT.parent / "frontend" / "dist"

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]

app = FastAPI(title="Simulation Arena", version="1.0.0",
              description="A local, deterministic business simulation studio.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(router, prefix="/api")

# Downloadable artifacts: /api/artifacts/<run_id>/<filename>
store.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/api/artifacts", StaticFiles(directory=str(store.EXPORTS_DIR)),
          name="artifacts")


@asynccontextmanager
async def _lifespan(_: FastAPI):
    registry.discover()
    errors = registry.load_errors()
    print(f"Simulation Arena ready: {len(registry.all_use_cases())} simulations "
          f"loaded.", file=sys.stderr, flush=True)
    for name, err in errors.items():
        print(f"  [load error] {name}: {err}", file=sys.stderr, flush=True)
    limits = jobs.limits_payload()
    print(f"Upload limits: files up to {limits['max_file_human']}, "
          f"{limits['max_working_rows_human']} kept per file, "
          f"{limits['max_concurrent_uploads']} large file(s) read at a time, "
          f"{int(limits['parse_timeout_s'])}s to read one.",
          file=sys.stderr, flush=True)
    store.ensure_dirs()
    store.clear_uploads()          # spooled files left behind by a hard stop
    try:
        yield
    finally:
        jobs.shutdown()            # stop workers, drop any spooled uploads


app.router.lifespan_context = _lifespan


# -- optional: serve the compiled front-end so :8000 works on its own --------
if (FRONTEND_DIST / "index.html").exists():
    if (FRONTEND_DIST / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")),
                  name="frontend-assets")

    # brand assets (favicon, touch icon, manifest, logo) live in public/ and are
    # copied to dist/ at build time - serve them explicitly so the tab icon and
    # the PWA manifest resolve when the app is served from :8000.
    _public = FRONTEND_DIST.parent / "public"
    if _public.is_dir():
        for _f in ("favicon.svg", "favicon.ico", "icon-192.png", "icon-512.png",
                   "apple-touch-icon.png", "logo.svg", "manifest.webmanifest"):
            if (_public / _f).is_file():
                app.mount(f"/{_f}", StaticFiles(directory=str(_public)),
                          name=f"brand-{_f.replace('.', '-')}")

    @app.get("/", include_in_schema=False)
    def _index():
        return FileResponse(str(FRONTEND_DIST / "index.html"))

    @app.get("/{full_path:path}", include_in_schema=False)
    def _spa(full_path: str):
        target = FRONTEND_DIST / full_path
        if target.is_file():
            return FileResponse(str(target))
        return FileResponse(str(FRONTEND_DIST / "index.html"))


if __name__ == "__main__":
    import argparse
    import os
    import socket

    import uvicorn

    registry.discover()
    print(f"Simulation Arena: {len(registry.all_use_cases())} simulations loaded.",
          flush=True)

    # QA SP-10: during testing another product took over port 8000 mid-cycle and
    # a whole test pass had to be discarded. Claim the port up front and fail
    # loudly (with a fix) instead of quietly binding somewhere else or being
    # displaced later.
    parser = argparse.ArgumentParser(description="Simulation Arena")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("SIMARENA_PORT", "8000")),
                        help="Port to serve on (default 8000, or $SIMARENA_PORT).")
    parser.add_argument("--host", default=os.environ.get("SIMARENA_HOST", "127.0.0.1"))
    args = parser.parse_args()

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind((args.host, args.port))
    except OSError as exc:
        probe.close()
        raise SystemExit(
            f"\nSimulation Arena could not claim port {args.port} on {args.host}: {exc}.\n"
            f"Something else is already serving there, so the app would be displaced "
            f"mid-session.\n"
            f"Fix: start on a different port, e.g.  python -m app.main --port 8123\n"
            f"     or set SIMARENA_PORT=8123 once and leave it pinned.\n"
        ) from exc
    finally:
        try:
            probe.close()
        except OSError:
            pass

    print(f"Simulation Arena serving on http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
