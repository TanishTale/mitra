#!/usr/bin/env python3
"""Launch the MITRA web app.

    python run.py                       # local: http://127.0.0.1:8000
    python run.py --host 0.0.0.0 --port $PORT   # cloud deployment (Render, etc.)

Reads PORT from the environment automatically if --port is not given, so this
also works unmodified as a platform-detected start command.
"""
import argparse
import os

import uvicorn

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument("--reload", action="store_true")
    a = ap.parse_args()
    print(f"\n  MITRA is running at http://{a.host}:{a.port}\n"
          f"  API docs at http://{a.host}:{a.port}/docs\n")
    uvicorn.run("app.api:app", host=a.host, port=a.port, reload=a.reload)
