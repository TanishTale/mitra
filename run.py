#!/usr/bin/env python3
"""Launch the MITRA web app.

    python run.py                 # http://127.0.0.1:8000
    python run.py --port 9000
"""
import argparse

import uvicorn

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    a = ap.parse_args()
    print(f"\n  MITRA is running at http://{a.host}:{a.port}\n"
          f"  API docs at http://{a.host}:{a.port}/docs\n")
    uvicorn.run("app.api:app", host=a.host, port=a.port, reload=a.reload)
