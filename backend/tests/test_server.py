"""Integration test to verify live FastAPI server endpoints."""

import json
from pathlib import Path
import sys
import threading
import time
import urllib.request

# Ensure project root is in sys.path
_test_dir = Path(__file__).resolve().parent
_root_dir = _test_dir.parent.parent
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

import uvicorn
from backend.app.main import app


def run():
    # Pass the app object directly to uvicorn.Config to ensure clean in-memory execution
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait briefly for server to bind
    time.sleep(2)

    try:
        # 1. Test GET /health
        with urllib.request.urlopen("http://127.0.0.1:8000/health") as resp:
            health_code = resp.getcode()
            health_body = json.loads(resp.read().decode("utf-8"))
            print("=== HEALTH ENDPOINT TEST ===")
            print(f"HTTP Status: {health_code}")
            print(f"Response Body: {json.dumps(health_body, indent=2)}")

        # 2. Test GET /docs
        with urllib.request.urlopen("http://127.0.0.1:8000/docs") as resp:
            docs_code = resp.getcode()
            docs_content = resp.read().decode("utf-8")
            swagger_found = "swagger-ui" in docs_content.lower()
            print("\n=== SWAGGER DOCS TEST ===")
            print(f"HTTP Status: {docs_code}")
            print(f"Swagger UI Available: {swagger_found}")

        # 3. Test GET /
        with urllib.request.urlopen("http://127.0.0.1:8000/") as resp:
            root_code = resp.getcode()
            root_body = json.loads(resp.read().decode("utf-8"))
            print("\n=== ROOT ENDPOINT TEST ===")
            print(f"HTTP Status: {root_code}")
            print(f"Response Body: {json.dumps(root_body, indent=2)}")

        print("\nAll integration verification tests passed successfully!")
    finally:
        server.should_exit = True
        thread.join(timeout=3)
        print("Server stopped cleanly.")


if __name__ == "__main__":
    run()
