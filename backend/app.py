import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request

# Allow direct execution with `python backend/app.py`.
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.practice import init_app

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

app = Flask(__name__)
init_app(app)


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    allowed = os.getenv(
        "CTRLPASS_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin in allowed:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Vary"] = "Origin"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-CtrlPass-Request"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PATCH, OPTIONS"
    return response


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
