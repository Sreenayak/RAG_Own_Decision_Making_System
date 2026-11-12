"""Local Flask website for Smash AI."""

from pathlib import Path

import requests
from flask import Flask, jsonify, render_template, request

from full_pipeline.pipeline import RAGPipeline


ROOT = Path(__file__).resolve().parents[1]
app = Flask(
    __name__,
    template_folder=str(ROOT / "templates"),
    static_folder=str(ROOT / "static"),
)

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Create the retrieval pipeline on the first chat request."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send your question as JSON."}), 400

    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "Type a question first."}), 400

    try:
        result = get_pipeline().answer(question.strip())
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 503
    except requests.RequestException:
        app.logger.exception("The answer provider request failed.")
        return jsonify(
            {"error": "Smash AI could not reach the answer service. Try again shortly."}
        ), 502

    return jsonify({"answer": result["answer"]})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
