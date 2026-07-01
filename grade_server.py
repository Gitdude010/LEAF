"""
LEAF Submission Validation Server (MLEBench Integration)
=========================================================

This Flask server validates ML competition submissions against the official
MLEBench grading rules. It is an optional component of the LEAF framework.

MLEBench (https://github.com/openai/mle-bench) is a benchmark suite of 75
Kaggle competitions designed to evaluate ML engineering agents. Each competition
includes a dataset, task description, and official grading logic that checks
submission format compliance (correct columns, row counts, ID matching, etc.).

How it works:
    1. The LEAF agent generates a submission CSV during training
    2. Before accepting a submission as valid, the agent sends it to this server
    3. The server calls mlebench.grade.validate_submission() to check format
    4. If valid, the agent keeps the submission; if invalid, the agent fixes it

Usage:
    # Start the server with MLEBench data directory
    python -m leaf.grade_server --data_dir /path/to/mle-bench/data

    # Or set port via environment variable
    GRADING_SERVER_PORT=5005 python -m leaf.grade_server --data_dir /path/to/data

    # Health check
    curl http://localhost:5005/health

    # Validate a submission
    curl -X POST http://localhost:5005/validate \\
         -H "exp-id: spaceship-titanic" \\
         -F "file=@submission.csv"

Requirements:
    pip install flask mlebench

    MLEBench data must be prepared following:
    https://github.com/openai/mle-bench/blob/main/docs/DATA.md

Note:
    This server is OPTIONAL. Set `use_grade_server: false` in config.yaml
    to disable format validation and run without it.
"""

import argparse
import logging
import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("leaf.grade_server")

app = Flask(__name__)

# Will be set via CLI argument
_data_dir = None


def init_app(data_dir: str):
    """Initialize the app with the MLEBench data directory."""
    global _data_dir
    _data_dir = Path(data_dir)
    if not _data_dir.exists():
        logger.warning(f"Data directory does not exist: {_data_dir}")
    else:
        logger.info(f"MLEBench data directory: {_data_dir}")


@app.post("/validate")
def validate():
    """Validate a submission CSV against MLEBench competition rules.

    Expects:
        - POST with multipart form data containing 'file' field
        - Header 'exp-id' with the competition ID (e.g., 'spaceship-titanic')

    Returns:
        JSON with 'is_valid' (bool) and 'result' (message string)
    """
    if "file" not in request.files:
        return jsonify({"error": "Missing 'file' in request"}), 400

    competition_id = request.headers.get("exp-id")
    if not competition_id:
        return jsonify({"error": "Missing 'exp-id' header"}), 400

    if _data_dir is None:
        return jsonify({"error": "Server not initialized with data directory"}), 500

    # Save uploaded file to a temp path (auto-cleaned)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        request.files["file"].save(tmp)
        tmp_path = Path(tmp.name)

    try:
        from mlebench.grade import validate_submission
        from mlebench.registry import registry

        comp = registry.set_data_dir(_data_dir).get_competition(competition_id)
        is_valid, message = validate_submission(tmp_path, comp)
        logger.info(f"Validation for {competition_id}: is_valid={is_valid}")
        return jsonify({"is_valid": is_valid, "result": message})
    except ImportError:
        logger.error("mlebench is not installed. Install with: pip install mlebench")
        return jsonify({"error": "mlebench not installed", "details": "pip install mlebench"}), 500
    except Exception as e:
        logger.exception("Validation failed")
        return jsonify({"error": "Validation failed", "details": str(e)}), 500
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "running"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LEAF Submission Validation Server (MLEBench)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to MLEBench data directory")
    parser.add_argument("--port", type=int, default=int(os.getenv("GRADING_SERVER_PORT", "5005")),
                        help="Server port (default: 5005, or GRADING_SERVER_PORT env var)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server host (default: 0.0.0.0)")
    args = parser.parse_args()

    init_app(args.data_dir)

    logger.info(f"Starting LEAF grade server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port)
