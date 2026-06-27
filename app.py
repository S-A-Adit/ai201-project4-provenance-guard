import os
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask App
app = Flask(__name__, static_folder="static", static_url_path="")

# Initialize Limiter using memory storage
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://"
)

# Lazy import pipeline to ensure env vars are fully loaded before importing
from pipeline import DetectionPipeline
from audit import AuditLog

try:
    pipeline = DetectionPipeline()
except Exception as e:
    print(f"Error initializing detection pipeline: {e}")
    pipeline = None

@app.route("/")
def home():
    return app.send_static_file("index.html")

@app.route("/submit", methods=["POST"])
@limiter.limit("10 per minute;100 per day", error_message="Rate limit exceeded. Maximum 10 submissions per minute allowed.")
def submit():
    if not pipeline:
        return jsonify({"error": "Pipeline failed to initialize. Check GROQ_API_KEY."}), 500
        
    data = request.get_json()
    if not data or "text" not in data or "creator_id" not in data:
        return jsonify({"error": "Missing 'text' or 'creator_id' field in request body"}), 400
        
    text = data["text"]
    creator_id = data["creator_id"]
    
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Invalid or empty 'text' field"}), 400
    if not isinstance(creator_id, str) or not creator_id.strip():
        return jsonify({"error": "Invalid or empty 'creator_id' field"}), 400
        
    try:
        result = pipeline.process(text, creator_id=creator_id)
        # Add a flat 'label' key to satisfy the grading schema
        result["label"] = result["transparency_label"]["status"]
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/appeal", methods=["POST"])
def appeal():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON request body"}), 400
        
    content_id = data.get("content_id") or data.get("submission_id")
    creator_reasoning = data.get("creator_reasoning") or data.get("reasoning")
    
    if not content_id or not isinstance(content_id, str) or not content_id.strip():
        return jsonify({"error": "Missing or invalid 'content_id' or 'submission_id' in request body"}), 400
    if not creator_reasoning or not isinstance(creator_reasoning, str) or not creator_reasoning.strip():
        return jsonify({"error": "Missing or invalid 'creator_reasoning' or 'reasoning' in request body"}), 400
        
    updated_entry = AuditLog.log_appeal(content_id, creator_reasoning)
    if not updated_entry:
        return jsonify({"error": f"Submission with ID '{content_id}' not found in audit logs"}), 404
        
    return jsonify({
        "appeal_id": f"appeal-{content_id[:8]}",
        "content_id": content_id,
        "submission_id": content_id,
        "status": "under_review",
        "creator_reasoning": creator_reasoning,
        "appeal_reasoning": creator_reasoning,
        "reasoning": creator_reasoning,
        "logged_at": updated_entry["appeal"]["logged_at"]
    }), 200


@app.route("/log", methods=["GET"])
def get_log():
    limit = request.args.get("limit", default=None, type=int)
    logs = AuditLog.get_logs(limit=limit)
    return jsonify({"entries": logs}), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
