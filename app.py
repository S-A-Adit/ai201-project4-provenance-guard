import os
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask App
app = Flask(__name__, static_folder="static", static_url_path="")

# Initialize Limiter (default: 100 per day, 10 per hour across all endpoints)
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["100 per day", "10 per hour"]
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
@limiter.limit("5 per minute", error_message="Rate limit exceeded. Maximum 5 submissions per minute allowed.")
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
    if not data or "submission_id" not in data or "reasoning" not in data:
        return jsonify({"error": "Missing 'submission_id' or 'reasoning' field in request body"}), 400
        
    submission_id = data["submission_id"]
    reasoning = data["reasoning"]
    
    if not isinstance(submission_id, str) or not submission_id.strip():
        return jsonify({"error": "Invalid 'submission_id'"}), 400
    if not isinstance(reasoning, str) or not reasoning.strip():
        return jsonify({"error": "Invalid 'reasoning'"}), 400
        
    updated_entry = AuditLog.log_appeal(submission_id, reasoning)
    if not updated_entry:
        return jsonify({"error": f"Submission with ID '{submission_id}' not found in audit logs"}), 404
        
    return jsonify({
        "appeal_id": f"appeal-{submission_id[:8]}",
        "submission_id": submission_id,
        "status": "under review",
        "reasoning": reasoning,
        "logged_at": updated_entry["appeal"]["logged_at"]
    }), 200

@app.route("/log", methods=["GET"])
def get_log():
    limit = request.args.get("limit", default=None, type=int)
    logs = AuditLog.get_logs(limit=limit)
    return jsonify({"entries": logs}), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
