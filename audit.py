import json
import os
from datetime import datetime

AUDIT_LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_log.json")

class AuditLog:
    @staticmethod
    def _read_logs():
        if not os.path.exists(AUDIT_LOG_FILE):
            return []
        try:
            with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    @staticmethod
    def _write_logs(logs):
        try:
            with open(AUDIT_LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error writing to audit log: {e}")
            return False

    @classmethod
    def log_submission(cls, submission_id, text, local_score, llm_score, combined_prob, confidence, label_details, creator_id="unknown"):
        logs = cls._read_logs()
        
        # Ensure we support both new content_id/attribution schema and our UI schema
        entry = {
            "content_id": submission_id,
            "submission_id": submission_id,
            "creator_id": creator_id,
            "text_snippet": text[:100] + ("..." if len(text) > 100 else ""),
            "attribution": label_details["attribution_result"],
            "attribution_result": label_details["attribution_result"],
            "confidence": round(confidence, 4),
            "confidence_score": round(confidence, 4),
            "llm_score": round(llm_score, 4),
            "signals": {
                "lexical_diversity": round(local_score, 4),
                "style_pattern_match": round(llm_score, 4)
            },
            "transparency_label": label_details,
            "status": "classified",
            "appeal": None,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "created_at": datetime.utcnow().isoformat() + "Z"
        }
        
        logs.append(entry)
        cls._write_logs(logs)
        return entry

    @classmethod
    def log_appeal(cls, submission_id, reasoning):
        logs = cls._read_logs()
        updated = False
        
        for entry in logs:
            if entry["submission_id"] == submission_id:
                entry["status"] = "under review"
                entry["appeal"] = {
                    "status": "under review",
                    "reasoning": reasoning,
                    "logged_at": datetime.utcnow().isoformat() + "Z"
                }
                updated = True
                break
                
        if updated:
            cls._write_logs(logs)
            return next(entry for entry in logs if entry["submission_id"] == submission_id)
        return None


    @classmethod
    def get_logs(cls, limit=None):
        logs = cls._read_logs()
        # Sort by created_at descending
        logs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        if limit:
            return logs[:limit]
        return logs
