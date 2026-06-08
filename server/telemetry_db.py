import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "datasets" / "telemetry.db"

def init_telemetry_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_telemetry (
                task_id TEXT PRIMARY KEY,
                user_prompt TEXT,
                status TEXT,
                latency_ms REAL,
                rag_enabled INTEGER,
                validated_json TEXT,
                error_log TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

def log_telemetry(task_id: str, prompt: str, status: str, latency: float, rag: bool, json_out: str, error: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO model_telemetry 
            (task_id, user_prompt, status, latency_ms, rag_enabled, validated_json, error_log)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (task_id, prompt, status, latency, 1 if rag else 0, json_out, error))