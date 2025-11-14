# app.py
from flask import Flask, jsonify
import subprocess
import sys
from pathlib import Path

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

@app.post("/admin/assign-tomorrow")
def run_assign_tomorrow():
    try:
        result = subprocess.run(
            [sys.executable, "-m", "scripts.assign_tomorrow"],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            check=True,
        )

        return jsonify({
            "ok": True,
            "message": "assign_tomorrow 실행 완료",
            "stdout": result.stdout,
            "stderr": result.stderr
        })
    except subprocess.CalledProcessError as e:
        return jsonify({
            "ok": False,
            "message": "assign_tomorrow 실행 중 오류 발생",
            "stdout": e.stdout,
            "stderr": e.stderr
        }), 500
