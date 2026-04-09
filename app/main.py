from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.graph import build_attendance_graph
from app.db.models import AttendanceLog, SessionLocal, User, init_db
from app.services.chroma_service import ChromaService
from app.services.openai_service import OpenAIService
from app.utils.qr_generator import generate_user_qr


class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class ScanRequest(BaseModel):
    user_id: int


app = FastAPI(title="AI QR Attendance Agent", version="1.0.0")

chroma_service = ChromaService()
openai_service: OpenAIService | None = None


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def on_startup() -> None:
    global openai_service
    init_db()
    openai_service = OpenAIService()


@app.get("/", response_class=HTMLResponse)
def ui_home() -> str:
    return """
    <!DOCTYPE html>
    <html>
    <head><title>AI QR Attendance Agent</title></head>
    <body style="font-family: sans-serif; max-width: 720px; margin: 2rem auto;">
      <h1>AI QR Attendance Agent</h1>
      <p>Minimal UI scaffold. Webcam hook included for future QR decoding integration.</p>
      <video id="camera" width="320" height="240" autoplay style="border:1px solid #ccc"></video>
      <div style="margin-top: 1rem;">
        <label>User ID:</label>
        <input id="userId" type="number" />
        <button onclick="scan()">Scan</button>
      </div>
      <pre id="result"></pre>
      <script>
        navigator.mediaDevices?.getUserMedia({ video: true })
          .then(stream => { document.getElementById('camera').srcObject = stream; })
          .catch(() => {});

        async function scan() {
          const userId = Number(document.getElementById('userId').value);
          const response = await fetch('/scan', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ user_id: userId })
          });
          const data = await response.json();
          document.getElementById('result').textContent = JSON.stringify(data, null, 2);
        }
      </script>
    </body>
    </html>
    """


@app.post("/register")
def register_user(payload: RegisterRequest, db: Session = Depends(get_db)):
    user = User(name=payload.name)
    db.add(user)
    db.commit()
    db.refresh(user)

    qr_path = generate_user_qr(user.id, output_dir=Path("data/qrcodes"))
    return {
        "user_id": user.id,
        "user_name": user.name,
        "qr_path": qr_path,
        "message": "User registered successfully",
    }


@app.post("/scan")
def scan_attendance(payload: ScanRequest, db: Session = Depends(get_db)):
    if openai_service is None:
        raise HTTPException(status_code=500, detail="OpenAI service is not initialized")

    user = db.get(User, payload.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    log = AttendanceLog(user_id=payload.user_id, timestamp=datetime.now(timezone.utc))
    db.add(log)
    db.commit()

    workflow = build_attendance_graph(db=db, chroma_service=chroma_service, openai_service=openai_service)
    result = workflow.invoke({"user_id": payload.user_id})

    return {
        "user_id": payload.user_id,
        "user_name": result["user_name"],
        "timestamp": result["timestamp"],
        "message": result["generated_message"],
    }


@app.get("/attendance")
def get_attendance(db: Session = Depends(get_db)):
    logs = db.scalars(select(AttendanceLog).order_by(AttendanceLog.timestamp.desc())).all()

    return [
        {
            "log_id": log.id,
            "user_id": log.user_id,
            "user_name": log.user.name,
            "timestamp": log.timestamp.isoformat(),
        }
        for log in logs
    ]
