# AI QR Attendance Agent

A locally runnable attendance system using **FastAPI + LangGraph + ChromaDB + OpenAI + SQLite**.

## Features

- Register users and generate QR codes.
- Record attendance via `/scan`.
- Run a LangGraph workflow:
  - Input → Fetch User → Compute Context → Memory Retrieval → Prompt Builder → LLM → Store Memory → Output
- Save attendance logs in SQLite.
- Save and retrieve personalized message memory in ChromaDB.
- Minimal UI scaffold at `/` with webcam structure and scan trigger.

## Project Structure

```text
app/
  main.py
  agent/
    graph.py
    nodes.py
  db/
    models.py
  services/
    chroma_service.py
    openai_service.py
  utils/
    qr_generator.py
main.py
requirements.txt
.env.example
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables:

```bash
cp .env.example .env
# then edit .env and set OPENAI_API_KEY
```

## Run

```bash
uvicorn main:app --reload
```

Open:
- API docs: `http://127.0.0.1:8000/docs`
- Minimal UI: `http://127.0.0.1:8000/`

## API Endpoints

### `POST /register`
Create a user and generate a QR image in `data/qrcodes/`.

Request:
```json
{ "name": "Alice" }
```

### `POST /scan`
Input `user_id`, record attendance, execute LangGraph agent, return AI message.

Request:
```json
{ "user_id": 1 }
```

### `GET /attendance`
Fetch attendance logs with timestamps.

## Notes

- SQLite DB path: `data/attendance.db`
- Chroma persistence path: `data/chroma/`
- Chroma collection: `attendance_memory`
