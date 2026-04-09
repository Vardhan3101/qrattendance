from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TypedDict

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import AttendanceLog, User
from app.services.chroma_service import ChromaService
from app.services.openai_service import OpenAIService


class AttendanceState(TypedDict, total=False):
    user_id: int
    user_name: str
    timestamp: str
    time_of_day: str
    attendance_streak: int
    past_messages: list[str]
    prompt: str
    generated_message: str


def input_node(state: AttendanceState) -> AttendanceState:
    if "user_id" not in state:
        raise ValueError("user_id is required")
    return state


def fetch_user_node(state: AttendanceState, db: Session) -> AttendanceState:
    user = db.get(User, state["user_id"])
    if not user:
        raise ValueError(f"User {state['user_id']} not found")
    return {**state, "user_name": user.name}


def _attendance_streak_for_user(db: Session, user_id: int) -> int:
    result = db.scalars(
        select(AttendanceLog)
        .where(AttendanceLog.user_id == user_id)
        .order_by(AttendanceLog.timestamp.desc())
    ).all()

    if not result:
        return 0

    unique_days: list[datetime.date] = []
    for row in result:
        day = row.timestamp.date()
        if day not in unique_days:
            unique_days.append(day)

    today = datetime.now(timezone.utc).date()
    streak = 0
    for i, day in enumerate(unique_days):
        expected_day = today - timedelta(days=i)
        if day == expected_day:
            streak += 1
        else:
            break
    return streak


def _time_of_day(now: datetime) -> str:
    hour = now.hour
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    return "evening"


def compute_context_node(state: AttendanceState, db: Session) -> AttendanceState:
    now = datetime.now(timezone.utc)
    return {
        **state,
        "timestamp": now.isoformat(),
        "time_of_day": _time_of_day(now),
        "attendance_streak": _attendance_streak_for_user(db, state["user_id"]),
    }


def memory_retrieval_node(state: AttendanceState, chroma_service: ChromaService) -> AttendanceState:
    messages = chroma_service.get_recent_messages(user_id=state["user_id"], limit=5)
    return {**state, "past_messages": messages}


def prompt_builder_node(state: AttendanceState) -> AttendanceState:
    past_messages = "\n".join(f"- {m}" for m in state.get("past_messages", [])) or "- No previous messages"
    prompt = f"""You are a friendly AI attendance assistant.

User: {state['user_name']}
Time: {state['time_of_day']}
Streak: {state['attendance_streak']}

Previous Messages:
{past_messages}

Rules:
- Keep it under 20 words
- Be unique and not repetitive
- Friendly and motivating tone

Output:
Single-line greeting + motivation"""
    return {**state, "prompt": prompt}


def llm_node(state: AttendanceState, openai_service: OpenAIService) -> AttendanceState:
    generated_message = openai_service.generate_message(state["prompt"])
    return {**state, "generated_message": generated_message}


def memory_store_node(
    state: AttendanceState,
    chroma_service: ChromaService,
    openai_service: OpenAIService,
) -> AttendanceState:
    embedding = openai_service.embed_text(state["generated_message"])
    chroma_service.save_message(
        user_id=state["user_id"],
        timestamp=state["timestamp"],
        message=state["generated_message"],
        embedding=embedding,
    )
    return state


def output_node(state: AttendanceState) -> AttendanceState:
    return state
