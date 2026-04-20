from typing import Any
from typing_extensions import TypedDict


class MedicalState(TypedDict):
    user_input: str
    user_id: str
    session_id: str
    intents: list[str]
    agent_results: dict[str, str]
    patient_info: dict[str, Any]
    health_summary: str
    final_answer: str
    timing: dict[str, Any]
