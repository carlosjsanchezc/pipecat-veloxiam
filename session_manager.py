import uuid
from typing import Dict

class CallSession:
    def __init__(self, call_id: str, bot_id: str, tenant_id: str, phone_number: str):
        self.id = str(uuid.uuid4())
        self.call_id = call_id
        self.bot_id = bot_id
        self.tenant_id = tenant_id
        self.phone_number = phone_number
        self.transcript = []

class SessionManager:
    def __init__(self, max_sessions: int = 10):
        self.max_sessions = max_sessions
        self.sessions: Dict[str, CallSession] = {}

    def is_full(self) -> bool:
        return len(self.sessions) >= self.max_sessions

    def count(self) -> int:
        return len(self.sessions)

    async def create(self, call_id: str, bot_id: str, tenant_id: str, phone_number: str) -> str:
        session = CallSession(call_id, bot_id, tenant_id, phone_number)
        self.sessions[call_id] = session
        return session.id

    async def terminate(self, call_id: str):
        if call_id in self.sessions:
            del self.sessions[call_id]

    async def close_all(self):
        self.sessions.clear()
