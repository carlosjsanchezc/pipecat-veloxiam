"""Gestión de sesiones de llamada.

Cada ``CallSession`` acumula el transcript (turnos usuario/agente con confidence)
en memoria durante la llamada. Al terminar, el transcript completo se envía a
NestJS.
"""

import time
import uuid
from typing import Dict, Optional

import httpx
from loguru import logger


class CallSession:
    def __init__(self, call_id: str, bot_id: str, tenant_id: str, phone_number: str):
        self.id = str(uuid.uuid4())
        self.call_id = call_id
        self.bot_id = bot_id
        self.tenant_id = tenant_id
        self.phone_number = phone_number
        self.started_at = time.time()

        # Transcript en memoria: [{role, text, confidence, ts}, ...]
        self.transcript: list[dict] = []
        # Confidence de la última transcripción final del usuario (lo fija
        # TranscriptionTap y lo lee NestJSAgentProcessor al cerrar el turno).
        self.last_user_confidence: Optional[float] = None
        # PipelineWorker en ejecución (lo asigna pipeline.bridge.run_call).
        self.worker = None

    def add_turn(self, role: str, text: str, confidence: Optional[float] = None):
        self.transcript.append(
            {"role": role, "text": text, "confidence": confidence, "ts": time.time()}
        )

    def to_payload(self) -> dict:
        return {
            "sessionId": self.id,
            "callId": self.call_id,
            "botId": self.bot_id,
            "tenantId": self.tenant_id,
            "from": self.phone_number,
            "durationSecs": round(time.time() - self.started_at, 2),
            "transcript": self.transcript,
        }


class SessionManager:
    def __init__(self, max_sessions: int, http: httpx.AsyncClient, transcript_url: str = ""):
        self.max_sessions = max_sessions
        self._http = http
        # Opcional: endpoint para volcar el transcript completo al terminar.
        # Vacío = no se envía (Veloxiam ya recibe cada turno vía llm-response).
        self._transcript_url = (transcript_url or "").strip()
        self.sessions: Dict[str, CallSession] = {}  # por call_id

    # --- estado ---
    def is_full(self) -> bool:
        return len(self.sessions) >= self.max_sessions

    def count(self) -> int:
        return len(self.sessions)

    def get(self, call_id: str) -> Optional[CallSession]:
        return self.sessions.get(call_id)

    # --- ciclo de vida ---
    def create(
        self, call_id: str, bot_id: str, tenant_id: str, phone_number: str
    ) -> CallSession:
        session = CallSession(call_id, bot_id, tenant_id, phone_number)
        self.sessions[call_id] = session
        logger.info(f"Sesión creada call_id={call_id} session={session.id} bot={bot_id}")
        return session

    async def terminate(self, call_id: str):
        session = self.sessions.pop(call_id, None)
        if session is None:
            return
        if session.worker is not None:
            try:
                await session.worker.cancel(reason="terminate")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Error al cancelar worker de {call_id}: {e}")
        await self._flush_transcript(session)

    async def close_all(self):
        for call_id in list(self.sessions.keys()):
            await self.terminate(call_id)

    # --- envío del transcript a Veloxiam (opcional) ---
    async def _flush_transcript(self, session: CallSession):
        if not self._transcript_url:
            return
        if not session.transcript:
            logger.info(f"Sin transcript que enviar (session={session.id})")
            return
        try:
            resp = await self._http.post(
                self._transcript_url,
                json=session.to_payload(),
                headers={"x-bot-id": session.bot_id},
                timeout=15.0,
            )
            resp.raise_for_status()
            logger.info(f"Transcript enviado (session={session.id})")
        except Exception as e:  # noqa: BLE001
            logger.error(f"No se pudo enviar el transcript (session={session.id}): {e}")
