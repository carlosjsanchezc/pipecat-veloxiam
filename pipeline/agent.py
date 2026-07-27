"""Procesador del agente Veloxiam (POST llm-response → TTS).

Compartido por WhatsApp y Telnyx; cada transporte monta su propio pipeline.

Si Veloxiam responde con ``action`` de transferencia o fin de llamada,
se encola la despedida (si hay texto) y se libera el pipeline para que
Pipecat deje de hablar aunque el media stream siga abierto.
"""

import asyncio
import re
import time
from typing import Any, Optional

import httpx
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from pipeline.release import begin_release

_PLACEHOLDER_RE = re.compile(r"//[^/]+//")
# Tags tipo Cartesia/Veloxiam que no deben sintetizarse.
_EMOTION_TAG_RE = re.compile(r"<emotion\b[^>]*/?>", re.IGNORECASE)
_XML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")

# Acciones de Veloxiam que deben cortar el pipeline de voz.
_TRANSFER_ACTIONS = frozenset(
    {"transfer", "transfer_call", "call_transfer", "redirect", "transferring", "transferred"}
)
_END_ACTIONS = frozenset(
    {"end", "end_call", "hangup", "hang_up", "disconnect", "close"}
)

# Si Veloxiam no manda ``action`` pero el texto es claramente una despedida de transfer.
_REPLY_TRANSFER_RE = re.compile(
    r"(?i)("
    r"te (estoy )?(paso|pasando|transfiero|transfiriendo|conectando|enviando)|"
    r"transfiri\w*|"
    r"transferir\w*|"
    r"te conecto|"
    r"pasar(te|lo)? con|"
    r"solicitud para transfer|"
    r"transferirte|"
    r"paso a \w+|"
    r"comunic(o|arte) con"
    r")"
)

# Pedido del usuario de hablar con alguien / transferir.
_USER_TRANSFER_RE = re.compile(
    r"(?i)("
    r"transfer\w*|"
    r"comun[ií]que\w*|"
    r"pas[aá]me|"
    r"quiero (hablar )?con|"
    r"hable con|"
    r"con (un )?(agente|asesor|humano|operador)"
    r")"
)

_DEFAULT_FAREWELL = {
    "transfer": "Claro, te transfiero en este momento.",
    "end_call": "Gracias por llamar. Hasta luego.",
}


def _content_to_text(content: Any) -> str:
    """Normaliza el ``content`` de un mensaje del contexto a texto plano."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return " ".join(p for p in parts if p).strip()
    return str(content)


def _latest_user_text(context: LLMContext) -> Optional[str]:
    """Devuelve el texto del último mensaje de usuario en el contexto."""
    for msg in reversed(context.get_messages()):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "user":
            content = (
                msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            )
            text = _content_to_text(content)
            return text or None
    return None


def _clean_tts_text(text: str) -> Optional[str]:
    """Elimina placeholders y tags XML (p. ej. emotion) que no deben sintetizarse."""
    cleaned = _PLACEHOLDER_RE.sub("", text)
    cleaned = _EMOTION_TAG_RE.sub("", cleaned)
    cleaned = _XML_TAG_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


def _session_is_dead(session) -> bool:
    """True si ya no debemos hablar (transfer, hangup o transporte cerrado)."""
    return bool(
        getattr(session, "released", False)
        or getattr(session, "disconnected", False)
    )


def _normalize_release_action(action: Optional[str]) -> Optional[str]:
    """Mapea action cruda a ``transfer`` / ``end_call``, o None si no aplica."""
    if not action:
        return None
    key = action.strip().lower().replace("-", "_").replace(" ", "_")
    if key in _TRANSFER_ACTIONS:
        return "transfer"
    if key in _END_ACTIONS:
        return "end_call"
    return None


def _parse_veloxiam_body(data: Any) -> tuple[Optional[str], Optional[str]]:
    """Extrae (texto TTS, action de liberación) del JSON de llm-response.

    Contrato Veloxiam (ejemplos)::

        {"text": "Te transfiero…", "action": "transfer"}
        {"reply": "Adiós", "action": "end_call"}
        {"text": "…", "transfer": true}
        {"message": "…", "endCall": true}

    Si no hay ``action`` pero el texto habla de transferir, se infiere ``transfer``.
    """
    if isinstance(data, str):
        reply = _clean_tts_text(data)
        action = "transfer" if reply and _REPLY_TRANSFER_RE.search(reply) else None
        return reply, action

    if not isinstance(data, dict):
        return None, None

    raw = data.get("text") or data.get("reply") or data.get("message")
    if raw is None and isinstance(data.get("data"), dict):
        nested = data["data"]
        raw = nested.get("text") or nested.get("reply") or nested.get("message")
    reply = _clean_tts_text(raw) if isinstance(raw, str) else None

    action = _normalize_release_action(
        data.get("action") if isinstance(data.get("action"), str) else None
    )
    if action is None and isinstance(data.get("type"), str):
        action = _normalize_release_action(data["type"])
    if action is None and isinstance(data.get("status"), str):
        action = _normalize_release_action(data["status"])

    if action is None:
        if data.get("transfer") or data.get("shouldTransfer") or data.get("transferCall"):
            action = "transfer"
        elif (
            data.get("endCall")
            or data.get("hangup")
            or data.get("hangUp")
            or data.get("end") is True
        ):
            action = "end_call"

    if action is None and reply and _REPLY_TRANSFER_RE.search(reply):
        action = "transfer"

    return reply, action


def _ensure_farewell(reply: Optional[str], action: Optional[str]) -> Optional[str]:
    """Garantiza texto de despedida cuando hay transfer/end_call sin reply útil."""
    if reply:
        return reply
    if action and action in _DEFAULT_FAREWELL:
        return _DEFAULT_FAREWELL[action]
    return None


class NestJSAgentProcessor(FrameProcessor):
    """Hace de "LLM": al cerrarse el turno del usuario consulta a Veloxiam y
    emite la respuesta como texto hacia el TTS."""

    def __init__(self, *, session, http: httpx.AsyncClient, llm_url: str, **kwargs):
        super().__init__(**kwargs)
        self._session = session
        self._http = http
        self._llm_url = llm_url
        self._pending = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            if getattr(self._session, "release_pending", False) or getattr(
                self._session, "released", False
            ):
                await self.push_frame(frame, direction)
                return
            logger.info("[agent] ⚡ Barge-in — cancelando petición en vuelo")
            await self._cancel_pending()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            await self._on_user_turn(frame.context)
            return

        await self.push_frame(frame, direction)

    async def _on_user_turn(self, context: LLMContext):
        if _session_is_dead(self._session) or getattr(
            self._session, "release_pending", False
        ):
            logger.info(
                f"[agent] Turno ignorado — pipeline en liberación/desconexión "
                f"session={self._session.id}"
            )
            return

        user_text = _latest_user_text(context)
        if not user_text:
            return
        confidence = getattr(self._session, "last_user_confidence", None)
        self._session.add_turn("user", user_text, confidence)
        conf_pct = f"{confidence * 100:.0f}%" if confidence is not None else "?"
        logger.info(f"[STT→usuario] {user_text!r}  conf={conf_pct}")

        if _USER_TRANSFER_RE.search(user_text):
            self._session.expect_transfer = True
            logger.info(
                f"[agent] Pedido de transfer detectado — session={self._session.id}"
            )

        await self._cancel_pending()
        self._pending = self.create_task(self._respond(user_text))

    async def _respond(self, user_text: str):
        t0 = time.perf_counter()
        action: Optional[str] = None
        try:
            reply, action, http_status = await self._call_veloxiam(user_text)
            # Si el usuario pidió transfer y Veloxiam no mandó action, igual liberamos.
            if (
                action is None
                and getattr(self._session, "expect_transfer", False)
                and reply
            ):
                action = "transfer"
                logger.info(
                    f"[agent] action=transfer inferida por pedido del usuario "
                    f"session={self._session.id}"
                )
            llm_ms = (time.perf_counter() - t0) * 1000
            action_log = f" action={action}" if action else ""
            if reply:
                logger.info(
                    f"[Veloxiam] respondió HTTP {http_status} en {llm_ms:.0f}ms — "
                    f"session={self._session.id}{action_log} reply={reply!r}"
                )
            else:
                logger.warning(
                    f"[Veloxiam] respondió HTTP {http_status} en {llm_ms:.0f}ms — "
                    f"session={self._session.id}{action_log} reply vacía"
                )
        except asyncio.CancelledError:
            llm_ms = (time.perf_counter() - t0) * 1000
            logger.warning(
                f"[Veloxiam] cancelada tras {llm_ms:.0f}ms — session={self._session.id} "
                f"(barge-in o cierre de llamada)"
            )
            raise
        except httpx.TimeoutException:
            llm_ms = (time.perf_counter() - t0) * 1000
            logger.error(
                f"[Veloxiam] timeout tras {llm_ms:.0f}ms — session={self._session.id} "
                f"url={self._llm_url}"
            )
            reply = "Lo siento, tuve un problema. ¿Puedes repetirlo?"
            action = None
        except Exception as e:  # noqa: BLE001
            llm_ms = (time.perf_counter() - t0) * 1000
            logger.exception(
                f"[Veloxiam] error tras {llm_ms:.0f}ms — session={self._session.id}: {e}"
            )
            reply = "Lo siento, tuve un problema. ¿Puedes repetirlo?"
            action = None

        # Tras transfer Telnyx a veces corta el WS antes de llm-response.
        if _session_is_dead(self._session):
            logger.info(
                f"[agent] Respuesta descartada — transporte cerrado "
                f"session={self._session.id} action={action} reply={reply!r}"
            )
            logger.warning(
                "[agent] Veloxiam transfirió ANTES de que Pipecat pudiera despedirse. "
                "Orden correcto: 1) responder llm-response con texto+action=transfer "
                "2) esperar ~3s (TTS) 3) transferir y NO abrir otro /telnyx al bot"
            )
            return

        reply = _ensure_farewell(reply, action)

        # Liberar tras despedida para que el bot no siga hablando en la llamada transferida.
        if action:
            logger.info(
                f"[agent] Liberación programada action={action} "
                f"farewell={bool(reply)} session={self._session.id}"
            )
            begin_release(self._session, action, wait_for_speech=bool(reply))

        if reply:
            self._session.add_turn("assistant", reply, None)
            t1 = time.perf_counter()
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(LLMTextFrame(reply))
            await self.push_frame(LLMFullResponseEndFrame())
            tts_queue_ms = (time.perf_counter() - t1) * 1000
            logger.info(
                f"[TTS] encolado en {tts_queue_ms:.0f}ms — session={self._session.id}"
            )
        elif action:
            logger.info(
                f"[agent] Sin texto de despedida — liberando ya "
                f"action={action} session={self._session.id}"
            )

    async def _call_veloxiam(
        self, user_text: str
    ) -> tuple[Optional[str], Optional[str], int]:
        """POST a Veloxiam llm-response. Devuelve (texto, action, status HTTP)."""
        payload = {
            "sessionId": self._session.id,
            "callId": self._session.call_id,
            "tenantId": self._session.tenant_id,
            "from": self._session.phone_number,
            "text": user_text,
        }
        logger.info(
            f"[Veloxiam] → POST session={self._session.id} call_id={self._session.call_id} "
            f"bot={self._session.bot_id} text={user_text!r}"
        )
        t0 = time.perf_counter()
        resp = await self._http.post(
            self._llm_url,
            json=payload,
            headers={"x-bot-id": self._session.bot_id},
            timeout=20.0,
        )
        req_ms = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        data = resp.json()
        reply, action = _parse_veloxiam_body(data)
        logger.debug(
            f"[Veloxiam] HTTP {resp.status_code} body recibido en {req_ms:.0f}ms — "
            f"session={self._session.id} action={action}"
        )
        return reply, action, resp.status_code

    async def _cancel_pending(self):
        if self._pending is not None:
            task, self._pending = self._pending, None
            await self.cancel_task(task)
