"""Liberación del pipeline tras transfer / end_call.

Flujo transfer (despedida en Pipecat, ring en Veloxiam después)::

1. Veloxiam responde ``action=transfer`` en llm-response **sin** transferir aún.
2. Pipecat dice la despedida por TTS.
3. Al terminar el TTS → POST ``TRANSFER_EXECUTE_URL`` (Veloxiam hace el transfer / ring).
4. Se cancela el pipeline (deja de hablar; no cuelga la llamada).
"""

import asyncio
import os
from typing import Any, Optional

from loguru import logger

from pipecat.frames.frames import BotStoppedSpeakingFrame, Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


def begin_release(session, reason: str, *, wait_for_speech: bool) -> None:
    """Marca la sesión para liberar el pipeline (ya o tras el TTS)."""
    if getattr(session, "released", False):
        return
    session.release_reason = reason
    if wait_for_speech:
        session.release_pending = True
        max_sec = float(os.getenv("RELEASE_MAX_SEC", "15"))
        asyncio.create_task(_release_watchdog(session, max_sec))
        logger.info(
            f"[release] Esperando fin de TTS antes de liberar "
            f"reason={reason} session={session.id}"
        )
    else:
        asyncio.create_task(_finish_release(session, reason))


async def _release_watchdog(session, max_sec: float) -> None:
    await asyncio.sleep(max_sec)
    if getattr(session, "release_pending", False):
        logger.warning(
            f"[release] Watchdog ({max_sec:.0f}s) — liberando sin BotStoppedSpeaking "
            f"session={session.id}"
        )
        await _finish_release(session, getattr(session, "release_reason", "timeout"))


def _transfer_execute_url() -> Optional[str]:
    """URL para que Veloxiam ejecute el transfer tras la despedida TTS."""
    explicit = (os.getenv("TRANSFER_EXECUTE_URL") or "").strip()
    if explicit:
        return explicit
    llm = (os.getenv("LLM_RESPONSE_URL") or "").strip()
    if not llm:
        return None
    # .../llm-response → .../transfer-execute
    if llm.rstrip("/").endswith("llm-response"):
        return llm.rstrip("/").removesuffix("llm-response") + "transfer-execute"
    return None


async def notify_transfer_execute(session) -> None:
    """Avisa a Veloxiam: la despedida ya sonó; ahora sí transferir (ring)."""
    if getattr(session, "transfer_notified", False):
        return
    session.transfer_notified = True

    url = _transfer_execute_url()
    http = getattr(session, "http_client", None)
    if not url or http is None:
        logger.warning(
            f"[release] Sin TRANSFER_EXECUTE_URL/http — Veloxiam no recibirá "
            f"señal post-despedida session={session.id}. "
            f"Configura TRANSFER_EXECUTE_URL o endpoint .../transfer-execute"
        )
        return

    payload: dict[str, Any] = {
        "sessionId": session.id,
        "callId": session.call_id,
        "tenantId": session.tenant_id,
        "from": session.phone_number,
        "action": "transfer",
        "ready": True,
    }
    meta = getattr(session, "transfer_meta", None) or {}
    if isinstance(meta, dict):
        payload.update({k: v for k, v in meta.items() if k not in payload})

    logger.info(
        f"[release] → POST transfer-execute session={session.id} "
        f"call_id={session.call_id} url={url}"
    )
    try:
        resp = await http.post(
            url,
            json=payload,
            headers={"x-bot-id": session.bot_id},
            timeout=15.0,
        )
        logger.info(
            f"[release] transfer-execute HTTP {resp.status_code} "
            f"session={session.id}"
        )
    except Exception as e:  # noqa: BLE001
        logger.error(
            f"[release] transfer-execute falló session={session.id}: {e}"
        )


async def _finish_release(session, reason: str) -> None:
    """Tras despedida: si es transfer, avisar a Veloxiam; luego cancelar pipeline."""
    if getattr(session, "released", False):
        return
    session.release_pending = False

    if reason == "transfer" or getattr(session, "transfer_execute_pending", False):
        await notify_transfer_execute(session)

    await _cancel_worker(session, reason)


async def _cancel_worker(session, reason: str) -> None:
    if getattr(session, "released", False):
        return
    session.release_pending = False
    session.released = True
    worker = getattr(session, "worker", None)
    if worker is None:
        logger.warning(f"[release] Sin worker — session={session.id} reason={reason}")
        return
    logger.info(f"[release] Cancelando pipeline reason={reason} session={session.id}")
    try:
        await worker.cancel(reason=f"release:{reason}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[release] Error cancelando worker session={session.id}: {e}")


class CallReleaseTap(FrameProcessor):
    """Tras la despedida TTS: avisa transfer-execute y cancela el worker."""

    def __init__(self, session, **kwargs):
        super().__init__(**kwargs)
        self._session = session

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStoppedSpeakingFrame) and getattr(
            self._session, "release_pending", False
        ):
            reason = getattr(self._session, "release_reason", "release")
            logger.info(
                f"[release] TTS despedida terminado — reason={reason} "
                f"session={self._session.id}"
            )
            asyncio.create_task(_finish_release(self._session, reason))

        await self.push_frame(frame, direction)
