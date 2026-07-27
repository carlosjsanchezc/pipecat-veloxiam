"""Liberación del pipeline tras transfer / end_call de Veloxiam.

Cuando el agente marca ``session.release_pending``, se espera a que termine
el TTS de despedida (``BotStoppedSpeakingFrame``) y se cancela el worker.
Así Pipecat deja de hablar sin depender de que Telnyx cierre el media stream.
"""

import asyncio
import os

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
        asyncio.create_task(_cancel_worker(session, reason))


async def _release_watchdog(session, max_sec: float) -> None:
    await asyncio.sleep(max_sec)
    if getattr(session, "release_pending", False):
        logger.warning(
            f"[release] Watchdog ({max_sec:.0f}s) — liberando sin BotStoppedSpeaking "
            f"session={session.id}"
        )
        await _cancel_worker(session, getattr(session, "release_reason", "timeout"))


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
    """Tras la despedida TTS, cancela el worker (transfer / end_call)."""

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
                f"[release] TTS terminado — liberando pipeline "
                f"reason={reason} session={self._session.id}"
            )
            # No await: cancelar el worker desde dentro del processor puede bloquear.
            asyncio.create_task(_cancel_worker(self._session, reason))

        await self.push_frame(frame, direction)
