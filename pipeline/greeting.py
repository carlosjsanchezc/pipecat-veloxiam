"""Protección del saludo inicial contra barge-in (Silero VAD).

Durante el primer ``TTSSpeakFrame`` (o saludo IA en Telnyx) se ignoran
``InterruptionFrame`` para que el VAD no corte el saludo. Al terminar de
reproducirse el audio (``BotStoppedSpeakingFrame``) se rehabilita el barge-in.
"""

import asyncio
import os

from loguru import logger

from pipecat.frames.frames import BotStoppedSpeakingFrame, Frame, InterruptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


def begin_greeting(session) -> None:
    """Marca el inicio del saludo y programa un fallback por si no llega fin de TTS."""
    session.greeting_active = True
    session.greeting_done = False
    max_sec = float(os.getenv("GREETING_MAX_SEC", "45"))
    asyncio.create_task(_greeting_watchdog(session, max_sec))


async def _greeting_watchdog(session, max_sec: float) -> None:
    await asyncio.sleep(max_sec)
    if getattr(session, "greeting_active", False):
        session.greeting_active = False
        session.greeting_done = True
        logger.warning(
            f"[greeting] Watchdog ({max_sec:.0f}s) — barge-in habilitado "
            f"session={session.id}"
        )


def end_greeting(session) -> None:
    session.greeting_active = False
    session.greeting_done = True


class GreetingBargeInGate(FrameProcessor):
    """Suprime InterruptionFrame mientras ``session.greeting_active``."""

    def __init__(self, session, **kwargs):
        super().__init__(**kwargs)
        self._session = session

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame) and getattr(
            self._session, "greeting_active", False
        ):
            logger.info(
                f"[greeting] Barge-in ignorado durante saludo — session={self._session.id}"
            )
            return

        await self.push_frame(frame, direction)


class GreetingCompleteTap(FrameProcessor):
    """Detecta fin del saludo y rehabilita barge-in."""

    def __init__(self, session, **kwargs):
        super().__init__(**kwargs)
        self._session = session

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStoppedSpeakingFrame) and getattr(
            self._session, "greeting_active", False
        ):
            end_greeting(self._session)
            logger.info(
                f"[greeting] Saludo terminado — barge-in habilitado "
                f"session={self._session.id}"
            )

        await self.push_frame(frame, direction)
