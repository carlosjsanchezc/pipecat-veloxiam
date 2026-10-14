"""Colchón de audio para el TTS en streaming y monitor de cortes.

Cartesia entrega el audio a medida que lo genera. El transporte lo reproduce en
tiempo real desde el primer fragmento, así que si un fragmento llega más tarde
de lo que dura el anterior, la cola se vacía y el transporte rellena con
silencio (audio entrecortado). Los clips de audio del bot no sufren esto porque
llegan enteros de golpe.

- ``TTSJitterBuffer`` retiene el inicio de cada respuesta hasta acumular
  ``TTS_PREBUFFER_MS`` de audio (o hasta el fin de la respuesta) y luego deja
  pasar el resto: ese adelanto absorbe los retrasos de entrega posteriores.
- ``TTSUnderrunMonitor`` mide, por respuesta, cuánto silencio habría tenido que
  insertar el transporte porque el audio llegó tarde.
"""

import asyncio
import os
import time
from typing import Optional

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    SystemFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


def _duration(frame: TTSAudioRawFrame) -> float:
    return len(frame.audio) / (frame.sample_rate * frame.num_channels * 2)


class TTSJitterBuffer(FrameProcessor):
    """Retiene el inicio de cada respuesta del TTS para tener audio adelantado."""

    def __init__(self, prebuffer_ms: Optional[float] = None, **kwargs):
        super().__init__(**kwargs)
        if prebuffer_ms is None:
            prebuffer_ms = float(os.getenv("TTS_PREBUFFER_MS", "250"))
        self._prebuffer_secs = max(prebuffer_ms, 0.0) / 1000
        # Si el audio llega muy lento, no esperar indefinidamente al colchón.
        self._max_hold_secs = self._prebuffer_secs + 0.5
        self._holding = False
        self._streaming = False
        self._held: list[Frame] = []
        self._held_secs = 0.0
        self._timer: Optional[asyncio.TimerHandle] = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (InterruptionFrame, CancelFrame)):
            await self._reset()
            await self.push_frame(frame, direction)
            return

        if (
            direction == FrameDirection.UPSTREAM
            or isinstance(frame, SystemFrame)
            or self._prebuffer_secs <= 0
        ):
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, EndFrame):
            await self._release(ended=True)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TTSStartedFrame) and not self._holding:
            self._streaming = False

        if isinstance(frame, TTSAudioRawFrame) and not self._holding and not self._streaming:
            self._holding = True
            self._held_secs = 0.0
            self._timer = asyncio.get_running_loop().call_later(
                self._max_hold_secs, self._on_hold_timeout
            )

        if self._holding:
            self._held.append(frame)
            if isinstance(frame, TTSAudioRawFrame):
                self._held_secs += _duration(frame)
            if isinstance(frame, TTSStoppedFrame):
                await self._release(ended=True)
            elif self._held_secs >= self._prebuffer_secs:
                await self._release(ended=False)
            return

        await self.push_frame(frame, direction)
        if isinstance(frame, TTSStoppedFrame):
            self._streaming = False

    def _on_hold_timeout(self):
        self._timer = None
        if self._holding:
            self.create_task(self._release(ended=False))

    def _cancel_timer(self):
        timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()

    async def _release(self, *, ended: bool):
        if not self._holding and not self._held:
            return
        self._cancel_timer()
        held, self._held = self._held, []
        self._holding = False
        self._streaming = not ended
        for frame in held:
            await self.push_frame(frame)

    async def _reset(self):
        self._cancel_timer()
        self._held = []
        self._holding = False
        self._streaming = False

    async def cleanup(self):
        self._cancel_timer()
        await super().cleanup()


class TTSUnderrunMonitor(FrameProcessor):
    """Registra por respuesta los huecos de audio por entrega tardía del TTS."""

    _TOLERANCE_SECS = 0.03

    def __init__(self, session_id: str, **kwargs):
        super().__init__(**kwargs)
        self._session_id = session_id
        self._reset()

    def _reset(self):
        self._t0: Optional[float] = None
        self._audio_secs = 0.0
        self._gaps = 0
        self._gap_secs = 0.0
        self._max_gap = 0.0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            self._reset()
        elif isinstance(frame, TTSAudioRawFrame) and direction == FrameDirection.DOWNSTREAM:
            now = time.monotonic()
            if self._t0 is None:
                self._t0 = now
            # Cuánto audio debería haberse reproducido ya vs. cuánto ha llegado.
            deficit = (now - self._t0) - self._audio_secs
            if deficit > self._TOLERANCE_SECS:
                self._gaps += 1
                self._gap_secs += deficit
                self._max_gap = max(self._max_gap, deficit)
                # El transporte rellenó ese hueco con silencio: la reproducción se desplaza.
                self._t0 += deficit
            self._audio_secs += _duration(frame)
        elif isinstance(frame, TTSStoppedFrame) and self._t0 is not None:
            if self._gaps:
                logger.warning(
                    f"[tts-buffer] {self._audio_secs:.1f}s de audio con {self._gaps} "
                    f"corte(s): total {self._gap_secs * 1000:.0f}ms, máximo "
                    f"{self._max_gap * 1000:.0f}ms — session={self._session_id}"
                )
            else:
                logger.info(
                    f"[tts-buffer] {self._audio_secs:.1f}s de audio sin cortes — "
                    f"session={self._session_id}"
                )
            self._reset()

        await self.push_frame(frame, direction)
