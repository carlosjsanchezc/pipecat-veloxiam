"""STT con Deepgram en streaming (tiempo real).

- ``create_stt()`` devuelve el servicio Deepgram configurado en español con
  resultados intermedios (interim) activados, de modo que emite transcripción
  parcial "palabra por palabra" además de la final.
- ``TranscriptionTap`` es un FrameProcessor que se coloca justo después del STT
  para registrar cada transcripción (parcial y final) junto con su confidence
  score en la sesión, sin alterar el flujo del pipeline.

El confidence de Deepgram viaja en ``TranscriptionFrame.result`` (el resultado
crudo del SDK). De ahí lo extraemos con ``transcription_confidence``.
"""

import os
from typing import Optional

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transcriptions.language import Language


def create_stt() -> DeepgramSTTService:
    """Crea el servicio Deepgram STT en streaming, en español."""
    return DeepgramSTTService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        settings=DeepgramSTTService.Settings(
            model=os.getenv("DEEPGRAM_MODEL", "nova-2-general"),
            language=Language.ES,
            interim_results=True,  # transcripción parcial, palabra por palabra
            smart_format=True,
            punctuate=True,
        ),
    )


def transcription_confidence(frame: Frame) -> Optional[float]:
    """Extrae el confidence score (0..1) de un (Interim)TranscriptionFrame.

    Deepgram lo entrega en result.channel.alternatives[0].confidence. El SDK
    puede devolverlo como dict o como objeto, así que probamos ambos.
    """
    result = getattr(frame, "result", None)
    if result is None:
        return None
    # dict-style
    try:
        return float(result["channel"]["alternatives"][0]["confidence"])
    except (TypeError, KeyError, IndexError, ValueError):
        pass
    # object-style
    try:
        return float(result.channel.alternatives[0].confidence)
    except Exception:
        return None


class TranscriptionTap(FrameProcessor):
    """Observa las transcripciones STT, las registra con su confidence en la
    sesión y reenvía todos los frames intactos (no consume nada)."""

    def __init__(self, session, **kwargs):
        super().__init__(**kwargs)
        self._session = session

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            conf = transcription_confidence(frame)
            logger.info(f"[STT final] ({conf}) {frame.text!r}")
            # Guardamos el confidence de la última final para asociarlo al turno
            # de usuario cuando el agregador cierre el turno (ver bridge.py).
            self._session.last_user_confidence = conf
        elif isinstance(frame, InterimTranscriptionFrame):
            conf = transcription_confidence(frame)
            logger.debug(f"[STT interim] ({conf}) {frame.text!r}")

        await self.push_frame(frame, direction)
