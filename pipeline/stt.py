"""STT con Deepgram en streaming (tiempo real).

- ``create_stt(cfg)`` devuelve el servicio Deepgram configurado con el idioma
  del bot y con
  resultados intermedios (interim) activados, de modo que emite transcripción
  parcial "palabra por palabra" además de la final.
- ``TranscriptionTap`` es un FrameProcessor que se coloca justo después del STT
  para registrar cada transcripción (parcial y final) junto con su confidence
  score en la sesión, sin alterar el flujo del pipeline.

El confidence de Deepgram viaja en ``TranscriptionFrame.result`` (el resultado
crudo del SDK). De ahí lo extraemos con ``transcription_confidence``.
"""

import audioop
import os
import time
from typing import Optional

from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    OutputAudioRawFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService

from pipeline.config import BotConfig
from pipeline.lang import to_language


def create_stt(cfg: BotConfig, sample_rate: int = 16000) -> DeepgramSTTService:
    """Crea el servicio Deepgram STT en streaming, con el idioma/modelo del bot.

    ``sample_rate`` por defecto 16000 (WhatsApp/WebRTC). Telefonía Telnyx usa 8000.
    """
    language = to_language(cfg.language)
    logger.info(
        f"[Deepgram] STT init — bot_id={cfg.bot_id} model={cfg.deepgram_model} "
        f"lang={language} sample_rate={sample_rate}"
    )

    return DeepgramSTTService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        sample_rate=sample_rate,
        settings=DeepgramSTTService.Settings(
            model=cfg.deepgram_model,
            language=language,
            interim_results=True,
            smart_format=True,
            punctuate=True,
            # Sin endpointing/vad_events: el cierre de turno lo lleva Silero (pipeline/vad.py).
            endpointing=False,
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


class AudioInputTap(FrameProcessor):
    """Observa audio crudo del transporte para diagnosticar si WebRTC envía media."""

    _SPEECH_RMS = 1500  # umbral aprox. de voz en PCM s16le

    def __init__(self, session_id: str, **kwargs):
        super().__init__(**kwargs)
        self._session_id = session_id
        self._chunks = 0
        self._rms_max = 0
        self._last_log = 0.0
        self._last_speech_log = 0.0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            self._chunks += 1
            rms = 0
            if frame.audio:
                rms = audioop.rms(frame.audio, 2)
                self._rms_max = max(self._rms_max, rms)
                now = time.monotonic()
                if rms >= self._SPEECH_RMS and now - self._last_speech_log >= 2.0:
                    logger.info(
                        f"[audio-in] voz detectada rms={rms} "
                        f"session={self._session_id}"
                    )
                    self._last_speech_log = now
            now = time.monotonic()
            if now - self._last_log >= 5.0:
                if self._chunks == 0:
                    logger.warning(
                        f"[audio-in] session={self._session_id} sin chunks en 5s "
                        f"— WebRTC no está enviando audio entrante"
                    )
                else:
                    logger.info(
                        f"[audio-in] session={self._session_id} "
                        f"chunks={self._chunks} rms_max={self._rms_max} "
                        f"sr={frame.sample_rate}"
                    )
                self._chunks = 0
                self._rms_max = 0
                self._last_log = now

        await self.push_frame(frame, direction)


class AudioOutputTap(FrameProcessor):
    """Observa audio TTS hacia el transporte (diagnóstico WebRTC saliente)."""

    def __init__(self, session_id: str, **kwargs):
        super().__init__(**kwargs)
        self._session_id = session_id
        self._chunks = 0
        self._bytes = 0
        self._last_log = 0.0
        self._early_chunks = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, OutputAudioRawFrame):
            nbytes = len(frame.audio or b"")
            self._chunks += 1
            self._bytes += nbytes
            if self._early_chunks < 12:
                self._early_chunks += 1
                logger.info(
                    f"[audio-out] chunk #{self._early_chunks} "
                    f"bytes={nbytes} sr={frame.sample_rate} "
                    f"session={self._session_id}"
                )
            now = time.monotonic()
            if now - self._last_log >= 5.0:
                if self._chunks:
                    logger.info(
                        f"[audio-out] session={self._session_id} "
                        f"chunks={self._chunks} bytes={self._bytes} "
                        f"sr={frame.sample_rate}"
                    )
                self._chunks = 0
                self._bytes = 0
                self._last_log = now

        await self.push_frame(frame, direction)


class PipelineErrorTap(FrameProcessor):
    """Registra ErrorFrame del pipeline (p. ej. Deepgram/Cartesia desconectados)."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, ErrorFrame):
            logger.error(f"[pipeline-error] {frame.error!r} fatal={frame.fatal}")

        await self.push_frame(frame, direction)


class UserSpeechTap(FrameProcessor):
    """Registra cuándo el VAD detecta voz del usuario (antes del STT final)."""

    def __init__(self, session, **kwargs):
        super().__init__(**kwargs)
        self._session = session

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._session.user_spoke = True
            logger.info(f"[VAD] usuario hablando — session={self._session.id}")
        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.info(f"[VAD] usuario en silencio — session={self._session.id}")

        await self.push_frame(frame, direction)


class TranscriptionTap(FrameProcessor):
    """Observa las transcripciones STT, las registra con su confidence en la
    sesión y reenvía todos los frames intactos (no consume nada)."""

    def __init__(self, session, **kwargs):
        super().__init__(**kwargs)
        self._session = session

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            self._session.user_spoke = True
            conf = transcription_confidence(frame)
            logger.info(f"[STT final] ({conf}) {frame.text!r}")
            # Guardamos el confidence de la última final para asociarlo al turno
            # de usuario cuando el agregador cierre el turno (ver bridge.py).
            self._session.last_user_confidence = conf
        elif isinstance(frame, InterimTranscriptionFrame):
            if frame.text and frame.text.strip():
                self._session.user_spoke = True
            conf = transcription_confidence(frame)
            if frame.text and frame.text.strip():
                logger.info(f"[STT interim] ({conf}) {frame.text!r}")

        await self.push_frame(frame, direction)
