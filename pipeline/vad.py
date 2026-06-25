"""Silero VAD para barge-in / interrupción.

En Pipecat 1.3.0 el analizador de VAD NO se pasa al transporte: se pasa al
agregador de usuario (``LLMUserAggregatorParams(vad_analyzer=...)``). Ese
agregador es quien, usando este VAD, emite ``UserStartedSpeakingFrame`` e
``InterruptionFrame`` cuando el usuario empieza a hablar mientras el bot habla.
El servicio de TTS reacciona a ``InterruptionFrame`` cancelando el audio en
curso automáticamente: ahí está el barge-in. Ver pipeline/whatsapp.py y pipeline/telnyx.py.
"""

import os

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams


def create_whatsapp_vad_analyzer() -> SileroVADAnalyzer:
    """VAD para WhatsApp/WebRTC 16 kHz (config estable pre-Telnyx)."""
    return SileroVADAnalyzer(
        params=VADParams(
            confidence=0.7,
            start_secs=0.2,
            stop_secs=0.2,
            min_volume=0.6,
        )
    )


def create_telnyx_vad_analyzer() -> SileroVADAnalyzer:
    """VAD más sensible para audio PSTN Telnyx 8 kHz."""
    return SileroVADAnalyzer(
        params=VADParams(
            confidence=0.65,
            start_secs=0.15,
            stop_secs=0.2,
            min_volume=0.35,
        )
    )


def create_vad_analyzer() -> SileroVADAnalyzer:
    """Alias Telnyx — preferir ``create_telnyx_vad_analyzer`` / ``create_whatsapp_vad_analyzer``."""
    return create_telnyx_vad_analyzer()


def create_user_aggregator_params(vad: SileroVADAnalyzer) -> LLMUserAggregatorParams:
    """Parámetros del agregador de usuario: VAD + timeout de cierre de turno."""
    return LLMUserAggregatorParams(
        vad_analyzer=vad,
        # Tras dejar de hablar, cuánto esperar silencio antes de llamar al agente.
        user_turn_stop_timeout=float(os.getenv("USER_TURN_STOP_TIMEOUT_SEC", "1.0")),
    )
