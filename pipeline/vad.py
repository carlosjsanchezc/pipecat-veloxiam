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


def create_vad_analyzer() -> SileroVADAnalyzer:
    """Crea el analizador Silero VAD afinado para conversación en español."""
    return SileroVADAnalyzer(
        params=VADParams(
            confidence=0.65,
            start_secs=0.15,  # reaccionar antes al habla (barge-in / turno)
            stop_secs=0.2,    # default Pipecat (requerido para turn detection + STT p99)
            min_volume=0.35,  # audio PSTN/WhatsApp suele llegar más bajo que mic de PC
        )
    )


def create_user_aggregator_params(vad: SileroVADAnalyzer) -> LLMUserAggregatorParams:
    """Parámetros del agregador de usuario: VAD + timeout de cierre de turno."""
    return LLMUserAggregatorParams(
        vad_analyzer=vad,
        # Tras dejar de hablar, cuánto esperar silencio antes de llamar al agente.
        user_turn_stop_timeout=float(os.getenv("USER_TURN_STOP_TIMEOUT_SEC", "1.0")),
    )
