"""Silero VAD + Smart Turn v3.2 para detección de turnos.

Silero detecta voz vs silencio y barge-in (``InterruptionFrame``).
Smart Turn (``LocalSmartTurnAnalyzerV3``) decide si el usuario **terminó** de
hablar tras una pausa — mejor que solo timeout o endpointing de Deepgram.

Pipecat 1.3 incluye el modelo **smart-turn-v3.2-cpu.onnx** embebido.

Requisito Pipecat: ``stop_secs=0.2`` en Silero (como en los datos de entrenamiento
de Smart Turn).
"""

import os

from loguru import logger

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

# Smart Turn exige stop_secs=0.2 en Silero.
_SMART_TURN_STOP_SECS = 0.2


def create_smart_turn_analyzer() -> LocalSmartTurnAnalyzerV3:
    """Smart Turn v3.2-cpu bundled con Pipecat 1.3."""
    cpu_count = int(os.getenv("SMART_TURN_CPU_COUNT", "1"))
    logger.info(f"[SmartTurn] v3.2-cpu bundled — cpu={cpu_count}")
    return LocalSmartTurnAnalyzerV3(cpu_count=cpu_count)


def create_whatsapp_vad_analyzer() -> SileroVADAnalyzer:
    """Silero para WhatsApp/WebRTC 16 kHz + Smart Turn."""
    return SileroVADAnalyzer(
        params=VADParams(
            confidence=float(os.getenv("WHATSAPP_VAD_CONFIDENCE", "0.7")),
            start_secs=float(os.getenv("WHATSAPP_VAD_START_SECS", "0.2")),
            stop_secs=_SMART_TURN_STOP_SECS,
            min_volume=float(os.getenv("WHATSAPP_VAD_MIN_VOLUME", "0.6")),
        )
    )


def create_telnyx_vad_analyzer() -> SileroVADAnalyzer:
    """Silero más sensible para PSTN Telnyx 8 kHz + Smart Turn."""
    return SileroVADAnalyzer(
        params=VADParams(
            confidence=float(os.getenv("TELNYX_VAD_CONFIDENCE", "0.65")),
            start_secs=float(os.getenv("TELNYX_VAD_START_SECS", "0.15")),
            stop_secs=_SMART_TURN_STOP_SECS,
            min_volume=float(os.getenv("TELNYX_VAD_MIN_VOLUME", "0.35")),
        )
    )


def create_vad_analyzer() -> SileroVADAnalyzer:
    """Alias Telnyx."""
    return create_telnyx_vad_analyzer()


def create_user_aggregator_params(vad: SileroVADAnalyzer) -> LLMUserAggregatorParams:
    """Agregador de usuario: Silero (VAD/barge-in) + Smart Turn v3.2 (fin de turno)."""
    turn_analyzer = create_smart_turn_analyzer()
    fallback_timeout = float(os.getenv("USER_TURN_STOP_TIMEOUT_SEC", "5.0"))

    return LLMUserAggregatorParams(
        vad_analyzer=vad,
        user_turn_strategies=UserTurnStrategies(
            stop=[
                TurnAnalyzerUserTurnStopStrategy(turn_analyzer=turn_analyzer),
            ],
        ),
        user_turn_stop_timeout=fallback_timeout,
    )
