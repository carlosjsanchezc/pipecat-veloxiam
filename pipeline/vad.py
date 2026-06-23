"""Silero VAD para barge-in / interrupción.

En Pipecat 1.3.0 el analizador de VAD NO se pasa al transporte: se pasa al
agregador de usuario (``LLMUserAggregatorParams(vad_analyzer=...)``). Ese
agregador es quien, usando este VAD, emite ``UserStartedSpeakingFrame`` e
``InterruptionFrame`` cuando el usuario empieza a hablar mientras el bot habla.
El servicio de TTS reacciona a ``InterruptionFrame`` cancelando el audio en
curso automáticamente: ahí está el barge-in. Ver pipeline/bridge.py.
"""

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams


def create_vad_analyzer() -> SileroVADAnalyzer:
    """Crea el analizador Silero VAD afinado para conversación en español."""
    return SileroVADAnalyzer(
        params=VADParams(
            confidence=0.6,   # umbral de confianza para considerar "voz"
            start_secs=0.2,   # voz sostenida antes de marcar "empezó a hablar"
            stop_secs=0.4,    # pausa antes de cerrar turno (teléfono/WhatsApp)
            min_volume=0.35,  # audio PSTN/WhatsApp suele llegar más bajo que mic de PC
        )
    )
