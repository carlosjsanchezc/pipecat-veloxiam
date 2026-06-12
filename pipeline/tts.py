"""TTS con Cartesia, con voz e idioma por bot (multi-tenant).

Cartesia produce audio raw PCM que el transporte WebRTC reempaqueta y envía a
Meta. El barge-in es automático: el servicio cancela la síntesis en curso al
recibir un ``InterruptionFrame`` generado por el VAD del agregador de usuario.
"""

import os

from pipecat.services.cartesia.tts import CartesiaTTSService

from pipeline.config import BotConfig
from pipeline.lang import to_language


def create_tts(cfg: BotConfig) -> CartesiaTTSService:
    """Crea el servicio Cartesia TTS con la voz/idioma/modelo del bot."""
    if not cfg.voice_id:
        raise RuntimeError(f"Bot {cfg.bot_id} sin voice_id en su configuración.")

    return CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        voice_id=cfg.voice_id,
        model=cfg.cartesia_model,
        sample_rate=16000,
        encoding="pcm_s16le",
        container="raw",
        max_buffer_delay_ms=0,
        cartesia_version="2026-03-01",
        params=CartesiaTTSService.InputParams(language=to_language(cfg.language)),
    )
