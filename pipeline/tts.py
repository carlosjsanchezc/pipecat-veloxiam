"""TTS con Cartesia, con voz e idioma por bot (multi-tenant).

Cartesia produce audio raw PCM que el transporte WebRTC reempaqueta y envía a
Meta. El barge-in es automático: el servicio cancela la síntesis en curso al
recibir un ``InterruptionFrame`` generado por el VAD del agregador de usuario.
"""

import os

from loguru import logger
from pipecat.services.cartesia.tts import CartesiaTTSService

from pipeline.config import BotConfig
from pipeline.lang import to_language


def create_tts(cfg: BotConfig, sample_rate: int = 16000) -> CartesiaTTSService:
    """Crea el servicio Cartesia TTS con la voz/idioma/modelo del bot.

    ``sample_rate`` por defecto 16000 (WhatsApp/WebRTC). Telefonía Telnyx usa
    8000 (PCMU), por lo que el transporte Telnyx pasa sample_rate=8000.
    """
    if not cfg.voice_id:
        raise RuntimeError(f"Bot {cfg.bot_id} sin voice_id en su configuración.")

    logger.info(
        f"[Cartesia] TTS init — bot_id={cfg.bot_id} voice_id={cfg.voice_id} "
        f"model={cfg.cartesia_model} lang={to_language(cfg.language)} "
        f"sample_rate={sample_rate}"
    )

    return CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        voice_id=cfg.voice_id,
        model=cfg.cartesia_model,
        sample_rate=sample_rate,
        encoding="pcm_s16le",
        container="raw",
        max_buffer_delay_ms=100,
        cartesia_version="2026-03-01",
        params=CartesiaTTSService.InputParams(language=to_language(cfg.language)),
    )
