"""TTS con Cartesia en español.

Cartesia produce audio raw PCM que el transporte WebRTC de Pipecat reempaqueta
y envía a Meta (resampleo/encoding los gestiona el transporte). El barge-in es
automático: el servicio cancela la síntesis en curso al recibir un
``InterruptionFrame`` generado por el VAD del agregador de usuario.
"""

import os

from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transcriptions.language import Language


def create_tts() -> CartesiaTTSService:
    """Crea el servicio Cartesia TTS configurado en español.

    Requiere CARTESIA_VOICE_ID: un voice_id de Cartesia con soporte de español.
    """
    voice_id = os.environ.get("CARTESIA_VOICE_ID")
    if not voice_id:
        raise RuntimeError(
            "Falta CARTESIA_VOICE_ID. Elige un voice_id en español del panel de "
            "Cartesia y ponlo en .env."
        )

    return CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        voice_id=voice_id,
        model=os.getenv("CARTESIA_MODEL", "sonic-2"),
        params=CartesiaTTSService.InputParams(language=Language.ES),
    )
