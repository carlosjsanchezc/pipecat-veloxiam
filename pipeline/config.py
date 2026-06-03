"""Configuración de un bot para una llamada concreta.

Veloxiam maneja WhatsApp/Meta y envía estos datos a Pipecat en POST /call/start.
Pipecat solo los usa: voz/idioma/modelo para el TTS-STT, y bot_id para el header
``x-bot-id`` de la llamada al endpoint LLM.
"""

import os
from dataclasses import dataclass


@dataclass
class BotConfig:
    bot_id: str
    voice_id: str
    language: str = "es"
    cartesia_model: str = "sonic-2"
    deepgram_model: str = "nova-2-general"
    tenant_id: str = "default"
    greeting: str = ""

    @classmethod
    def from_request(cls, data: dict) -> "BotConfig":
        """Construye la config desde el body de POST /call/start."""
        return cls(
            bot_id=data["botId"],
            voice_id=data["voiceId"],
            language=data.get("language", "es"),
            cartesia_model=data.get("cartesiaModel", os.getenv("CARTESIA_MODEL", "sonic-2")),
            deepgram_model=data.get(
                "deepgramModel", os.getenv("DEEPGRAM_MODEL", "nova-2-general")
            ),
            tenant_id=data.get("tenantId", "default"),
            greeting=data.get("greeting", ""),
        )
