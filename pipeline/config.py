"""Configuración de un bot para una llamada concreta.

Veloxiam maneja WhatsApp/Meta y envía estos datos a Pipecat en POST /call/start.
Pipecat solo los usa: voz/idioma/modelo para el TTS-STT, y bot_id para el header
``x-bot-id`` de la llamada al endpoint LLM.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger
from pipecat.transcriptions.language import Language


def default_language() -> str:
    return (os.getenv("DEFAULT_LANGUAGE") or "en").strip()


def default_deepgram_model() -> str:
    return (os.getenv("DEEPGRAM_MODEL") or "nova-3").strip()


def resolve_language(value: Optional[str]) -> str:
    """Idioma pedido al iniciar la llamada, o DEFAULT_LANGUAGE si falta o no es válido."""
    code = (value or "").strip()
    if not code:
        return default_language()
    try:
        Language(code)
        return code
    except ValueError:
        try:
            Language(code.split("-")[0])
            return code.split("-")[0]
        except ValueError:
            fallback = default_language()
            logger.warning(f"[config] language={code!r} no soportado — usando {fallback!r}")
            return fallback


@dataclass
class BotConfig:
    bot_id: str
    voice_id: str
    language: str = field(default_factory=default_language)
    cartesia_model: str = "sonic-3.5"
    # nova-2-phonecall solo soporta en/en-US; para español usar nova-2-general.
    deepgram_model: str = field(default_factory=default_deepgram_model)
    tenant_id: str = "default"
    greeting: str = ""
    # Outbound: si True, `greeting` es una instrucción para que la IA genere el
    # saludo (no se dice literal). Si False, se dice tal cual.
    is_ia_content: bool = False
    # Outbound: chatId para mantener el historial de la conversación.
    chat_id: str = ""

    @classmethod
    def from_request(cls, data: dict) -> "BotConfig":
        """Construye la config desde el body de POST /call/start."""
        return cls(
            bot_id=data["botId"],
            voice_id=data["voiceId"],
            language=resolve_language(data.get("language")),
            cartesia_model=data.get("cartesiaModel", os.getenv("CARTESIA_MODEL", "sonic-3.5")),
            deepgram_model=data.get("deepgramModel") or default_deepgram_model(),
            tenant_id=data.get("tenantId", "default"),
            greeting=data.get("greeting", "Hola, ¿cómo estás?"),
        )
