"""Conversión de código de idioma (str) al enum Language de Pipecat."""

import os
from collections import Counter
from typing import Optional

from pipecat.frames.frames import TranscriptionFrame
from pipecat.transcriptions.language import Language


def to_language(code: str, default: Language = Language.ES) -> Language:
    """'es' -> Language.ES, 'es-419' -> Language.ES_419 (o base 'es'), etc."""
    if not code:
        return default
    try:
        return Language(code)
    except ValueError:
        try:
            return Language(code.split("-")[0])
        except ValueError:
            return default


def base_code(code) -> str:
    return str(getattr(code, "value", code) or "").split("-")[0].strip().lower()


def reply_languages() -> set[str]:
    """Idiomas a los que puede cambiar la respuesta según lo que detecta Deepgram."""
    raw = os.getenv("REPLY_LANGUAGES") or "es,en"
    return {base_code(c) for c in raw.split(",") if c.strip()}


def transcription_languages(frame: TranscriptionFrame) -> Counter:
    """Palabras por idioma (código base) de una transcripción final de Deepgram multi."""
    counts: Counter = Counter()
    try:
        words = frame.result.channel.alternatives[0].words or []
    except (AttributeError, IndexError):
        words = []
    for word in words:
        lang = base_code(getattr(word, "language", None))
        if lang:
            counts[lang] += 1
    if not counts and frame.language:
        counts[base_code(frame.language)] = max(len(frame.text.split()), 1)
    return counts


def pick_turn_language(counts: Counter, current: str) -> Optional[str]:
    """Idioma dominante del turno, o None si no hay evidencia suficiente para cambiar.

    Un turno corto ("ok", "yes") no cambia el idioma: necesita
    LANGUAGE_SWITCH_MIN_WORDS palabras en el idioma nuevo.
    """
    allowed = reply_languages()
    candidates = Counter({lang: n for lang, n in counts.items() if lang in allowed})
    if not candidates:
        return None
    lang, words = candidates.most_common(1)[0]
    if lang == base_code(current):
        return lang
    if words < int(os.getenv("LANGUAGE_SWITCH_MIN_WORDS", "2")):
        return None
    return lang
