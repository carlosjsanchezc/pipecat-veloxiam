"""Conversión de código de idioma (str) al enum Language de Pipecat."""

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
