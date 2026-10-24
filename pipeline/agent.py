"""Procesador del agente Veloxiam (POST llm-response → TTS).

Compartido por WhatsApp y Telnyx; cada transporte monta su propio pipeline.

Si Veloxiam responde con ``action`` de transferencia o fin de llamada,
se encola la despedida (si hay texto) y se libera el pipeline para que
Pipecat deje de hablar aunque el media stream siga abierto.
"""

import asyncio
import os
import re
import time
from typing import Any, Optional

import httpx
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    TTSSpeakFrame,
    TTSUpdateSettingsFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.settings import TTSSettings
from pipecat.transcriptions.language import Language

from pipeline.audio_clip import AudioClipEndFrame, load_audio_clip
from pipeline.lang import base_code, pick_turn_language
from pipeline.release import begin_release

_PLACEHOLDER_RE = re.compile(r"//[^/]+//")
# Tags tipo Cartesia/Veloxiam que no deben sintetizarse.
# Veloxiam a veces la manda sin < >: emotion value="neutral"/
_EMOTION_TAG_RE = re.compile(
    r"<emotion\b[^>]*/?>|\bemotion\s+value\s*=\s*\"[^\"]*\"\s*/?>?", re.IGNORECASE
)
_XML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")

# Acciones de Veloxiam que deben cortar el pipeline de voz.
_TRANSFER_ACTIONS = frozenset(
    {"transfer", "transfer_call", "call_transfer", "redirect", "transferring", "transferred"}
)
_END_ACTIONS = frozenset(
    {"end", "end_call", "hangup", "hang_up", "disconnect", "close"}
)

# Si Veloxiam no manda ``action`` pero el texto es claramente una despedida de transfer.
_REPLY_TRANSFER_RE = re.compile(
    r"(?i)("
    r"te (estoy )?(paso|pasando|transfiero|transfiriendo|conectando|enviando)|"
    r"transfiri\w*|"
    r"transferir\w*|"
    r"te conecto|"
    r"pasar(te|lo)? con|"
    r"solicitud para transfer|"
    r"transferirte|"
    r"paso a \w+|"
    r"comunic(o|arte) con"
    r")"
)

# Pedido del usuario de hablar con alguien / transferir.
_USER_TRANSFER_RE = re.compile(
    r"(?i)("
    r"transfer\w*|"
    r"comun[ií]que\w*|"
    r"pas[aá]me|"
    r"quiero (hablar )?con|"
    r"hable con|"
    r"con (un )?(agente|asesor|humano|operador)"
    r")"
)

_DEFAULT_FAREWELL = {
    "es": {
        "transfer": "Claro, te transfiero en este momento.",
        "end_call": "Gracias por llamar. Hasta luego.",
    },
    "en": {
        "transfer": "Sure, I'm transferring you now.",
        "end_call": "Thanks for calling. Goodbye.",
    },
}

_ERROR_REPLY = {
    "es": "Lo siento, tuve un problema. ¿Puedes repetirlo?",
    "en": "Sorry, I had a problem. Could you repeat that?",
}


_LANGUAGE_NAMES = {
    "es": "español",
    "en": "inglés",
    "pt": "portugués",
    "fr": "francés",
    "it": "italiano",
    "de": "alemán",
}


def _with_language_hint(text: str, language: str) -> str:
    """Veloxiam ignora el campo ``language``: la instrucción va en el propio texto."""
    if os.getenv("REPLY_LANGUAGE_HINT", "true").strip().lower() in ("0", "false", "no"):
        return text
    name = _LANGUAGE_NAMES.get(language, language)
    return f"{text} RESPONDE EN EL IDIOMA {name.upper()}"


def _veloxiam_language(data: Any) -> Optional[str]:
    """``language`` opcional de llm-response: fuerza el idioma de la respuesta."""
    if not isinstance(data, dict):
        return None
    raw = data.get("language")
    if raw is None and isinstance(data.get("data"), dict):
        raw = data["data"].get("language")
    if not isinstance(raw, str) or not raw.strip():
        return None
    code = base_code(raw)
    try:
        Language(code)
    except ValueError:
        logger.warning(f"[agent] language={raw!r} de Veloxiam no soportado — se ignora")
        return None
    return code


def _content_to_text(content: Any) -> str:
    """Normaliza el ``content`` de un mensaje del contexto a texto plano."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return " ".join(p for p in parts if p).strip()
    return str(content)


def _latest_user_text(context: LLMContext) -> Optional[str]:
    """Devuelve el texto del último mensaje de usuario en el contexto."""
    for msg in reversed(context.get_messages()):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "user":
            content = (
                msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            )
            text = _content_to_text(content)
            return text or None
    return None


def _clean_tts_text(text: str) -> Optional[str]:
    """Elimina placeholders y tags XML (p. ej. emotion) que no deben sintetizarse."""
    cleaned = _PLACEHOLDER_RE.sub("", text)
    cleaned = _EMOTION_TAG_RE.sub("", cleaned)
    cleaned = _XML_TAG_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


def _session_is_dead(session) -> bool:
    """True si ya no debemos hablar (transfer, hangup o transporte cerrado)."""
    return bool(
        getattr(session, "released", False)
        or getattr(session, "disconnected", False)
    )


def _normalize_release_action(action: Optional[str]) -> Optional[str]:
    """Mapea action cruda a ``transfer`` / ``end_call``, o None si no aplica."""
    if not action:
        return None
    key = action.strip().lower().replace("-", "_").replace(" ", "_")
    if key in _TRANSFER_ACTIONS:
        return "transfer"
    if key in _END_ACTIONS:
        return "end_call"
    return None


def _audio_url(data: dict) -> Optional[str]:
    raw = data.get("audiofile")
    if raw is None and isinstance(data.get("data"), dict):
        raw = data["data"].get("audiofile")
    if isinstance(raw, str) and raw.strip().lower().startswith(("http://", "https://")):
        return raw.strip()
    return None


def _parse_veloxiam_body(
    data: Any,
) -> tuple[Optional[str], Optional[str], dict, Optional[str]]:
    """Extrae (texto TTS, action, meta extra, URL de audio) del JSON de llm-response.

    Contrato Veloxiam::

        {"text": "…", "action": "transfer"}   # NO transferir aún en Telnyx
        # Tras la despedida TTS, Pipecat POSTea TRANSFER_EXECUTE_URL y ahí sí ring.

        {"text": "Te dejo el audio.", "audiofile": "https://…"}
        # URL firmada (2 h) del audio del bot; suena después del texto.
    """
    meta: dict = {}
    if isinstance(data, str):
        reply = _clean_tts_text(data)
        action = "transfer" if reply and _REPLY_TRANSFER_RE.search(reply) else None
        return reply, action, meta, None

    if not isinstance(data, dict):
        return None, None, meta, None

    raw = data.get("text") or data.get("reply") or data.get("message")
    if raw is None and isinstance(data.get("data"), dict):
        nested = data["data"]
        raw = nested.get("text") or nested.get("reply") or nested.get("message")
    reply = _clean_tts_text(raw) if isinstance(raw, str) else None
    audio_url = _audio_url(data)

    action = _normalize_release_action(
        data.get("action") if isinstance(data.get("action"), str) else None
    )
    if action is None and isinstance(data.get("type"), str):
        action = _normalize_release_action(data["type"])
    if action is None and isinstance(data.get("status"), str):
        action = _normalize_release_action(data["status"])

    if action is None:
        if data.get("transfer") or data.get("shouldTransfer") or data.get("transferCall"):
            action = "transfer"
        elif (
            data.get("endCall")
            or data.get("hangup")
            or data.get("hangUp")
            or data.get("end") is True
        ):
            action = "end_call"

    if action is None and reply and _REPLY_TRANSFER_RE.search(reply):
        action = "transfer"

    # Metadatos útiles para transfer-execute (destino, etc.).
    for key in ("target", "destination", "to", "transferTo", "agentId", "agentName"):
        if key in data and data[key] is not None:
            meta[key] = data[key]

    return reply, action, meta, audio_url


def _ensure_farewell(
    reply: Optional[str], action: Optional[str], language: str = "es"
) -> Optional[str]:
    """Despedida hablada por Pipecat. En transfer siempre hay texto (default si falta)."""
    farewells = _DEFAULT_FAREWELL.get(language, _DEFAULT_FAREWELL["es"])
    if action == "transfer":
        # La despedida la dice Pipecat aquí; el ring lo dispara Veloxiam después del TTS.
        if reply and _REPLY_TRANSFER_RE.search(reply):
            return reply
        return farewells["transfer"]
    if reply:
        return reply
    if action and action in farewells:
        return farewells[action]
    return None


class NestJSAgentProcessor(FrameProcessor):
    """Hace de "LLM": al cerrarse el turno del usuario consulta a Veloxiam y
    emite la respuesta como texto hacia el TTS."""

    def __init__(
        self,
        *,
        session,
        http: httpx.AsyncClient,
        llm_url: str,
        audio_sample_rate: int,
        language: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._session = session
        self._http = http
        self._llm_url = llm_url
        self._audio_sample_rate = audio_sample_rate
        # Idioma de la conversación (cambia con lo que detecta Deepgram multi) y
        # el que tiene ahora Cartesia (arranca con el de la llamada).
        self._language = base_code(language)
        self._tts_language = self._language
        self._pending = None
        # True desde que se pide la respuesta a Veloxiam hasta que se encola al
        # TTS: una interrupción en ese tramo no corta nada audible.
        self._reply_not_started = False
        self._interrupted_during_wait = False
        self._user_speaking = False
        self._user_stopped_at = 0.0
        # Para que release.py pueda avisar transfer-execute tras la despedida.
        session.http_client = http

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            self._user_speaking = True
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._user_speaking = False
            self._user_stopped_at = time.monotonic()

        if isinstance(frame, InterruptionFrame):
            if getattr(self._session, "release_pending", False) or getattr(
                self._session, "released", False
            ):
                await self.push_frame(frame, direction)
                return
            if self._pending is not None and self._reply_not_started:
                # El bot aún no habla: si el usuario dice algo nuevo, _on_user_turn
                # reemplaza la petición; si era ruido, la respuesta se retiene y suena.
                self._interrupted_during_wait = True
                logger.info(
                    "[agent] Voz del usuario mientras se espera a Veloxiam — "
                    "se mantiene la petición"
                )
                await self.push_frame(frame, direction)
                return
            logger.info("[agent] ⚡ Barge-in — cancelando petición en vuelo")
            await self._cancel_pending()
            self._session.audio_clip_pending = False
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            await self._on_user_turn(frame.context)
            return

        await self.push_frame(frame, direction)

    def _accepts_turns(self) -> bool:
        if _session_is_dead(self._session) or getattr(
            self._session, "release_pending", False
        ):
            logger.info(
                f"[agent] Turno ignorado — pipeline en liberación/desconexión "
                f"session={self._session.id}"
            )
            return False
        return True

    async def handle_user_text(self, text: str, language: Optional[str] = None) -> bool:
        """Turno de usuario recibido por API (no por voz): Veloxiam responde y suena.

        Devuelve False si la llamada ya se está liberando o cerrando.
        """
        if not self._accepts_turns():
            return False
        self._session.add_turn("user", text, None)
        if language and base_code(language) != self._language:
            logger.info(
                f"[idioma] {self._language} → {base_code(language)} (API) "
                f"session={self._session.id}"
            )
            self._language = base_code(language)
        logger.info(f"[API→usuario] {text!r} lang={self._language}")
        await self._start_reply(text)
        return True

    async def _on_user_turn(self, context: LLMContext):
        if not self._accepts_turns():
            return

        user_text = _latest_user_text(context)
        if not user_text:
            return
        confidence = getattr(self._session, "last_user_confidence", None)
        self._session.add_turn("user", user_text, confidence)
        conf_pct = f"{confidence * 100:.0f}%" if confidence is not None else "?"
        words_by_lang = self._session.turn_language_words
        self._session.turn_language_words = type(words_by_lang)()
        turn_language = pick_turn_language(words_by_lang, self._language)
        if turn_language and turn_language != self._language:
            logger.info(
                f"[idioma] {self._language} → {turn_language} "
                f"(palabras={dict(words_by_lang)}) session={self._session.id}"
            )
            self._language = turn_language
        logger.info(
            f"[STT→usuario] {user_text!r}  conf={conf_pct} lang={self._language}"
        )
        await self._start_reply(user_text)

    async def _start_reply(self, user_text: str):
        if _USER_TRANSFER_RE.search(user_text):
            self._session.expect_transfer = True
            logger.info(
                f"[agent] Pedido de transfer detectado — session={self._session.id}"
            )

        await self._cancel_pending()
        self._reply_not_started = True
        self._interrupted_during_wait = False
        self._pending = self.create_task(self._respond(user_text, self._language))

    async def _wait_user_quiet(self):
        """Retiene la respuesta hasta que el usuario calle y pase un margen sin
        turno nuevo (si llega uno, _on_user_turn cancela esta tarea)."""
        grace = float(os.getenv("REPLY_HOLD_GRACE_SEC", "1.0"))
        max_wait = float(os.getenv("REPLY_HOLD_MAX_SEC", "15"))
        t0 = time.monotonic()
        while time.monotonic() - t0 < max_wait:
            if not self._user_speaking and time.monotonic() - self._user_stopped_at >= grace:
                break
            await asyncio.sleep(0.05)
        logger.info(
            f"[agent] Respuesta retenida {(time.monotonic() - t0) * 1000:.0f}ms "
            f"(sin turno nuevo) — se reproduce session={self._session.id}"
        )

    async def _respond(self, user_text: str, language: str):
        try:
            await self._respond_inner(user_text, language)
        finally:
            self._reply_not_started = False

    async def _respond_inner(self, user_text: str, language: str):
        t0 = time.perf_counter()
        action: Optional[str] = None
        audio_url: Optional[str] = None
        try:
            (
                reply,
                action,
                meta,
                audio_url,
                http_status,
                reply_language,
            ) = await self._call_veloxiam(user_text, language)
            if reply_language:
                language = reply_language
            # Si el usuario pidió transfer y Veloxiam no mandó action, igual liberamos.
            if (
                action is None
                and getattr(self._session, "expect_transfer", False)
                and reply
            ):
                action = "transfer"
                logger.info(
                    f"[agent] action=transfer inferida por pedido del usuario "
                    f"session={self._session.id}"
                )
            llm_ms = (time.perf_counter() - t0) * 1000
            action_log = f" action={action}" if action else ""
            if audio_url:
                action_log += " +audiofile"
            if reply:
                logger.info(
                    f"[Veloxiam] respondió HTTP {http_status} en {llm_ms:.0f}ms — "
                    f"session={self._session.id}{action_log} reply={reply!r}"
                )
            else:
                logger.warning(
                    f"[Veloxiam] respondió HTTP {http_status} en {llm_ms:.0f}ms — "
                    f"session={self._session.id}{action_log} reply vacía"
                )
        except asyncio.CancelledError:
            llm_ms = (time.perf_counter() - t0) * 1000
            logger.warning(
                f"[Veloxiam] cancelada tras {llm_ms:.0f}ms — session={self._session.id} "
                f"(barge-in o cierre de llamada)"
            )
            raise
        except httpx.TimeoutException:
            llm_ms = (time.perf_counter() - t0) * 1000
            logger.error(
                f"[Veloxiam] timeout tras {llm_ms:.0f}ms — session={self._session.id} "
                f"url={self._llm_url}"
            )
            reply = _ERROR_REPLY.get(language, _ERROR_REPLY["es"])
            action = None
            meta = {}
        except Exception as e:  # noqa: BLE001
            llm_ms = (time.perf_counter() - t0) * 1000
            logger.exception(
                f"[Veloxiam] error tras {llm_ms:.0f}ms — session={self._session.id}: {e}"
            )
            reply = _ERROR_REPLY.get(language, _ERROR_REPLY["es"])
            action = None
            meta = {}

        # Si Veloxiam ya transfirió (ring), el stream muere y no hay despedida audible.
        if _session_is_dead(self._session):
            logger.info(
                f"[agent] Respuesta descartada — transporte cerrado "
                f"session={self._session.id} action={action} reply={reply!r}"
            )
            logger.warning(
                "[agent] Veloxiam NO debe transferir en llm-response. "
                "Debe devolver action=transfer, Pipecat dice 'te transfiero', "
                "y SOLO entonces Veloxiam ejecuta el transfer en TRANSFER_EXECUTE_URL "
                "(ahí empieza el ring)."
            )
            return

        if self._interrupted_during_wait:
            await self._wait_user_quiet()
            if _session_is_dead(self._session):
                return
        self._reply_not_started = False

        reply = _ensure_farewell(reply, action, language)

        # Mientras el clip esté pendiente, CallReleaseTap no libera con el
        # BotStoppedSpeaking del texto: espera a AudioClipEndFrame.
        if audio_url:
            self._session.audio_clip_pending = True
        has_speech = bool(reply or audio_url)

        if action == "transfer":
            self._session.transfer_execute_pending = True
            self._session.transfer_meta = meta or {}
            logger.info(
                f"[agent] Despedida en Pipecat; transfer Telnyx tras TTS "
                f"session={self._session.id}"
            )
            begin_release(self._session, action, wait_for_speech=has_speech)
        elif action:
            begin_release(self._session, action, wait_for_speech=has_speech)

        if reply:
            self._session.add_turn("assistant", reply, None)
            t1 = time.perf_counter()
            await self._set_tts_language(language)
            # Respuesta completa en una sola generación de Cartesia: partida por
            # frases, las uniones entre fragmentos suenan con saltos.
            await self.push_frame(TTSSpeakFrame(reply))
            tts_queue_ms = (time.perf_counter() - t1) * 1000
            logger.info(
                f"[TTS] encolado en {tts_queue_ms:.0f}ms — session={self._session.id}"
            )
        elif action and not audio_url:
            logger.info(
                f"[agent] Sin texto de despedida — liberando ya "
                f"action={action} session={self._session.id}"
            )

        if audio_url:
            await self._play_audio_clip(audio_url)

    async def _set_tts_language(self, language: str):
        """Cambia el idioma de Cartesia; se aplica en orden, antes del TTSSpeakFrame."""
        if language == self._tts_language:
            return
        logger.info(
            f"[TTS] idioma {self._tts_language} → {language} session={self._session.id}"
        )
        self._tts_language = language
        await self.push_frame(
            TTSUpdateSettingsFrame(delta=TTSSettings(language=Language(language)))
        )

    async def _play_audio_clip(self, url: str):
        """Descarga el audio (mientras suena el texto) y lo encola tras el TTS."""
        try:
            clip = await load_audio_clip(self._http, url, self._audio_sample_rate)
        except asyncio.CancelledError:
            self._session.audio_clip_pending = False
            raise
        except Exception as e:  # noqa: BLE001
            logger.error(
                f"[audio-clip] No se pudo cargar el audio — session={self._session.id}: {e}"
            )
            clip = None

        if _session_is_dead(self._session):
            self._session.audio_clip_pending = False
            return

        if clip and clip.frames:
            self._session.add_turn("assistant", "[audio]", None)
            for frame in clip.frames:
                await self.push_frame(frame)
            logger.info(
                f"[audio-clip] Encolado {clip.duration_secs:.1f}s — "
                f"session={self._session.id}"
            )
        # Siempre se empuja el fin de clip: si falló la carga, libera igualmente
        # una despedida pendiente detrás del texto.
        await self.push_frame(AudioClipEndFrame())

    async def _call_veloxiam(
        self, user_text: str, language: str
    ) -> tuple[Optional[str], Optional[str], dict, Optional[str], int, Optional[str]]:
        """POST a Veloxiam llm-response.

        Devuelve (texto, action, meta, audio, status, idioma forzado por Veloxiam).
        """
        payload = {
            "sessionId": self._session.id,
            "callId": self._session.call_id,
            "tenantId": self._session.tenant_id,
            "from": self._session.phone_number,
            "text": _with_language_hint(user_text, language),
            "language": language,
        }
        logger.info(
            f"[Veloxiam] → POST session={self._session.id} call_id={self._session.call_id} "
            f"bot={self._session.bot_id} lang={language} text={payload['text']!r}"
        )
        t0 = time.perf_counter()
        resp = await self._http.post(
            self._llm_url,
            json=payload,
            headers={"x-bot-id": self._session.bot_id},
            timeout=20.0,
        )
        req_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[Veloxiam] ← HTTP {resp.status_code} en {req_ms:.0f}ms "
            f"session={self._session.id} body={resp.text[:4000]}"
        )
        resp.raise_for_status()
        data = resp.json()
        reply, action, meta, audio_url = _parse_veloxiam_body(data)
        return reply, action, meta, audio_url, resp.status_code, _veloxiam_language(data)

    async def _cancel_pending(self):
        if self._pending is not None:
            task, self._pending = self._pending, None
            await self.cancel_task(task)
