"""Bridge WebRTC <-> pipeline STT -> agente Veloxiam -> TTS.

Veloxiam negocia la llamada con Meta y pasa el SDP offer a /call/start; main.py
crea la ``SmallWebRTCConnection`` y la entrega aquí. Este módulo construye y
ejecuta el pipeline Pipecat sobre esa conexión.

Orden del pipeline::

    transport.input()      audio entrante (Opus) de Meta
      -> stt               Deepgram -> TranscriptionFrame
      -> TranscriptionTap  registra texto + confidence
      -> user_aggregator   VAD (barge-in) + cierre de turno -> LLMContextFrame
      -> NestJSAgentProcessor   POST a llm-response, devuelve texto -> LLMTextFrame
      -> tts               Cartesia (voiceId del bot) -> audio
      -> transport.output()     audio saliente a Meta
      -> assistant_aggregator   registra la respuesta en el contexto

El barge-in lo gestiona el framework: el VAD del ``user_aggregator`` emite
``InterruptionFrame`` cuando el usuario habla encima del bot; el TTS corta el
audio y este procesador cancela cualquier petición a Veloxiam en vuelo.
"""

import asyncio
import os
import time
from typing import Any, Optional

import httpx
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.workers.runner import WorkerRunner

from pipeline.config import BotConfig
from pipeline.stt import TranscriptionTap, create_stt
from pipeline.tts import create_tts
from pipeline.vad import create_vad_analyzer


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


import re as _re

_PLACEHOLDER_RE = _re.compile(r"//[^/]+//")


def _clean_tts_text(text: str) -> Optional[str]:
    """Elimina placeholders tipo //logo// que no deben sintetizarse."""
    cleaned = _PLACEHOLDER_RE.sub("", text).strip()
    return cleaned or None


class NestJSAgentProcessor(FrameProcessor):
    """Hace de "LLM": al cerrarse el turno del usuario consulta a NestJS y
    emite la respuesta como texto hacia el TTS.

    Flujo síncrono: ``POST {LLM_RESPONSE_URL}`` con header ``x-bot-id`` -> texto.
    La llamada se ejecuta como tarea para no bloquear el pipeline (así puede
    procesar un ``InterruptionFrame`` si el usuario interrumpe mientras esperamos
    la respuesta); en ese caso la tarea se cancela y no se habla nada obsoleto.
    """

    def __init__(self, *, session, http: httpx.AsyncClient, llm_url: str, **kwargs):
        super().__init__(**kwargs)
        self._session = session
        self._http = http
        self._llm_url = llm_url  # URL fija del endpoint LLM de Veloxiam
        self._pending = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Barge-in / interrupción: cancela la petición en vuelo a NestJS.
        if isinstance(frame, InterruptionFrame):
            logger.info(f"[agent] ⚡ Barge-in — cancelando petición en vuelo")
            await self._cancel_pending()
            await self.push_frame(frame, direction)
            return

        # Fin de turno del usuario: el agregador empuja el contexto.
        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            await self._on_user_turn(frame.context)
            return  # consumimos el frame; nosotros generamos la respuesta

        await self.push_frame(frame, direction)

    async def _on_user_turn(self, context: LLMContext):
        user_text = _latest_user_text(context)
        if not user_text:
            return
        confidence = getattr(self._session, "last_user_confidence", None)
        self._session.add_turn("user", user_text, confidence)
        conf_pct = f"{confidence * 100:.0f}%" if confidence is not None else "?"
        logger.info(f"[STT→usuario] {user_text!r}  conf={conf_pct}")

        await self._cancel_pending()
        self._pending = self.create_task(self._respond(user_text))

    async def _respond(self, user_text: str):
        t0 = time.perf_counter()
        try:
            reply = await self._call_nestjs(user_text)
            llm_ms = (time.perf_counter() - t0) * 1000
            logger.info(f"[NestJS] {llm_ms:.0f}ms → {reply!r}")
        except Exception as e:  # noqa: BLE001 - queremos seguir hablando aunque falle
            llm_ms = (time.perf_counter() - t0) * 1000
            logger.exception(f"[NestJS] {llm_ms:.0f}ms → ERROR: {e}")
            reply = "Lo siento, tuve un problema. ¿Puedes repetirlo?"

        if not reply:
            logger.warning(f"[NestJS] Respuesta vacía — no se habla")
            return

        self._session.add_turn("assistant", reply, None)

        # Enmarcamos como respuesta de LLM para que el TTS la sintetice y el
        # assistant_aggregator la registre en el contexto.
        t1 = time.perf_counter()
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame(reply))
        await self.push_frame(LLMFullResponseEndFrame())
        tts_queue_ms = (time.perf_counter() - t1) * 1000
        logger.info(f"[TTS] encolado en {tts_queue_ms:.0f}ms")

    async def _call_nestjs(self, user_text: str) -> Optional[str]:
        # El bot se identifica por header x-bot-id (no en la URL).
        payload = {
            "sessionId": self._session.id,
            "callId": self._session.call_id,
            "tenantId": self._session.tenant_id,
            "from": self._session.phone_number,
            "text": user_text,
        }
        resp = await self._http.post(
            self._llm_url,
            json=payload,
            headers={"x-bot-id": self._session.bot_id},
            timeout=20.0,
        )
        resp.raise_for_status()
        data = resp.json()
        # Aceptamos varias formas habituales de respuesta.
        if isinstance(data, str):
            return _clean_tts_text(data)
        raw = data.get("text") or data.get("reply") or data.get("message")
        return _clean_tts_text(raw) if raw else None

    async def _cancel_pending(self):
        if self._pending is not None:
            task, self._pending = self._pending, None
            await self.cancel_task(task)


async def run_call(
    connection: SmallWebRTCConnection,
    session,
    http: httpx.AsyncClient,
    cfg: BotConfig,
):
    """Construye y ejecuta el pipeline para una llamada conectada, usando la
    voz/idioma/modelo del bot resuelto (``cfg``).

    Bloquea hasta que la llamada termina (desconexión o cancelación). Pensada
    para lanzarse como tarea de fondo desde el callback del webhook.
    """
    llm_url = os.environ["LLM_RESPONSE_URL"]
    logger.info(
        f"[run_call] Construyendo pipeline — session={session.id} "
        f"stt_model={cfg.deepgram_model} tts_model={cfg.cartesia_model} "
        f"voice={cfg.voice_id} lang={cfg.language} llm_url={llm_url}"
    )

    SAMPLE_RATE = 16000  # consistente en todo el pipeline

    stt = create_stt(cfg)
    tts = create_tts(cfg)
    vad = create_vad_analyzer()
    logger.info(f"[run_call] STT/TTS/VAD creados — session={session.id}")

    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=SAMPLE_RATE,
            audio_out_sample_rate=SAMPLE_RATE,
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=vad),
    )

    agent = NestJSAgentProcessor(session=session, http=http, llm_url=llm_url)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            TranscriptionTap(session),
            user_aggregator,
            agent,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )
    logger.info(f"[run_call] Pipeline construido — session={session.id}")

    worker = PipelineWorker(pipeline, params=PipelineParams(enable_metrics=False))
    session.worker = worker

    greeting = (cfg.greeting or "").strip()
    if greeting:
        logger.info(f"[run_call] Saludo configurado: {greeting!r} — session={session.id}")
    else:
        logger.info(f"[run_call] Sin saludo inicial — session={session.id}")

    @transport.event_handler("on_client_connected")
    async def _on_connected(_transport, _client):
        logger.info(f"[WebRTC] Cliente conectado — session={session.id}")
        if greeting:
            async def _send_greeting():
                await asyncio.sleep(1.0)
                logger.info(f"[WebRTC] Encolando saludo TTS: {greeting!r}")
                session.add_turn("assistant", greeting, None)
                await worker.queue_frame(TTSSpeakFrame(greeting))
            asyncio.create_task(_send_greeting())
        else:
            logger.info(f"[WebRTC] Sin saludo; esperando que el usuario hable — session={session.id}")

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(_transport, _client):
        logger.info(f"[WebRTC] Cliente desconectado — session={session.id}")
        await worker.cancel(reason="client disconnected")

    logger.info(f"[run_call] Lanzando WorkerRunner — session={session.id}")
    runner = WorkerRunner()
    try:
        await runner.run(worker)
    finally:
        logger.info(f"[run_call] WorkerRunner finalizado — session={session.id}")
