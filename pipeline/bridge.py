"""Bridge WebRTC <-> pipeline STT -> agente NestJS -> TTS.

Pipecat 1.3.0 trae el transporte WhatsApp (``pipecat.transports.whatsapp``) que
ya resuelve el webhook de Meta y el intercambio SDP offer/answer contra la Graph
API. Cuando una llamada se conecta, el ``WhatsAppClient`` nos entrega una
``SmallWebRTCConnection`` ya negociada. Este módulo construye y ejecuta el
pipeline Pipecat sobre esa conexión.

Orden del pipeline::

    transport.input()      audio entrante (Opus) de Meta
      -> stt               Deepgram -> TranscriptionFrame
      -> TranscriptionTap  registra texto + confidence
      -> user_aggregator   VAD (barge-in) + cierre de turno -> LLMContextFrame
      -> NestJSAgentProcessor   POST a NestJS, devuelve texto -> LLMTextFrame
      -> tts               Cartesia -> audio
      -> transport.output()     audio saliente a Meta
      -> assistant_aggregator   registra la respuesta en el contexto

El barge-in lo gestiona el framework: el VAD del ``user_aggregator`` emite
``InterruptionFrame`` cuando el usuario habla encima del bot; el TTS corta el
audio y este procesador cancela cualquier petición a NestJS en vuelo.
"""

import os
from typing import Any, Optional

import httpx
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext, LLMContextFrame
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.workers.runner import WorkerRunner

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


class NestJSAgentProcessor(FrameProcessor):
    """Hace de "LLM": al cerrarse el turno del usuario consulta a NestJS y
    emite la respuesta como texto hacia el TTS.

    Flujo síncrono: ``POST {NESTJS_AGENT_URL}/{botId}/message`` -> texto.
    La llamada se ejecuta como tarea para no bloquear el pipeline (así puede
    procesar un ``InterruptionFrame`` si el usuario interrumpe mientras esperamos
    la respuesta); en ese caso la tarea se cancela y no se habla nada obsoleto.
    """

    def __init__(self, *, session, http: httpx.AsyncClient, nestjs_url: str, **kwargs):
        super().__init__(**kwargs)
        self._session = session
        self._http = http
        self._nestjs_url = nestjs_url.rstrip("/")
        self._pending = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Barge-in / interrupción: cancela la petición en vuelo a NestJS.
        if isinstance(frame, InterruptionFrame):
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
        logger.info(f"[turno usuario] {user_text!r} (conf={confidence})")

        await self._cancel_pending()
        self._pending = self.create_task(self._respond(user_text))

    async def _respond(self, user_text: str):
        try:
            reply = await self._call_nestjs(user_text)
        except Exception as e:  # noqa: BLE001 - queremos seguir hablando aunque falle
            logger.error(f"NestJS falló: {e}")
            reply = "Lo siento, tuve un problema. ¿Puedes repetirlo?"

        if not reply:
            return

        self._session.add_turn("assistant", reply, None)
        logger.info(f"[turno agente] {reply!r}")

        # Enmarcamos como respuesta de LLM para que el TTS la sintetice y el
        # assistant_aggregator la registre en el contexto.
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame(reply))
        await self.push_frame(LLMFullResponseEndFrame())

    async def _call_nestjs(self, user_text: str) -> Optional[str]:
        url = f"{self._nestjs_url}/{self._session.bot_id}/message"
        payload = {
            "sessionId": self._session.id,
            "callId": self._session.call_id,
            "tenantId": self._session.tenant_id,
            "from": self._session.phone_number,
            "text": user_text,
        }
        resp = await self._http.post(url, json=payload, timeout=20.0)
        resp.raise_for_status()
        data = resp.json()
        # Aceptamos varias formas habituales de respuesta.
        if isinstance(data, str):
            return data
        return data.get("text") or data.get("reply") or data.get("message")

    async def _cancel_pending(self):
        if self._pending is not None:
            task, self._pending = self._pending, None
            await self.cancel_task(task)


async def run_call(
    connection: SmallWebRTCConnection,
    session,
    http: httpx.AsyncClient,
):
    """Construye y ejecuta el pipeline para una llamada conectada.

    Bloquea hasta que la llamada termina (desconexión o cancelación). Pensada
    para lanzarse como tarea de fondo desde el callback del webhook.
    """
    nestjs_url = os.environ["NESTJS_AGENT_URL"]

    stt = create_stt()
    tts = create_tts()
    vad = create_vad_analyzer()

    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=vad),
    )

    agent = NestJSAgentProcessor(session=session, http=http, nestjs_url=nestjs_url)

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

    worker = PipelineWorker(pipeline, params=PipelineParams(enable_metrics=False))
    session.worker = worker

    greeting = os.getenv("GREETING_TEXT", "").strip()

    @transport.event_handler("on_client_connected")
    async def _on_connected(_transport, _client):
        logger.info(f"Llamada conectada (session={session.id})")
        if greeting:
            session.add_turn("assistant", greeting, None)
            await worker.queue_frame(TTSSpeakFrame(greeting))

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(_transport, _client):
        logger.info(f"Llamada desconectada (session={session.id})")
        await worker.cancel(reason="client disconnected")

    runner = WorkerRunner()
    try:
        await runner.run(worker)
    finally:
        logger.info(f"Pipeline finalizado (session={session.id})")
