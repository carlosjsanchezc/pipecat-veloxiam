"""Bridge Telnyx Media Streaming (WebSocket) <-> pipeline STT -> agente -> TTS.

Telefonía PSTN: a diferencia de WhatsApp (WebRTC vía Meta), aquí Telnyx se
conecta DIRECTO a Pipecat por WebSocket. Veloxiam (NestJS) ya contestó la
llamada y arrancó el streaming apuntando a ``{PIPECAT_URL}/telnyx``, pasando el
contexto en la query: ``botId``, ``voiceId``, ``from``, ``to``, ``callId``
(= call_control_id). Por eso aquí NO se toca la BD: el bot ya viene resuelto.

El pipeline es el MISMO que el de WhatsApp (``pipeline.bridge``): solo cambia el
transporte (FastAPIWebsocket + TelnyxFrameSerializer) y el sample rate (8000
PCMU, telefonía). El "cerebro" sigue siendo ``POST /agent-voice/llm-response``.

Orden del pipeline::

    transport.input()      audio entrante (PCMU 8000) de Telnyx
      -> stt               Deepgram -> TranscriptionFrame
      -> TranscriptionTap  registra texto + confidence
      -> user_aggregator   VAD (barge-in) + cierre de turno -> LLMContextFrame
      -> NestJSAgentProcessor   POST a llm-response -> LLMTextFrame
      -> tts               Cartesia (voiceId del bot) -> audio
      -> transport.output()     audio saliente a Telnyx
      -> assistant_aggregator   registra la respuesta en el contexto
"""

import asyncio
import json
import os

from fastapi import WebSocket
from loguru import logger

from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.workers.runner import WorkerRunner

from pipeline.bridge import NestJSAgentProcessor
from pipeline.config import BotConfig
from pipeline.stt import TranscriptionTap, create_stt
from pipeline.tts import create_tts
from pipeline.vad import create_vad_analyzer

# Telefonía: PCMU (g711 µ-law) a 8 kHz. Coincide con el streaming_start de NestJS
# (stream_bidirectional_codec=PCMU) y con el default del TelnyxFrameSerializer.
TELNYX_SAMPLE_RATE = 8000


async def run_telnyx_call(
    websocket: WebSocket,
    session,
    http,
    cfg: BotConfig,
    call_control_id: str,
    telnyx_api_key: str | None = None,
):
    """Construye y ejecuta el pipeline sobre una conexión WebSocket de Telnyx.

    Bloquea hasta que la llamada termina (stop de Telnyx, cuelgue o cancelación).
    """
    llm_url = os.environ["LLM_RESPONSE_URL"]

    # Telnyx abre el WS y manda primero "connected" y luego "start" (stream_id +
    # media_format.encoding). Los leemos antes de montar el transporte; este
    # seguirá leyendo desde los frames "media".
    start = await _read_telnyx_start(websocket)
    if not start:
        logger.error(f"[telnyx] No se recibió evento start — abortando session={session.id}")
        return

    logger.info(
        f"[run_telnyx_call] session={session.id} stream_id={start['stream_id']} "
        f"encoding={start['outbound_encoding']} call_control_id={call_control_id} "
        f"voice={cfg.voice_id} lang={cfg.language}"
    )

    serializer = TelnyxFrameSerializer(
        stream_id=start["stream_id"],
        outbound_encoding=start["outbound_encoding"],
        inbound_encoding=start["inbound_encoding"],
        # call_control_id + api_key (la del bot, vía query) permiten que el
        # serializer cuelgue la llamada en Telnyx automáticamente al terminar el
        # pipeline. Multi-bot: cada bot tiene su propia key. Si no llega, Telnyx
        # cuelga igual cuando el llamante corta.
        call_control_id=call_control_id or None,
        api_key=telnyx_api_key or os.getenv("TELNYX_API_KEY") or None,
        params=TelnyxFrameSerializer.InputParams(
            telnyx_sample_rate=TELNYX_SAMPLE_RATE,
            sample_rate=TELNYX_SAMPLE_RATE,
        ),
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    stt = create_stt(cfg, sample_rate=TELNYX_SAMPLE_RATE)
    tts = create_tts(cfg, sample_rate=TELNYX_SAMPLE_RATE)
    vad = create_vad_analyzer()

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

    worker = PipelineWorker(pipeline, params=PipelineParams(enable_metrics=False))
    session.worker = worker

    greeting = (cfg.greeting or "").strip()

    @transport.event_handler("on_client_connected")
    async def _on_connected(_transport, _client):
        logger.info(f"[telnyx] Cliente conectado — session={session.id}")
        if greeting:
            async def _send_greeting():
                await asyncio.sleep(0.5)
                logger.info(f"[telnyx] Encolando saludo TTS: {greeting!r}")
                session.add_turn("assistant", greeting, None)
                await worker.queue_frame(TTSSpeakFrame(greeting))
            asyncio.create_task(_send_greeting())

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(_transport, _client):
        logger.info(f"[telnyx] Cliente desconectado — session={session.id}")
        await worker.cancel(reason="client disconnected")

    logger.info(f"[run_telnyx_call] Lanzando WorkerRunner — session={session.id}")
    runner = WorkerRunner()
    try:
        await runner.run(worker)
    finally:
        logger.info(f"[run_telnyx_call] WorkerRunner finalizado — session={session.id}")


async def _read_telnyx_start(websocket: WebSocket) -> dict | None:
    """Lee los primeros eventos del WS de Telnyx hasta obtener el evento ``start``.

    Telnyx envía ``{"event":"connected"}`` y luego ``{"event":"start", ...}`` con
    ``stream_id`` y ``start.media_format.encoding`` (p. ej. PCMU).
    """
    default_encoding = "PCMU"
    for _ in range(5):  # tolera algún evento extra antes del "start"
        try:
            raw = await websocket.receive_text()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[telnyx] Error leyendo WS inicial: {e}")
            return None
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if msg.get("event") != "start":
            continue
        start = msg.get("start", {})
        stream_id = msg.get("stream_id") or start.get("stream_id")
        if not stream_id:
            return None
        encoding = start.get("media_format", {}).get("encoding", default_encoding)
        return {
            "stream_id": stream_id,
            "outbound_encoding": encoding,
            "inbound_encoding": encoding,
        }
    return None
