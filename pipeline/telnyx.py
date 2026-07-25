"""Pipeline de voz Telnyx PSTN vía WebSocket (FastAPIWebsocketTransport).

Telnyx se conecta directo a Pipecat; Veloxiam arranca el streaming hacia
``WS /telnyx`` con ``botId``, ``voiceId``, ``callId``, etc. en la query.

Orden del pipeline::

    transport.input()      audio entrante (PCMU 8 kHz) de Telnyx
      -> stt               Deepgram -> TranscriptionFrame
      -> TranscriptionTap  registra texto + confidence
      -> user_aggregator   VAD Silero (barge-in) + cierre de turno
      -> NestJSAgentProcessor   POST llm-response -> LLMTextFrame
      -> tts               Cartesia -> audio
      -> transport.output()     audio saliente a Telnyx
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
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.workers.runner import WorkerRunner

from pipeline.agent import NestJSAgentProcessor
from pipeline.config import BotConfig
from pipeline.greeting import GreetingBargeInGate, GreetingCompleteTap, begin_greeting
from pipeline.release import CallReleaseTap
from pipeline.stt import AudioInputTap, PipelineErrorTap, TranscriptionTap, create_stt
from pipeline.tts import create_tts
from pipeline.vad import create_telnyx_vad_analyzer, create_user_aggregator_params

TELNYX_SAMPLE_RATE = 8000


async def run_telnyx_call(
    websocket: WebSocket,
    session,
    http,
    cfg: BotConfig,
    call_control_id: str,
    telnyx_api_key: str | None = None,
):
    """Construye y ejecuta el pipeline Telnyx sobre una conexión WebSocket.

    Bloquea hasta stop de Telnyx, cuelgue o cancelación.
    """
    llm_url = os.environ["LLM_RESPONSE_URL"]

    start = await _read_telnyx_start(websocket)
    if not start:
        logger.error(f"[telnyx] No se recibió evento start — abortando session={session.id}")
        return

    logger.info(
        f"[telnyx] Construyendo pipeline — session={session.id} "
        f"stream_id={start['stream_id']} encoding={start['outbound_encoding']} "
        f"call_control_id={call_control_id} voice={cfg.voice_id} lang={cfg.language} "
        f"llm_url={llm_url}"
    )

    serializer = TelnyxFrameSerializer(
        stream_id=start["stream_id"],
        outbound_encoding=start["outbound_encoding"],
        inbound_encoding=start["inbound_encoding"],
        call_control_id=call_control_id or None,
        api_key=telnyx_api_key or os.getenv("TELNYX_API_KEY") or None,
        params=TelnyxFrameSerializer.InputParams(
            telnyx_sample_rate=TELNYX_SAMPLE_RATE,
            sample_rate=TELNYX_SAMPLE_RATE,
            # Veloxiam controla transfer/hangup. Si Pipecat cuelga al cancelar
            # el worker tras un transfer, corta la llamada transferida.
            auto_hang_up=False,
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
    vad = create_telnyx_vad_analyzer()
    logger.info(f"[telnyx] STT/TTS/VAD/SmartTurn creados — session={session.id}")

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=create_user_aggregator_params(vad),
    )

    agent = NestJSAgentProcessor(session=session, http=http, llm_url=llm_url)

    pipeline = Pipeline(
        [
            transport.input(),
            AudioInputTap(session.id),
            stt,
            TranscriptionTap(session),
            PipelineErrorTap(),
            user_aggregator,
            GreetingBargeInGate(session),
            agent,
            tts,
            transport.output(),
            GreetingCompleteTap(session),
            CallReleaseTap(session),
            assistant_aggregator,
        ]
    )
    logger.info(f"[telnyx] Pipeline construido — session={session.id}")

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=False,
            audio_in_sample_rate=TELNYX_SAMPLE_RATE,
            audio_out_sample_rate=TELNYX_SAMPLE_RATE,
        ),
    )
    session.worker = worker

    greeting = (cfg.greeting or "").strip()
    if greeting:
        logger.info(f"[telnyx] Saludo configurado: {greeting!r} — session={session.id}")

    @transport.event_handler("on_client_connected")
    async def _on_connected(_transport, _client):
        logger.info(f"[telnyx] Cliente conectado — session={session.id}")
        if not greeting:
            return

        async def _send_greeting():
            delay = float(os.getenv("TELNYX_GREETING_DELAY_SEC", "0.5"))
            await asyncio.sleep(delay)
            if cfg.is_ia_content:
                logger.info(f"[telnyx] Saludo IA desde instrucción: {greeting!r}")
                begin_greeting(session)
                await agent._respond(greeting)
            else:
                logger.info(f"[telnyx] Encolando saludo TTS: {greeting!r}")
                session.add_turn("assistant", greeting, None)
                begin_greeting(session)
                await worker.queue_frame(TTSSpeakFrame(greeting))

        asyncio.create_task(_send_greeting())

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(_transport, _client):
        logger.info(f"[telnyx] Cliente desconectado — session={session.id}")
        await worker.cancel(reason="client disconnected")

    logger.info(f"[telnyx] Lanzando WorkerRunner — session={session.id}")
    runner = WorkerRunner()
    try:
        await runner.run(worker)
    finally:
        logger.info(f"[telnyx] WorkerRunner finalizado — session={session.id}")


async def _read_telnyx_start(websocket: WebSocket) -> dict | None:
    """Lee eventos iniciales del WS de Telnyx hasta obtener ``start``."""
    default_encoding = "PCMU"
    for _ in range(5):
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
