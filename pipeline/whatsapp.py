"""Pipeline de voz WhatsApp / Meta vía WebRTC (SmallWebRTCTransport).

Veloxiam negocia la llamada con Meta y pasa el SDP offer a ``POST /call/start``;
``main.py`` crea la ``SmallWebRTCConnection`` y arranca ``run_whatsapp_call``.

Audio a **16 kHz** en transporte, STT, TTS y Cartesia (config estable WebRTC).
"""

import asyncio
import os

import httpx
from loguru import logger

from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.workers.runner import WorkerRunner

from pipeline.agent import NestJSAgentProcessor
from pipeline.config import BotConfig
from pipeline.greeting import GreetingBargeInGate, GreetingCompleteTap, begin_greeting
from pipeline.release import CallReleaseTap
from pipeline.stt import (
    AudioInputTap,
    AudioOutputTap,
    PipelineErrorTap,
    TranscriptionTap,
    UserSpeechTap,
    create_stt,
)
from pipeline.tts import create_tts
from pipeline.vad import create_user_aggregator_params, create_whatsapp_vad_analyzer

# WebRTC WhatsApp: 16 kHz en todo el pipeline (como la config original que funcionaba).
WHATSAPP_SAMPLE_RATE = 16000


async def run_whatsapp_call(
    connection: SmallWebRTCConnection,
    session,
    http: httpx.AsyncClient,
    cfg: BotConfig,
):
    """Construye y ejecuta el pipeline WhatsApp/WebRTC para una llamada conectada."""
    llm_url = os.environ["LLM_RESPONSE_URL"]
    logger.info(
        f"[whatsapp] Construyendo pipeline — session={session.id} "
        f"stt_model={cfg.deepgram_model} tts_model={cfg.cartesia_model} "
        f"voice={cfg.voice_id} lang={cfg.language} sr={WHATSAPP_SAMPLE_RATE}"
    )

    # STT/TTS sin sample_rate explícito: usan 16 kHz por defecto (create_stt/create_tts).
    stt = create_stt(cfg)
    tts = create_tts(cfg)
    vad = create_whatsapp_vad_analyzer()
    logger.info(f"[whatsapp] STT/TTS/VAD/SmartTurn creados — session={session.id}")

    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=WHATSAPP_SAMPLE_RATE,
            audio_out_sample_rate=WHATSAPP_SAMPLE_RATE,
        ),
    )

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
            UserSpeechTap(session),
            GreetingBargeInGate(session),
            agent,
            tts,
            AudioOutputTap(session.id),
            transport.output(),
            GreetingCompleteTap(session),
            CallReleaseTap(session),
            assistant_aggregator,
        ]
    )
    logger.info(f"[whatsapp] Pipeline construido — session={session.id}")

    # Sin audio_in/out_sample_rate en PipelineWorker (igual que config original WebRTC).
    worker = PipelineWorker(pipeline, params=PipelineParams(enable_metrics=False))
    session.worker = worker

    greeting = (cfg.greeting or "").strip()
    if greeting:
        logger.info(f"[whatsapp] Saludo configurado: {greeting!r} — session={session.id}")
    else:
        logger.info(f"[whatsapp] Sin saludo inicial — session={session.id}")

    @transport.event_handler("on_client_connected")
    async def _on_connected(_transport, _client):
        pc = connection.pc
        logger.info(
            f"[whatsapp] Cliente conectado — session={session.id} "
            f"ice={pc.iceConnectionState} conn={pc.connectionState}"
        )
        if not greeting:
            logger.info(
                f"[whatsapp] Esperando que el usuario hable — session={session.id}"
            )
            return

        async def _send_greeting():
            delay = float(os.getenv("GREETING_DELAY_SEC", "1.0"))
            await asyncio.sleep(delay)
            if getattr(session, "user_spoke", False):
                logger.info(
                    f"[whatsapp] Usuario habló primero — omitiendo saludo "
                    f"session={session.id}"
                )
                return
            logger.info(f"[whatsapp] Encolando saludo TTS: {greeting!r}")
            session.add_turn("assistant", greeting, None)
            begin_greeting(session)
            await worker.queue_frame(TTSSpeakFrame(greeting))

        asyncio.create_task(_send_greeting())

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(_transport, _client):
        logger.info(f"[whatsapp] Cliente desconectado — session={session.id}")
        await worker.cancel(reason="client disconnected")

    logger.info(f"[whatsapp] Lanzando WorkerRunner — session={session.id}")
    runner = WorkerRunner()
    try:
        await runner.run(worker)
    finally:
        logger.info(f"[whatsapp] WorkerRunner finalizado — session={session.id}")
