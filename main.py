"""Servidor FastAPI: motor de voz para Veloxiam (Pipecat 1.3.0).

Veloxiam maneja WhatsApp/Meta (webhook, tokens, SDP con Meta). Para cada llamada
le pasa a Pipecat el SDP offer + la config del bot (voiceId, botId, ...) por
POST /call/start. Pipecat:
  1. crea la conexión WebRTC y responde el SDP answer (que Veloxiam reenvía a Meta),
  2. levanta el pipeline STT -> llm-response -> TTS(voiceId),
  3. la media va directa Meta <-> Pipecat por WebRTC.

Endpoints:
  POST /call/start          { callId, botId, voiceId, sdpOffer, ... } -> { sdpAnswer }
  POST /call/terminate      cierra la llamada por callId (+ envía transcript)
  POST /call/agent-response inyecta texto del agente a una llamada (modo async)
  GET  /health              estado + sesiones activas
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket
from loguru import logger

from pipecat.frames.frames import TTSSpeakFrame
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection

from pipeline.bridge import run_call
from pipeline.config import BotConfig
from pipeline.telnyx import run_telnyx_call
from session_manager import SessionManager

load_dotenv()

# Nuestro código: INFO+  |  pipecat internals: WARNING+ (evita el flood de DEBUG)
logger.remove()
logger.add(
    sys.stderr,
    level="DEBUG",
    filter=lambda r: (
        r["level"].no >= 30   # WARNING+ para pipecat.*
        if r["name"].startswith("pipecat.")
        else r["level"].no >= 20  # INFO+ para pipeline.*, session_manager, __main__
    ),
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    colorize=True,
)

REQUIRED_VARS = ["DEEPGRAM_API_KEY", "CARTESIA_API_KEY", "LLM_RESPONSE_URL"]


def _ice_servers() -> list[IceServer]:
    servers = [IceServer(urls=os.getenv("STUN_SERVER", "stun:stun.l.google.com:19302"))]
    turn = os.getenv("TURN_SERVER")
    if turn:
        servers.append(
            IceServer(
                urls=turn,
                username=os.getenv("TURN_USERNAME"),
                credential=os.getenv("TURN_PASSWORD"),
            )
        )
    return servers


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Pipecat server starting...")
    missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
    if missing:
        logger.warning(f"Faltan variables: {missing}.")

    app.state.http = httpx.AsyncClient()
    app.state.sessions = SessionManager(
        max_sessions=int(os.getenv("MAX_SESSIONS", 10)),
        http=app.state.http,
        transcript_url=os.getenv("TRANSCRIPT_URL", ""),
    )
    try:
        yield
    finally:
        logger.info("🛑 Pipecat server shutting down...")
        await app.state.sessions.close_all()
        await app.state.http.aclose()


app = FastAPI(lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Llamadas
# --------------------------------------------------------------------------- #
@app.post("/call/start")
async def start_call(data: dict):
    """Inicia una llamada: recibe el SDP offer + config del bot y devuelve el
    SDP answer. Arranca el pipeline de voz en segundo plano.

    Body: { callId, botId, voiceId, sdpOffer, type?, phoneNumber?, language?,
            tenantId?, greeting?, cartesiaModel?, deepgramModel? }
    """
    sessions: SessionManager = app.state.sessions
    if sessions.is_full():
        raise HTTPException(status_code=503, detail="Máximo de sesiones alcanzado")

    for field in ("callId", "botId", "voiceId", "sdpOffer"):
        if not data.get(field):
            raise HTTPException(status_code=400, detail=f"{field} requerido")

    cfg = BotConfig.from_request(data)

    # Negociación WebRTC: oferta de Meta (vía Veloxiam) -> respuesta de Pipecat.
    connection = SmallWebRTCConnection(ice_servers=_ice_servers())
    await connection.initialize(sdp=data["sdpOffer"], type=data.get("type", "offer"))
    answer = connection.get_answer()
    if not answer:
        raise HTTPException(status_code=500, detail="No se pudo generar el SDP answer")

    session = sessions.create(
        call_id=data["callId"],
        bot_id=cfg.bot_id,
        tenant_id=cfg.tenant_id,
        phone_number=data.get("phoneNumber", "unknown"),
    )
    asyncio.create_task(_run_and_cleanup(connection, session, cfg))

    return {
        "sessionId": session.id,
        "sdpAnswer": answer["sdp"],
        "type": answer["type"],
        "pcId": answer["pc_id"],
    }


async def _run_and_cleanup(connection: SmallWebRTCConnection, session, cfg: BotConfig):
    """Ejecuta el pipeline y, al terminar, asegura el flush del transcript."""
    logger.info(
        f"[pipeline] Iniciando — session={session.id} call_id={session.call_id} "
        f"bot={cfg.bot_id} voice={cfg.voice_id} lang={cfg.language}"
    )
    try:
        await run_call(connection, session, app.state.http, cfg)
        logger.info(f"[pipeline] Terminó normalmente — session={session.id}")
    except Exception as e:  # noqa: BLE001
        logger.exception(f"[pipeline] Error inesperado — session={session.id}: {e}")
    finally:
        logger.info(f"[pipeline] Limpiando sesión — session={session.id}")
        await app.state.sessions.terminate(session.call_id)


@app.post("/call/terminate")
async def terminate_call(data: dict):
    """Cierra la llamada por callId y envía el transcript."""
    call_id = data.get("callId")
    if not call_id:
        raise HTTPException(status_code=400, detail="callId requerido")
    await app.state.sessions.terminate(call_id)
    return {"status": "terminated"}


@app.post("/call/agent-response")
async def agent_response(data: dict):
    """Inyecta texto del agente en una llamada activa (modo asíncrono/proactivo).

    El flujo normal usuario->agente es síncrono dentro del pipeline; esto es para
    que Veloxiam empuje un mensaje no solicitado hacia una llamada en curso.
    """
    call_id = data.get("callId")
    text = data.get("text")
    if not call_id or not text:
        raise HTTPException(status_code=400, detail="callId y text requeridos")
    session = app.state.sessions.get(call_id)
    if session is None or session.worker is None:
        raise HTTPException(status_code=404, detail="Sesión no activa")
    session.add_turn("assistant", text, None)
    await session.worker.queue_frame(TTSSpeakFrame(text))
    return {"status": "queued"}


# --------------------------------------------------------------------------- #
# Telnyx Media Streaming (telefonía PSTN)
# --------------------------------------------------------------------------- #
@app.websocket("/telnyx")
async def telnyx_stream(websocket: WebSocket):
    """WebSocket de Telnyx Media Streaming.

    Veloxiam (NestJS) ya contestó la llamada y resolvió el bot por el número
    llamado; arrancó el streaming hacia aquí con el contexto en la query:
    ``?botId=...&voiceId=...&telnyxApiKey=...&from=...&to=...&callId=<call_control_id>``.

    Reutiliza el mismo pipeline que WhatsApp, con transporte WebSocket + Telnyx
    serializer (PCMU 8000).
    """
    await websocket.accept()
    q = websocket.query_params
    bot_id = q.get("botId")
    voice_id = q.get("voiceId")
    telnyx_api_key = q.get("telnyxApiKey")  # API key propia del bot
    call_id = q.get("callId") or ""  # = call_control_id
    is_ia_content = q.get("isIaContent") == "1"  # outbound: greeting generado por IA
    chat_id = q.get("chatId") or ""

    if not bot_id or not voice_id:
        logger.error(f"[telnyx] WS sin botId/voiceId (q={dict(q)}) — cerrando")
        await websocket.close(code=1008)
        return

    sessions: SessionManager = app.state.sessions
    if sessions.is_full():
        logger.warning("[telnyx] Máximo de sesiones alcanzado — cerrando WS")
        await websocket.close(code=1013)
        return

    cfg = BotConfig(
        bot_id=bot_id,
        voice_id=voice_id,
        greeting=q.get("greeting", "Hola, gracias por llamar. ¿En qué puedo ayudarte?"),
        is_ia_content=is_ia_content,
        chat_id=chat_id,
    )
    session = sessions.create(
        call_id=call_id or bot_id,
        bot_id=bot_id,
        tenant_id=bot_id,
        phone_number=q.get("from", "unknown"),
    )
    # Outbound: usar el chatId provisto como sessionId para continuar el historial.
    if chat_id:
        session.id = chat_id

    logger.info(f"[telnyx] WS aceptado bot={bot_id} voiceId={voice_id} from={q.get('from')} call_id={call_id}")
    try:
        await run_telnyx_call(
            websocket, session, app.state.http, cfg, call_id, telnyx_api_key
        )
    except Exception as e:  # noqa: BLE001
        logger.exception(f"[telnyx] Error en la llamada session={session.id}: {e}")
    finally:
        await app.state.sessions.terminate(session.call_id)


@app.get("/health")
async def health():
    sessions: SessionManager = app.state.sessions
    return {
        "status": "ok",
        "active_sessions": sessions.count(),
        "max_sessions": sessions.max_sessions,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
