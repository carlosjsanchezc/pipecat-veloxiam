"""Servidor FastAPI para llamadas de voz WhatsApp con Pipecat 1.3.0.

WhatsApp Business Calling es **dirigido por webhook**: Meta verifica y notifica
en ``/whatsapp``. Pipecat (``WhatsAppClient``) resuelve la verificación y el SDP
offer/answer; cuando una llamada se conecta nos entrega una conexión WebRTC y
nosotros levantamos el pipeline STT -> NestJS -> TTS (ver pipeline/bridge.py).

Endpoints:
  GET  /whatsapp            verificación del webhook (Meta)
  POST /whatsapp            eventos de llamada (connect/terminate)
  GET  /health              sesiones activas
  POST /call/start          fija el bot/tenant por defecto para entrantes
  POST /call/terminate      fuerza el cierre de una llamada (+ envía transcript)
  POST /call/agent-response inyecta texto del agente a una llamada (modo async)
"""

import asyncio
import os
from contextlib import asynccontextmanager

import aiohttp
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from loguru import logger

from pipecat.frames.frames import TTSSpeakFrame
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.whatsapp.api import WhatsAppWebhookRequest
from pipecat.transports.whatsapp.client import WhatsAppClient

from pipeline.bridge import run_call
from session_manager import SessionManager

load_dotenv()

REQUIRED_WHATSAPP_VARS = [
    "WHATSAPP_TOKEN",
    "WHATSAPP_PHONE_NUMBER_ID",
    "WHATSAPP_APP_SECRET",
    "WHATSAPP_WEBHOOK_VERIFICATION_TOKEN",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Pipecat server starting...")

    missing = [v for v in REQUIRED_WHATSAPP_VARS if not os.getenv(v)]
    if missing:
        logger.warning(f"Faltan variables de WhatsApp: {missing}. El webhook no funcionará.")

    # Cliente HTTP para NestJS y sesión aiohttp para el WhatsAppClient.
    app.state.http = httpx.AsyncClient()
    app.state.aiohttp = aiohttp.ClientSession()

    app.state.sessions = SessionManager(
        max_sessions=int(os.getenv("MAX_SESSIONS", 10)),
        nestjs_url=os.environ["NESTJS_AGENT_URL"],
        http=app.state.http,
    )

    app.state.default_bot_id = os.getenv("DEFAULT_BOT_ID") or "default"
    app.state.default_tenant_id = os.getenv("DEFAULT_TENANT_ID") or "default"

    app.state.whatsapp = None
    if not missing:
        app.state.whatsapp = WhatsAppClient(
            whatsapp_token=os.environ["WHATSAPP_TOKEN"],
            phone_number_id=os.environ["WHATSAPP_PHONE_NUMBER_ID"],
            whatsapp_secret=os.environ["WHATSAPP_APP_SECRET"],
            session=app.state.aiohttp,
        )
        logger.info("WhatsApp client inicializado.")

    try:
        yield
    finally:
        logger.info("🛑 Pipecat server shutting down...")
        await app.state.sessions.close_all()
        if app.state.whatsapp is not None:
            await app.state.whatsapp.terminate_all_calls()
        await app.state.aiohttp.close()
        await app.state.http.aclose()


app = FastAPI(lifespan=lifespan)


# --------------------------------------------------------------------------- #
# WhatsApp webhook
# --------------------------------------------------------------------------- #
@app.get("/whatsapp")
async def verify_webhook(request: Request):
    """Verificación del webhook por parte de Meta (hub.challenge)."""
    client: WhatsAppClient = app.state.whatsapp
    if client is None:
        raise HTTPException(status_code=503, detail="WhatsApp client no inicializado")
    return await client.handle_verify_webhook_request(
        params=dict(request.query_params),
        expected_verification_token=os.environ["WHATSAPP_WEBHOOK_VERIFICATION_TOKEN"],
    )


@app.post("/whatsapp")
async def whatsapp_webhook(
    body: WhatsAppWebhookRequest,
    request: Request,
    x_hub_signature_256: str = Header(None),
):
    """Eventos de llamada de WhatsApp. En `connect` levantamos el pipeline."""
    client: WhatsAppClient = app.state.whatsapp
    sessions: SessionManager = app.state.sessions
    if client is None:
        raise HTTPException(status_code=503, detail="WhatsApp client no inicializado")
    if body.object != "whatsapp_business_account":
        raise HTTPException(status_code=400, detail="object inválido")

    async def connection_callback(connection: SmallWebRTCConnection):
        if sessions.is_full():
            logger.warning("Máximo de sesiones alcanzado; rechazando llamada.")
            await connection.disconnect()
            return
        # El callback no expone el caller ni el call_id de Meta, así que usamos
        # el id de la conexión WebRTC como clave y el bot por defecto.
        call_id = connection.pc_id
        session = sessions.create(
            call_id=call_id,
            bot_id=app.state.default_bot_id,
            tenant_id=app.state.default_tenant_id,
            phone_number="whatsapp",
        )
        asyncio.create_task(_run_and_cleanup(connection, session))

    raw_body = await request.body()
    try:
        await client.handle_webhook_request(
            body,
            connection_callback,
            sha256_signature=x_hub_signature_256,
            raw_body=raw_body,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        logger.error(f"Error procesando webhook: {e}")
        raise HTTPException(status_code=500, detail="Error interno")
    return {"status": "ok"}


async def _run_and_cleanup(connection: SmallWebRTCConnection, session):
    """Ejecuta el pipeline y, al terminar, asegura el flush del transcript."""
    try:
        await run_call(connection, session, app.state.http)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Pipeline error (session={session.id}): {e}")
    finally:
        await app.state.sessions.terminate(session.call_id)


# --------------------------------------------------------------------------- #
# Endpoints de control
# --------------------------------------------------------------------------- #
@app.get("/health")
async def health():
    sessions: SessionManager = app.state.sessions
    return {
        "status": "ok",
        "active_sessions": sessions.count(),
        "max_sessions": sessions.max_sessions,
        "whatsapp_ready": app.state.whatsapp is not None,
    }


@app.post("/call/start")
async def start_call(data: dict):
    """Fija el bot/tenant por defecto que usarán las llamadas entrantes.

    WhatsApp Calling es entrante (Meta inicia la llamada vía webhook); este
    endpoint NO origina la llamada, solo registra a qué bot/tenant enrutarla.
    """
    if "botId" in data:
        app.state.default_bot_id = data["botId"]
    if "tenantId" in data:
        app.state.default_tenant_id = data["tenantId"]
    return {
        "status": "ok",
        "defaultBotId": app.state.default_bot_id,
        "defaultTenantId": app.state.default_tenant_id,
    }


@app.post("/call/terminate")
async def terminate_call(data: dict):
    """Fuerza el cierre de una llamada por callId y envía el transcript."""
    call_id = data.get("callId")
    if not call_id:
        raise HTTPException(status_code=400, detail="callId requerido")
    await app.state.sessions.terminate(call_id)
    return {"status": "terminated"}


@app.post("/call/agent-response")
async def agent_response(data: dict):
    """Inyecta texto del agente en una llamada activa (modo asíncrono/proactivo).

    El flujo normal usuario->agente es síncrono dentro del pipeline; este
    endpoint es para que NestJS empuje un mensaje no solicitado (p.ej. una
    notificación) hacia una llamada en curso.
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
