import asyncio
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, HTTPException
from session_manager import SessionManager

load_dotenv()

session_manager = SessionManager(max_sessions=int(os.getenv("MAX_SESSIONS", 10)))

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Pipecat server starting...")
    yield
    print("🛑 Pipecat server shutting down...")
    await session_manager.close_all()

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_sessions": session_manager.count(),
        "max_sessions": session_manager.max_sessions
    }

@app.post("/call/start")
async def start_call(data: dict):
    if session_manager.is_full():
        raise HTTPException(status_code=503, detail="Max sessions reached")
    
    session_id = await session_manager.create(
        call_id=data["callId"],
        bot_id=data["botId"],
        tenant_id=data["tenantId"],
        phone_number=data["phoneNumber"]
    )
    return {"sessionId": session_id}

@app.post("/call/terminate")
async def terminate_call(data: dict):
    await session_manager.terminate(data["callId"])
    return {"status": "terminated"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
