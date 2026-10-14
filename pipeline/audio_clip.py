"""Reproducción de archivos de audio del bot (``audiofile`` de llm-response).

Veloxiam manda una URL firmada (mp3, wav, ogg, m4a…). Se descarga, se decodifica
con PyAV (dependencia de aiortc, no requiere ffmpeg del sistema) a PCM s16le mono
al sample rate del pipeline y se empuja como ``TTSAudioRawFrame`` detrás de la
respuesta de Cartesia: el TTS serializa los frames no-sistema después de su
audio, así que el clip suena justo al terminar el texto.

Al final del clip se empuja ``AudioClipEndFrame`` (un ``TTSStoppedFrame``) para
que el transporte emita ``BotStoppedSpeakingFrame`` y ``CallReleaseTap`` sepa
que el clip terminó.
"""

import asyncio
import io
import os
import time
from dataclasses import dataclass

import av
import httpx
from loguru import logger

from pipecat.frames.frames import TTSAudioRawFrame, TTSStoppedFrame

_MAX_BYTES = 25 * 1024 * 1024
_CHUNK_SECS = 0.1


@dataclass
class AudioClipEndFrame(TTSStoppedFrame):
    """Marca el fin de un clip de audio del bot."""


@dataclass
class AudioClip:
    frames: list[TTSAudioRawFrame]
    duration_secs: float


def _decode_to_pcm(data: bytes, sample_rate: int, max_secs: float) -> bytes:
    max_bytes = int(sample_rate * max_secs) * 2
    resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
    out = bytearray()
    with av.open(io.BytesIO(data)) as container:
        for frame in container.decode(audio=0):
            for f in resampler.resample(frame):
                out += f.to_ndarray().tobytes()
            if len(out) >= max_bytes:
                return bytes(out[:max_bytes])
    for f in resampler.resample(None):
        out += f.to_ndarray().tobytes()
    return bytes(out[:max_bytes])


async def _download(http: httpx.AsyncClient, url: str) -> bytes:
    buf = bytearray()
    async with http.stream("GET", url, timeout=15.0, follow_redirects=True) as resp:
        resp.raise_for_status()
        async for chunk in resp.aiter_bytes():
            buf += chunk
            if len(buf) > _MAX_BYTES:
                raise ValueError(f"audio supera {_MAX_BYTES // (1024 * 1024)} MB")
    return bytes(buf)


async def load_audio_clip(
    http: httpx.AsyncClient, url: str, sample_rate: int
) -> AudioClip:
    """Descarga y decodifica el audio a frames listos para el transporte."""
    max_secs = float(os.getenv("AUDIO_CLIP_MAX_SEC", "300"))
    t0 = time.perf_counter()
    data = await _download(http, url)
    t1 = time.perf_counter()
    pcm = await asyncio.to_thread(_decode_to_pcm, data, sample_rate, max_secs)
    t2 = time.perf_counter()

    chunk = int(sample_rate * _CHUNK_SECS) * 2
    frames = [
        TTSAudioRawFrame(audio=pcm[i : i + chunk], sample_rate=sample_rate, num_channels=1)
        for i in range(0, len(pcm), chunk)
    ]
    duration = len(pcm) / (sample_rate * 2)
    logger.info(
        f"[audio-clip] {len(data)} bytes → {duration:.1f}s PCM {sample_rate} Hz "
        f"(descarga {(t1 - t0) * 1000:.0f}ms, decodificación {(t2 - t1) * 1000:.0f}ms)"
    )
    return AudioClip(frames=frames, duration_secs=duration)
