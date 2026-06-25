"""Pipeline de voz: STT (Deepgram) -> agente Veloxiam -> TTS (Cartesia).

- WhatsApp/Meta: ``pipeline.whatsapp.run_whatsapp_call`` (WebRTC 16 kHz)
- Telnyx PSTN: ``pipeline.telnyx.run_telnyx_call`` (WebSocket PCMU 8 kHz)
"""
