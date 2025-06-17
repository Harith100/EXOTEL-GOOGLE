import asyncio
import websockets
import base64
import json
import wave
import io
import logging
import signal
import sys

# Sarvam STT config
SARVAM_STT_WS = "wss://api.sarvam.ai/speech-to-text/ws?language-code=ml-IN&model=saarika:v2.5"
SAMPLE_RATE = 8000  # Hz

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SarvamSTTClient")

# Graceful shutdown
shutdown_event = asyncio.Event()


def pcm_to_base64_wav(pcm_data: bytes, sample_rate: int = 8000) -> str:
    """Convert PCM bytes to base64-encoded WAV format"""
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit PCM
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)
        wav_bytes = wav_io.getvalue()
    return base64.b64encode(wav_bytes).decode('utf-8')


async def send_audio(ws, audio_bytes):
    """Send audio chunk as base64 WAV"""
    b64_data = pcm_to_base64_wav(audio_bytes, SAMPLE_RATE)
    payload = {
        "audio": {
            "data": b64_data,
            "encoding": "audio/wav",
            "sample_rate": str(SAMPLE_RATE)
        }
    }
    await ws.send(json.dumps(payload))
    logger.info(f"Sent {len(audio_bytes)} bytes of audio.")


async def receive_transcription(ws):
    """Receive transcription results"""
    async for message in ws:
        try:
            response = json.loads(message)
            if "transcript" in response:
                logger.info("📝 Transcript: %s", response["transcript"])
            else:
                logger.debug("Received message: %s", response)
        except Exception as e:
            logger.error("Error decoding message: %s", e)


async def stt_main():
    async with websockets.connect(SARVAM_STT_WS) as ws:
        logger.info("Connected to Sarvam STT WebSocket.")

        # Simulated audio source (replace with real Exotel PCM buffer)
        async def audio_streamer():
            with open("sample_8khz_pcm.raw", "rb") as f:  # Raw PCM file
                while not shutdown_event.is_set():
                    chunk = f.read(1600)  # 100ms @ 8kHz 16-bit mono = 1600 bytes
                    if not chunk:
                        break
                    await send_audio(ws, chunk)
                    await asyncio.sleep(0.1)

        await asyncio.gather(audio_streamer(), receive_transcription(ws))


def main():
    loop = asyncio.get_event_loop()

    def shutdown_handler(*_):
        logger.info("Shutdown signal received.")
        shutdown_event.set()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        loop.run_until_complete(stt_main())
    finally:
        logger.info("Client shutdown complete.")
        loop.close()


if __name__ == "__main__":
    main()
