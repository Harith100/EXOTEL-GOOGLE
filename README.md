# AI Voice Assistant for Vythiri Park

## Overview

An end-to-end conversational voice AI system built to handle inbound phone calls for Vythiri Park, a tourist attraction in Wayanad, Kerala. The system enables callers to speak naturally in Malayalam and receive real-time AI-generated responses through a fully automated voice conversation.

The application integrates telephony streaming, speech-to-text, large language models, and text-to-speech into a single real-time pipeline capable of handling live customer interactions.

## Features

* Real-time voice conversations over phone calls
* Malayalam speech recognition and response generation
* AI-powered question answering using LLMs
* Telephony integration through Exotel WebSocket media streams
* Silence detection and voice activity handling
* Conversation memory and contextual responses
* Automatic fallback mechanisms for STT and TTS failures
* Health monitoring and service diagnostics
* Production-oriented error handling and retry logic

## Architecture

Caller
→ Exotel Voice Stream
→ FastAPI WebSocket Server
→ Speech-to-Text (Sarvam AI)
→ Groq LLM (Llama 3 70B)
→ Google Cloud Text-to-Speech
→ Exotel Audio Stream
→ Caller

## Technology Stack

### Backend

* Python
* FastAPI
* WebSockets
* AsyncIO

### AI Services

* Groq (Llama 3 70B)
* Sarvam AI Speech-to-Text
* Google Cloud Text-to-Speech

### Telephony

* Exotel Voice Streaming API

### Infrastructure

* Docker
* Environment-based configuration
* Health monitoring endpoints

## Key Engineering Challenges

### Real-Time Audio Processing

Incoming phone audio arrives as low-bandwidth 8kHz PCM streams. The system processes audio incrementally, detects silence boundaries, and determines when a caller has completed a spoken query before triggering transcription and response generation.

### Telephony Compatibility

Exotel requires strict adherence to audio formats, chunk sizes, timing intervals, and sequence numbers. Custom PCM validation and chunking logic were implemented to ensure smooth audio playback during live calls.

### Reliability and Fault Tolerance

The application includes:

* LLM rate-limit handling with exponential backoff
* Mock fallback mode for service outages
* Service health checks
* Automatic recovery from transcription failures
* Buffer management to prevent oversized audio payloads

### Multilingual AI Experience

The assistant was designed specifically for Malayalam-speaking users and uses Malayalam prompts, transcription, and speech synthesis to create a natural conversational experience.

## Example Use Cases

* Ticket pricing inquiries
* Park timings
* Ride information
* Visitor assistance
* General customer support

## Example Conversation

User:
"പാർക്ക് എത്ര മണിക്ക് തുറക്കും?"

Assistant:
"വൈത്തിരി പാർക്ക് രാവിലെ 9 മണി മുതൽ വൈകിട്ട് 6 മണി വരെ പ്രവർത്തിക്കുന്നു."

## Learnings

This project provided hands-on experience with real-time voice AI systems, telephony integrations, streaming architectures, speech technologies, and production-grade LLM orchestration. It highlighted the challenges of latency, reliability, audio processing, and conversational user experience in live voice environments.

## Future Improvements

* Interruptible voice responses (barge-in support)
* Streaming speech recognition
* Streaming LLM responses
* Multi-language support
* Analytics and conversation insights dashboard
* Retrieval-Augmented Generation (RAG) for dynamic knowledge updates
