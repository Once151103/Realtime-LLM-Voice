# Realtime-LLM-Voice

A low-latency real-time voice agent that gives any LLM the ability to speak with streaming audio. Features sub-600ms time-to-first-audio, interruption handling (barge-in), and phoneme-level chunking for natural conversation flow.

## Features

- **Streaming Pipeline**: STT → LLM → TTS with minimal latency
- **Sub-600ms Response Time**: Optimized for real-time conversation
- **Voice Interruption (Barge-in)**: User can interrupt the assistant at any time
- **Predictive Speech Buffering**: Starts generating audio before full sentence completion
- **Desktop GUI**: Real-time visualization with PyQt6
- **Docker Support**: Easy deployment with containerization
- **Modular Architecture**: Replaceable components (STT, LLM, TTS)

## Architecture

```
User Microphone
 ↓
Voice Activity Detection (VAD)
 ↓
Streaming Speech To Text (STT)
 ↓
Voice Orchestrator Service
 ↓
LLM Streaming Engine
 ↓
Sentence / Phoneme Chunker
 ↓
TTS Streaming Worker
 ↓
Audio Queue / Mixer
 ↓
Speaker Output
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| STT | Whisper (streaming) |
| LLM | OpenAI-compatible API |
| TTS | Coqui XTTS v2 |
| VAD | Silero VAD |
| Transport | WebSocket |
| GUI | PyQt6 |
| Audio | PyAudio, SoundDevice |

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- Microphone and speakers

### Installation

```bash
# Clone the repository
git clone https://github.com/Once151103/Realtime-LLM-Voice.git
cd Realtime-LLM-Voice

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys and settings
```

### Running

```bash
# Run the desktop application
python main.py

# Or with Docker
docker-compose up
```

## Configuration

Create a `.env` file with the following:

```env
# LLM Configuration
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini

# Audio Configuration
SAMPLE_RATE=22050
CHUNK_SIZE=1024

# TTS Configuration
TTS_MODEL_PATH=./models/tts
SPEAKER_WAV_PATH=./samples/reference.wav
```

## Project Structure

```
realtime-voice-ai-agent/
├── core/              # Orchestrator and state management
├── services/          # STT, LLM, TTS, VAD services
├── transport/         # WebSocket and audio codec
├── desktop/           # GUI application
├── cache/             # Response caching
├── tests/             # Unit tests
├── main.py            # Entry point
├── config.py          # Configuration management
├── docker-compose.yml # Docker setup
└── ARCHITECTURE.md    # Detailed architecture docs
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Time To First Audio (TTFA) | < 600 ms |
| Interruption Response | < 150 ms |
| Audio Latency | < 100 ms |

## Contributing

Contributions are welcome! Please read the architecture documentation before submitting PRs.

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- Coqui TTS for the excellent text-to-speech engine
- OpenAI Whisper for speech recognition
- Silero for voice activity detection
