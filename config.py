"""
Typed configuration for the Realtime Voice AI Agent.
Uses pydantic-settings to load from .env file and environment variables.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerConfig(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8765
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_prefix="", extra="ignore")


class OllamaConfig(BaseSettings):
    base_url: str = "http://127.0.0.1:11434"
    model: str = "kimi-k2.5:cloud"
    fallback_model: str = "qwen3.5:27b"
    system_prompt_path: str = "../../SOUL.md"
    max_tokens: int = 250
    temperature: float = 0.25
    stream: bool = True

    model_config = SettingsConfigDict(env_prefix="OLLAMA_", extra="ignore")

    def load_system_prompt(self) -> str:
        """Load the system prompt from SOUL.md or return a default."""
        path = Path(self.system_prompt_path)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return (
            "Eres un asistente de inteligencia artificial personal, formal, "
            "competente y con humor seco. Responde en español latino."
        )


class WhisperConfig(BaseSettings):
    model_size: str = "base"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str | None = "es"
    beam_size: int = 1  # Faster for realtime
    vad_filter: bool = False  # We handle VAD separately

    model_config = SettingsConfigDict(env_prefix="WHISPER_", extra="ignore")


class PiperConfig(BaseSettings):
    model_path: str = "models/piper/es_MX-claude-medium.onnx"
    config_path: str = "models/piper/es_MX-claude-medium.onnx.json"
    speaker_id: int | None = 0
    length_scale: float = 1.0  # Speed: <1.0 = faster, >1.0 = slower
    noise_scale: float = 0.667
    noise_w: float = 0.8

    model_config = SettingsConfigDict(env_prefix="PIPER_", extra="ignore")


class VADConfig(BaseSettings):
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 300
    sample_rate: int = 16000
    window_size_samples: int = 512  # 32ms at 16kHz

    model_config = SettingsConfigDict(env_prefix="VAD_", extra="ignore")


class AudioConfig(BaseSettings):
    sample_rate_input: int = 16000
    sample_rate_output: int = 22050
    chunk_size_ms: int = 30
    channels: int = 1
    sample_width: int = 2  # 16-bit PCM
    crossfade_ms: int = 60
    queue_max_chunks: int = 10

    model_config = SettingsConfigDict(extra="ignore")


class InterruptConfig(BaseSettings):
    fade_out_ms: int = 50
    reaction_target_ms: int = 150

    model_config = SettingsConfigDict(env_prefix="INTERRUPT_", extra="ignore")


class CacheConfig(BaseSettings):
    redis_url: str = "redis://localhost:6379/0"
    ttl_seconds: int = 3600

    model_config = SettingsConfigDict(env_prefix="CACHE_", extra="ignore")


class OpenClawConfig(BaseSettings):
    gateway_url: str = "http://localhost:18789"
    gateway_token: str = ""

    model_config = SettingsConfigDict(env_prefix="OPENCLAW_", extra="ignore")


class Settings(BaseSettings):
    """Root settings that aggregates all config sections."""

    server: ServerConfig = ServerConfig()
    ollama: OllamaConfig = OllamaConfig()
    whisper: WhisperConfig = WhisperConfig()
    piper: PiperConfig = PiperConfig()
    vad: VADConfig = VADConfig()
    audio: AudioConfig = AudioConfig()
    interrupt: InterruptConfig = InterruptConfig()
    cache: CacheConfig = CacheConfig()
    openclaw: OpenClawConfig = OpenClawConfig()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Singleton instance
settings = Settings()
