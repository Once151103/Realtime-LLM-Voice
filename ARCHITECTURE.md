# Realtime Voice AI Agent Architecture (Jarvis-Style)

## Overview

This document describes the full architecture required to build a **low-latency realtime voice assistant** capable of:

* Streaming speech generation
* Phoneme / micro-chunk synthesis
* User interruption (barge-in)
* Predictive speech buffering
* Continuous conversation state

The system is designed for **local execution with GPU acceleration**.

---

# System Goals

* Time To First Audio (TTFA): **< 600 ms**
* Continuous speech streaming
* Natural conversational interruption
* Modular microservice design
* Replaceable TTS / LLM engines

---

# High Level Architecture

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

---

# Core Services

## 1. Voice Orchestrator Service (Central Brain)

This service coordinates ALL pipelines.

Responsibilities:

* Maintain conversation state machine
* Route audio + text streams
* Handle interruption signals
* Manage audio queue lifecycle
* Control LLM streaming sessions
* Dispatch TTS jobs
* Predict chunk boundaries

### State Machine

| State | Description |
| --------- | ----------------- |
| LISTENING | User speaking |
| THINKING | LLM generating |
| SPEAKING | Assistant talking |

Transitions:

```
LISTENING → THINKING
THINKING → SPEAKING
SPEAKING → LISTENING (interruption)
```

---

## 2. Voice Activity Detection (Barge-In Trigger)

Purpose:

Detect user speech while assistant is talking.

Recommended:

* silero VAD
* webrtc VAD

### Behavior

If speech detected:

```
interrupt_flag = true
```

Then:

* Stop playback
* Stop TTS synthesis
* Abort LLM stream
* Clear audio queue

---

## 3. Streaming LLM Engine

Must support:

* token streaming
* abort controller
* low temperature responses

### Example Flow

```
for token in llm_stream:
 append_to_sentence_buffer()
 if boundary_detected:
 dispatch_chunk_to_TTS()
```

---

## 4. Sentence / Phoneme Chunker

Goal:

Reduce perceived latency by sending speech early.

### Chunk Strategy

| Context | Words per chunk |
| --------------------- | --------------- |
| Greeting | 4-6 |
| Normal explanation | 8-15 |
| Technical description | 12-20 |

### Advanced

Predict chunk boundary on:

* commas
* conjunctions
* semantic pause
* tone shift

---

## 5. TTS Streaming Worker

Loads model once.

### Prewarm Procedure

```
load_model()
run_dummy_inference()
keep_gpu_context_alive()
```

### Chunk Synthesis Loop

```
while true:
 chunk = queue.pop()
 audio = synthesize(chunk)
 audio_queue.push(audio)
```

---

## 6. Audio Queue and Playback Engine

Responsible for:

* sequential playback
* crossfade blending
* interruption cancelation
* volume normalization

### Crossfade Strategy

* 40–80 ms overlap
* prevents robotic gaps

---

# Interruption (Barge-In) Handling

## Required Mechanisms

* Interrupt flag shared across services
* Abortable async tasks
* Playback fade-out

### Example Flow

```
User speaks
↓
VAD detects speech
↓
Interrupt signal emitted
↓
Playback fade-out
↓
Clear audio queue
↓
Abort LLM stream
↓
Switch to LISTENING
```

Reaction target:

**<150 ms**

---

# Phoneme / Micro-Chunk Streaming

True phoneme streaming is complex.

Recommended practical approach:

* micro sentence chunks
* predictive synthesis
* audio buffering

Optional advanced approach:

* expose phoneme frontend
* synthesize 100–200 ms speech windows

---

# Predictive Speech (Advanced Optimization)

Technique:

Start generating audio BEFORE full sentence confirmed.

Example:

LLM starts:

> "Certainly I will now initialize…"

System sends:

> "Certainly I will now…"

If LLM changes direction:

discard buffer.

---

# Performance Optimization

## GPU

* fp16 inference
* CUDA graphs
* batch size = 1
* pinned memory

## Audio

* sample rate 22050
* queue length = 2 chunks
* fade-out interruption

## LLM

* streaming enabled
* sentence boundary predictor
* response token limit tuning

---

# Caching Strategy

Cache frequent assistant responses:

* acknowledgements
* greetings
* confirmations

Reduces latency to **0 ms synthesis**.

---

# Recommended Stack

| Component | Tool |
| --------- | ---------------------- |
| STT | Whisper streaming |
| LLM | Local inference engine |
| TTS | Coqui XTTS |
| VAD | Silero |
| Transport | WebRTC / WebSocket |
| Memory | Redis |

---

# Future Enhancements

* emotional prosody tokens
* gaze / avatar sync
* predictive dialogue planning
* multi-speaker context memory
* tool-driven speech interruption

---

# Conclusion

A realtime voice assistant is not a single model.

It is a **coordinated orchestration system** where latency depends more on architecture than hardware.

Correct implementation results in:

* natural conversation flow
* cinematic assistant feeling
* immediate user control via interruption
* scalable voice intelligence layer
