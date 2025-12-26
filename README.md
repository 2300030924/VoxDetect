# VoxDetect — Target Speaker Diarization & ASR System

VoxDetect is an end-to-end audio intelligence system designed to isolate a
target speaker from multi-speaker conversations, perform speaker diarization,
and generate accurate, timestamped speech transcriptions.

The system supports both **offline batch processing** and **real-time
streaming**, making it suitable for meeting analysis, call intelligence,
and conversational AI applications.

---

## 🎯 Objective
The goal of VoxDetect is to identify and extract a specific speaker’s voice
from overlapping conversations using a short reference audio clip, while
simultaneously transcribing all speakers with accurate timestamps and
punctuation.

Key objectives:
- Separate a target speaker using a short reference sample
- Perform speaker diarization and ASR
- Restore punctuation and casing
- Support offline and streaming inference
- Output structured, machine-readable results

---

## 📥 Input & 📤 Output

### Input
- `mixture_audio.wav` — multi-speaker audio recording
- `target_sample.wav` — 3–10 second reference clip of the target speaker

### Output
- `target_speaker.wav` — clean isolated audio of the target speaker
- `diarization.json` — per-speaker transcription with timestamps and confidence

### Example Output
```json
[
  {
    "speaker": "Target",
    "start": 0.45,
    "end": 5.62,
    "text": "Hello, how are you?",
    "confidence": 0.97
  },
  {
    "speaker": "Speaker_B",
    "start": 5.63,
    "end": 10.25,
    "text": "I'm doing well, thank you.",
    "confidence": 0.95
  }
]
