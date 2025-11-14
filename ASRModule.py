import whisper
import numpy as np

class ASRModel:
    def __init__(self, model_name="small"):
        print(f"Loading Whisper model: {model_name} ...")
        self.model = whisper.load_model(model_name)

    def transcribe_audio(self, audio, sr=16000):
        """
        Runs Whisper on an audio numpy array and returns:
        - text
        - segments (with timestamps)
        - confidence score approximation
        """

        # Whisper requires float32, 16kHz
        audio = np.array(audio, dtype=np.float32)

        result = self.model.transcribe(audio, fp16=False)

        transcript_text = result.get("text", "").strip()
        segments = result.get("segments", [])

        # Estimate confidence (Whisper doesn't give real confidence)
        confidence = 0.90 if transcript_text else 0.50

        return transcript_text, segments, confidence
