import json

class JSONBuilder:
    """
    Builds diarization.json in the required format.
    """

    def __init__(self):
        self.entries = []

    def add_entry(self, speaker, start, end, text, confidence):
        self.entries.append({
            "speaker": speaker,
            "start": float(start),
            "end": float(end),
            "text": text,
            "confidence": float(confidence)
        })

    def save(self, path="outputs/diarization.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=4)
        print(f"Diarization JSON saved at: {path}")
