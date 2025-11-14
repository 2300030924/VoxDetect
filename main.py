from TargetDiarization import extract_target_speaker
from ASRModule import ASRModel
from PunctuationModule import PunctuationRestorer
from utils.json_builder import JSONBuilder
from utils.vad_utils import vad_segments
from utils.audio_utils import load_audio

import numpy as np


def run_pipeline():
    mixture_path = "sample_data/mixture_audio.wav"
    target_path = "sample_data/target_sample.wav"

    print("🔹 Extracting target speaker...")
    target_audio, sr, segments, labels = extract_target_speaker(mixture_path, target_path)

    print("🔹 Initializing ASR model...")
    asr = ASRModel()

    print("🔹 Initializing punctuation restorer...")
    pr = PunctuationRestorer()

    print("🔹 Creating JSON builder...")
    js = JSONBuilder()

    print("\n🔹 Running ASR on VAD segments...\n")

    mixture_audio, _ = load_audio(mixture_path)

    # Run ASR on each VAD speech segment
    for i, (start, end) in enumerate(segments):
        start_idx = int(start * sr)
        end_idx = int(end * sr)

        segment_audio = mixture_audio[start_idx:end_idx]

        text, segs, conf = asr.transcribe_audio(segment_audio, sr=sr)
        text = pr.restore(text)

        speaker_label = "Target" if i % 2 == 0 else "Speaker_B"

        js.add_entry(
            speaker=speaker_label,
            start=start,
            end=end,
            text=text,
            confidence=conf
        )

        print(f"Segment {i+1}: {speaker_label} → {text}")

    js.save("outputs/diarization.json")

    print("\n🎉 Pipeline completed successfully!")
    print("👉 Outputs generated in /outputs folder.")


if __name__ == "__main__":
    run_pipeline()
