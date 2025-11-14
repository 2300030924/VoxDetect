import librosa
import numpy as np
import soundfile as sf
from utils.audio_utils import load_audio, remove_silence, normalize_audio
from utils.vad_utils import vad_segments
from SpeakerSeparation import cluster_speakers, reconstruct_speakers, find_target_speaker


def extract_target_speaker(mixture_path, target_path, output_path="outputs/target_speaker.wav"):
    """
    Full pipeline:
    1. Load mixture + target sample
    2. Remove silence + normalize
    3. VAD to detect speech regions
    4. KMeans clustering to separate speakers
    5. Find matching speaker using MFCC
    6. Save target speaker audio
    7. Return target speaker audio for further processing (ASR + diarization)
    """

    # Step 1: Load audios
    mixture_audio, sr = load_audio(mixture_path)
    target_audio, _ = load_audio(target_path)

    # Step 2: Clean audio
    mixture_audio = remove_silence(mixture_audio, sr)
    mixture_audio = normalize_audio(mixture_audio)
    mixture_audio = mixture_audio.astype(np.float32)


    # Step 3: VAD → speech segments
    segments = vad_segments(mixture_audio, sr)

    print("\nDetected Speech Segments (VAD):")
    for seg in segments:
        print(" -", seg)

    # Step 4: Clustering-based speaker separation
    labels, _ = cluster_speakers(mixture_audio, sr, num_speakers=3)


    # Step 5: Reconstruct separate speaker tracks
    separated_speakers = reconstruct_speakers(mixture_audio, labels, sr, num_speakers=3)


    # Step 6: Identify target speaker
    target_index = find_target_speaker(separated_speakers, target_audio, sr)
    target_sp_audio = separated_speakers[target_index]

    # Step 7: Save output
    sf.write(output_path, target_sp_audio, sr)
    print(f"\nTarget speaker extracted and saved to: {output_path}")

    return target_sp_audio, sr, segments, labels
