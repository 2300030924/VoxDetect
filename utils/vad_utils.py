import numpy as np
import librosa


def frame_audio(audio, frame_size, hop_size, sample_rate=16000):
    """
    Splits audio into overlapping frames.
    Returns a 2D array: shape = (num_frames, frame_samples)
    """
    frame_length = int(frame_size * sample_rate)
    hop_length = int(hop_size * sample_rate)
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length).T
    return frames


def compute_energy(frames):
    """
    Computes short-term energy of each frame.
    Energy = sum of squares of samples.
    """
    return np.sum(frames ** 2, axis=1)


def vad_segments(audio, sample_rate=16000, frame_size=0.03, hop_size=0.015, energy_threshold=0.0005):
    """
    Simple energy-based VAD.
    Returns a list of (start_time, end_time) speech segments.
    """
    frames = frame_audio(audio, frame_size, hop_size, sample_rate)
    energies = compute_energy(frames)

    speech_flags = energies > energy_threshold

    segments = []
    start = None

    frame_duration = hop_size  # seconds per hop

    for i, is_speech in enumerate(speech_flags):
        if is_speech and start is None:
            start = i * frame_duration

        elif not is_speech and start is not None:
            end = i * frame_duration
            segments.append((start, end))
            start = None

    # If audio ends with speech
    if start is not None:
        end = len(speech_flags) * frame_duration
        segments.append((start, end))

    return segments
