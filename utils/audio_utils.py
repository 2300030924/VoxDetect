import librosa
import numpy as np
import soundfile as sf


def load_audio(path, sample_rate=16000):
    """
    Load audio file, convert to mono, resample to 16kHz
    """
    audio, sr = librosa.load(path, sr=sample_rate, mono=True)
    return audio, sample_rate


def remove_silence(audio, sample_rate=16000):
    """
    Removes silence at the beginning and end of the audio.
    Uses librosa.effects.trim with an energy threshold.
    """
    trimmed_audio, _ = librosa.effects.trim(audio, top_db=20)
    return trimmed_audio


def normalize_audio(audio):
    """
    Normalizes audio amplitude to -1 to +1 range.
    """
    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = audio / max_amp
    return audio.astype(np.float32)


def save_audio(path, audio, sample_rate=16000):
    """
    Saves audio to a file using soundfile.
    """
    sf.write(path, audio, sample_rate)
