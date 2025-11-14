import numpy as np
import librosa
from sklearn.cluster import KMeans


def extract_mfcc(audio, sample_rate=16000):
    """
    Extracts MFCC features from the audio.
    """
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sample_rate, n_mfcc=13, hop_length=512, n_fft=1024
    )
    return mfcc.T  # shape: (num_frames, features)


def cluster_speakers(audio, sample_rate=16000, num_speakers=2):
    """
    Performs simple KMeans clustering on MFCC frames
    to divide audio into multiple speakers.
    """
    mfcc = extract_mfcc(audio, sample_rate)
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=num_speakers, random_state=0, n_init="auto")
    labels = kmeans.fit_predict(mfcc)

    return labels, mfcc


def reconstruct_speakers(audio, labels, sample_rate=16000, num_speakers=2):
    """
    Reconstructs separate audio tracks from clustering labels.
    """
    frame_length = 1024
    hop_length = 512

    speakers = [np.zeros_like(audio) for _ in range(num_speakers)]
    frame_index = 0

    for i, label in enumerate(labels):
        start = i * hop_length
        end = start + frame_length
        if end <= len(audio):
            speakers[label][start:end] += audio[start:end]

    return speakers


def cosine_similarity(a, b):
    """
    Computes cosine similarity between two vectors.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_target_speaker(separated_speakers, target_audio, sr=16000):
    # Mean MFCC of target (fixed length)
    target_mfcc = extract_mfcc(target_audio, sr).mean(axis=0)

    scores = []
    for sp in separated_speakers:
        sp_mfcc = extract_mfcc(sp, sr).mean(axis=0)
        sim = cosine_similarity(target_mfcc, sp_mfcc)
        scores.append(sim)

    return np.argmax(scores)
  # highest similarity → target speaker
