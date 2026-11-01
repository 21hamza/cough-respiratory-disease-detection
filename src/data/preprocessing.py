"""Audio preprocessing shared by all models (Report Section IV-B, Listing 1).

Pipeline: load -> mono -> 16 kHz -> spectral gating -> energy VAD -> amplitude normalisation
          -> pad/crop to 4 s -> log-Mel (M2, M3) or MFCC statistics (M1).
"""
import numpy as np
import torch, torchaudio, librosa, noisereduce as nr, soundfile as sf
from src import config as C

_mel = torchaudio.transforms.MelSpectrogram(sample_rate=C.SR, n_fft=C.N_FFT, hop_length=C.HOP, n_mels=C.N_MELS)


def load_audio(path, min_sec=0.3):
    """Read a WAV file as mono float32 at C.SR. Returns None for unreadable or too-short files."""
    try:
        y, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        return None
    y = y.mean(1)
    if len(y) < min_sec * sr:
        return None
    if sr != C.SR:
        y = torchaudio.functional.resample(torch.from_numpy(y), sr, C.SR).numpy()
    return y


def energy_vad(y, frame=400, hop=160, k=1.5, pad=0.1):
    """Keep the span from the first to the last frame whose RMS exceeds k x median RMS, plus padding."""
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    active = rms > k * np.median(rms)
    if not active.any():
        return y
    idx = np.where(active)[0]
    s = max(0, idx[0] * hop - int(pad * C.SR))
    e = min(len(y), idx[-1] * hop + frame + int(pad * C.SR))
    return y[s:e]


def clean(y):
    """Denoise, segment, normalise and fix length."""
    y = nr.reduce_noise(y=y, sr=C.SR, stationary=True)          # spectral gating
    y = energy_vad(y)
    y = (y - y.mean()) / (y.std() + 1e-8)
    n = int(C.MAX_SEC * C.SR)
    return np.pad(y, (0, max(0, n - len(y))))[:n].astype(np.float32)


def log_mel(y):
    """(128, 401) log-Mel spectrogram used by M2 and M3."""
    return torch.log(_mel(torch.from_numpy(y)) + 1e-6).numpy()


def mfcc_stats(y):
    """78-dim vector for M1: mean and std of 13 MFCCs and their deltas and delta-deltas."""
    m = librosa.feature.mfcc(y=y, sr=C.SR, n_mfcc=C.N_MFCC)
    f = np.vstack([m, librosa.feature.delta(m), librosa.feature.delta(m, order=2)])
    return np.concatenate([f.mean(1), f.std(1)])


def clean_peak(y):
    """Variant for M4 (YAMNet expects a waveform in [-1, 1]): denoise, segment, peak-normalise, pad to 4 s."""
    y = nr.reduce_noise(y=y, sr=C.SR, stationary=True)
    y = energy_vad(y)
    y = 0.9 * y / (np.abs(y).max() + 1e-8)
    n = int(C.MAX_SEC * C.SR)
    return np.pad(y, (0, max(0, n - len(y))))[:n].astype(np.float32)


def add_noise(y, snr_db, rng):
    """Add white Gaussian noise at a given signal-to-noise ratio (used for the robustness test)."""
    p_signal = np.mean(y ** 2)
    noise = rng.standard_normal(len(y)).astype(np.float32)
    noise *= np.sqrt(p_signal / (10 ** (snr_db / 10)) / (np.mean(noise ** 2) + 1e-12))
    return (y + noise).astype(np.float32)
