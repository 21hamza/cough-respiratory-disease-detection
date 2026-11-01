"""Shared settings. Every model reads from here so the comparison stays fair."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
COSWARA_REPO = DATA / "coswara_repo"        # sparse clone of github.com/iiscleap/Coswara-Data
RAW = DATA / "raw"                          # raw/<batch>/<participant_id>/cough-heavy.wav
PROCESSED = DATA / "processed"              # cached features + splits
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Coswara collection batches used. Only 2021-22 batches contain both healthy and positive
# participants in useful numbers, so classes can be balanced within each batch.
BATCHES = ["20210406", "20210419", "20210426", "20210507", "20210523", "20210603", "20210618",
           "20210630", "20210714", "20210816", "20210830", "20210914", "20210930", "20220116", "20220224"]

# Audio / features
SR = 16000          # sample rate (Hz)
N_FFT = 512         # STFT window (32 ms)
HOP = 160           # STFT hop (10 ms)
N_MELS = 128        # Mel bands
MAX_SEC = 4.0       # every clip padded/cropped to 4 s -> 401 frames
N_MFCC = 13

# Training (M2, M3)
SEED = 42
EPOCHS = 40
BATCH = 16
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8        # early stopping on validation loss

# Pretrained YAMNet (M4): official TF Hub YAMNet in TFLite form, taken from Google's tflite-support repository
YAMNET_REPO = "https://github.com/tensorflow/tflite-support.git"
YAMNET_PATH_IN_REPO = "tensorflow_lite_support/metadata/python/tests/testdata/audio_classifier/yamnet_tfhub.tflite"
YAMNET_LABELS_IN_REPO = "tensorflow_lite_support/metadata/python/tests/testdata/audio_classifier/yamnet_521_labels.txt"
PRETRAINED = DATA / "pretrained"


def metadata_csv():
    """Coswara labels: the freshly downloaded copy if present, else the copy shipped in data/metadata/ (CC BY 4.0)."""
    p = COSWARA_REPO / "combined_data.csv"
    return p if p.exists() else DATA / "metadata" / "combined_data.csv"
