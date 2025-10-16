# src/config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Data
DATA_PKL = ROOT / "data" / "processed_dataset.pkl"
ACCENT2ID_JSON = ROOT / "data" / "accent2id.json"

# Checkpoints
CHECKPOINT_DIR = ROOT / "checkpoints" / "fine_tuned"
BASE_CHECKPOINT_DIR = ROOT / "checkpoints" / "base"

# Model / tokenizer
HF_MODEL_NAME = "facebook/mms-tts-vie"
TOKENIZER_NAME = HF_MODEL_NAME

# Training hyperparams
BATCH_SIZE = 4
NUM_EPOCHS = 10
LR = 1e-4
ACCENT_EMB_DIM = 256
DEVICE = "cuda" 

# Audio / training targets
TARGET_SAMPLE_RATE = 16000
MAX_WAVEFORM_LENGTH = None  # in samples, None = don't truncate; set if needed
