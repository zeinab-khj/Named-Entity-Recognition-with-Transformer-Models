from pathlib import Path


# =========================
# Project Paths
# =========================

ROOT_DIR = Path(__file__).resolve().parent.parent


DATA_DIR = ROOT_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"


MODEL_DIR = ROOT_DIR / "models"

OUTPUT_DIR = ROOT_DIR / "outputs"



# =========================
# Dataset Paths
# =========================

RAW_DATA_PATH = RAW_DATA_DIR / "dataset.csv"

TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train"

VALID_DATA_PATH = PROCESSED_DATA_DIR / "validation"



# =========================
# Model Configuration
# =========================

MODEL_NAME = "roberta-base"

NUM_LABELS = 77



# =========================
# Training Configuration
# =========================

SEED = 42

BATCH_SIZE = 16

NUM_EPOCHS = 5


# =========================
# Evaluation
# =========================

METRIC_FOR_BEST_MODEL = "macro_f1"

REPORT_DIR = ROOT_DIR / "reports"
PLOT_DIR = REPORT_DIR / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
