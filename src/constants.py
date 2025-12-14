from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic_samples"
EVALUATED_DATA_DIR = DATA_DIR / "evaluated_samples"
SYNTHETIC_DATA_FILE = SYNTHETIC_DATA_DIR / "synthetic_samples.json"
ANNOTATED_SYNTHETIC_DATA_FILE = SYNTHETIC_DATA_DIR / "annotated_synthetic_samples.json"
TRAIN_ANNOTATED_SYNTHETIC_DATA_FILE = SYNTHETIC_DATA_DIR / "annotated_synthetic_samples_train.json"
TEST_ANNOTATED_SYNTHETIC_DATA_FILE = SYNTHETIC_DATA_DIR / "annotated_synthetic_samples_test.json"


# Train test split ratio
TRAIN_TEST_SPLIT_RATIO = 0.8

# Other
SEED = 42
