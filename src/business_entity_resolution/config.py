"""Shared project paths and reproducible validation settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
VALIDATION_DIR = OUTPUT_DIR / "validation"
FINAL_OUTPUT_DIR = OUTPUT_DIR / "final"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
BLOCK_INDEX_DIR = PROJECT_ROOT / "models" / "indexes"
RANDOM_SEED = 42
VALIDATION_FRACTION = 0.20
TRAIN_ENTITY_SAMPLE = 10_000
VALIDATION_ENTITY_SAMPLE = 5_000
SQLITE_BATCH_SIZE = 10_000
BLOCK_EXACT_LIMIT = 300
BLOCK_PREFIX_LIMIT = 40
BLOCK_PREFIX_POOL_SIZE = 500
BLOCK_FTS_NAME_TOP_K = 16
BLOCK_FTS_ADDRESS_TOP_K = 8
BLOCK_FTS_POOL_SIZE = 500
BLOCK_TOKEN_MAX_DOCUMENTS = 10_000
PREDICTION_THRESHOLD_GRID = tuple(i / 100 for i in range(20, 100, 5)) + (0.97, 0.98, 0.99, 0.995)
SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
GROUND_TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")

TRAIN_FILES = {
    "source1": TRAIN_DIR / "train_source1.tsv",
    "source2": TRAIN_DIR / "train_source2.tsv",
    "source3": TRAIN_DIR / "train_source3.tsv",
    "ground_truth": TRAIN_DIR / "train_ground_truth.tsv",
}
TEST_FILES = {
    "source1": TEST_DIR / "test_source1.tsv",
    "source2": TEST_DIR / "test_source2.tsv",
    "source3": TEST_DIR / "test_source3.tsv",
}
