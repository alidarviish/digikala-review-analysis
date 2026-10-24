from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
SPLITS_DIR = DATA_DIR / "splits"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

COMMENTS_CSV = RAW_DIR / "digikala-comments.csv"
PRODUCTS_CSV = RAW_DIR / "digikala-products.csv"

# Fixed by the project protocol, shared by everyone. Do not change.
SPLIT_SEED = 2026
MASK_SEED = "digikala-semi-supervised-v1"
MASK_RATE = 0.20
SPLIT_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
PSEUDO_LABEL_THRESHOLD = 0.90

RECOMMENDATION_CLASSES = ["recommended", "not_recommended", "no_idea"]

# rating_band: low = [0, 2.5), medium = [2.5, 4.0), high = [4.0, 5.0]
RATING_BAND_EDGES = (2.5, 4.0)
RATING_BAND_CLASSES = ["low", "medium", "high"]  # ordinal: 0, 1, 2
RATE_MIN, RATE_MAX = 0.0, 5.0

# Parquet copies of the raw CSVs (same values, every column kept as text).
# Each one is a folder of part files written chunk by chunk; pandas reads the folder as one table.
COMMENTS_RAW_DIR = INTERIM_DIR / "comments_raw"
PRODUCTS_RAW_DIR = INTERIM_DIR / "products_raw"

# Cleaned comments (typed columns, duplicates resolved, raw text untouched)
COMMENTS_CLEAN_PARQUET = INTERIM_DIR / "comments_clean.parquet"
EXCLUDED_ROWS_PARQUET = INTERIM_DIR / "excluded_rows.parquet"

# How rate == 0 is treated for rating_band:
#   "low"     -> follow the project bands as written (0 <= rate < 2.5 is low)
#   "invalid" -> treat 0 as "no rating given" and leave rating_band empty
# The project brief requires evidence and approval before changing the bands,
# so the default stays "low" until the zero-rating analysis is done.
RATE_ZERO_POLICY = "low"
