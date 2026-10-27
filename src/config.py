"""Central configuration: paths and pipeline parameters."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"          # original CSV event files
BRONZE_DIR = DATA_DIR / "bronze"    # raw data stored as Parquet + ingestion metadata
SILVER_DIR = DATA_DIR / "silver"    # cleaned, typed, deduplicated
GOLD_DIR = DATA_DIR / "gold"        # business-ready aggregates
GRAPH_DIR = DATA_DIR / "graph"      # knowledge graph: nodes/<Entity>, edges/<RELATION>

# Kaggle: "eCommerce behavior data from multi category store" (REES46)
KAGGLE_DATASET = "mkechinov/ecommerce-behavior-data-from-multi-category-store"
KAGGLE_FILES = ("2019-Oct.csv", "2019-Nov.csv")

VALID_EVENT_TYPES = ("view", "cart", "remove_from_cart", "purchase")

# Gold-layer parameters
MIN_VIEWS_FOR_RANKING = 100   # smoothing strength for product conversion rates
TOP_N_PER_CATEGORY = 10

# Knowledge-graph parameters
BOUGHT_TOGETHER_MIN_CUSTOMERS = 3   # a pair needs at least this many shared buyers
MAX_BASKET_PRODUCTS = 50            # ignore buyers of more distinct products (likely resellers/bots)
PAGERANK_ITERATIONS = 15
PAGERANK_DAMPING = 0.85
