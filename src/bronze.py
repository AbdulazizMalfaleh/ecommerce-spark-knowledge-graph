"""BRONZE layer: land the raw CSV event files as Parquet, unchanged, plus ingestion metadata.

Why an explicit schema instead of inferSchema=True?
    inferSchema forces Spark to scan the whole file *immediately* just to guess the
    column types: an eager job that breaks lazy evaluation (and on a 5 GB file, a slow one).
    With a schema defined up front, spark.read is pure metadata and nothing runs
    until the write (the Action).

event_time is kept as a raw string here; parsing it is a cleaning step (Silver).
"""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

from src.config import BRONZE_DIR, RAW_DIR

EVENTS_SCHEMA = StructType([
    StructField("event_time", StringType()),
    StructField("event_type", StringType()),
    StructField("product_id", LongType()),
    StructField("category_id", LongType()),
    StructField("category_code", StringType()),
    StructField("brand", StringType()),
    StructField("price", DoubleType()),
    StructField("user_id", LongType()),
    StructField("user_session", StringType()),
])


def read_raw(spark: SparkSession) -> DataFrame:
    return (
        spark.read
        .option("header", True)
        .option("mode", "PERMISSIVE")          # malformed values become null instead of failing the job
        .option("pathGlobFilter", "*.csv")     # every CSV in data/raw/
        .schema(EVENTS_SCHEMA)
        .csv(str(RAW_DIR))
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_source_file", F.input_file_name())
    )


def run(spark: SparkSession) -> None:
    print("\n=== BRONZE ===")
    files = sorted(p.name for p in RAW_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {RAW_DIR}. Run src.download_data or src.generate_sample_data first.")
    print(f"  source files: {', '.join(files)}")

    read_raw(spark).write.mode("overwrite").parquet(str(BRONZE_DIR / "events"))   # Action -> job runs
    rows = spark.read.parquet(str(BRONZE_DIR / "events")).count()
    print(f"  events -> {BRONZE_DIR / 'events'}  ({rows:,} rows)")
