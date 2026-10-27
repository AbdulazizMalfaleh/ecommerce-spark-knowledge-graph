"""SILVER layer: clean, typed, deduplicated events you can trust, plus a products dimension.

Every function below only *describes* transformations (lazy). The work happens
when .write is called in run().
"""
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config import BRONZE_DIR, SILVER_DIR, VALID_EVENT_TYPES


def parse_event_time(col: str = "event_time") -> Column:
    """Accept both formats seen in copies of the dataset:
    '2019-10-01 00:00:04 UTC' and '2019-10-01 00:00:04+00:00'.
    try_to_timestamp returns null for garbage instead of failing (Spark 4 runs in ANSI mode)."""
    return F.coalesce(
        F.try_to_timestamp(F.regexp_replace(col, r"\s*UTC$", ""), F.lit("yyyy-MM-dd HH:mm:ss")),
        F.try_to_timestamp(F.col(col), F.lit("yyyy-MM-dd HH:mm:ssXXX")),
    )


def quality_rules(df: DataFrame) -> dict[str, Column]:
    """Each rule is a boolean column: True = the row violates the rule."""
    return {
        "bad_event_time": parse_event_time().isNull(),
        "bad_event_type": ~F.lower(F.trim("event_type")).isin(*VALID_EVENT_TYPES) | F.col("event_type").isNull(),
        "bad_price": F.col("price").isNull() | (F.col("price") <= 0),
        "missing_ids": F.col("product_id").isNull() | F.col("user_id").isNull() | F.col("user_session").isNull(),
    }


def clean_events(bronze: DataFrame) -> DataFrame:
    rules = quality_rules(bronze)
    is_bad = rules["bad_event_time"] | rules["bad_event_type"] | rules["bad_price"] | rules["missing_ids"]
    category_parts = F.split("category_code", r"\.")

    return (
        bronze
        .filter(~is_bad)
        .withColumn("event_time", parse_event_time())
        .withColumn("event_type", F.lower(F.trim("event_type")))
        .withColumn("brand", F.lower(F.trim("brand")))
        # "electronics.audio.headphone" -> main = electronics, sub = headphone
        .withColumn("category_main", F.coalesce(F.element_at(category_parts, 1), F.lit("unknown")))
        .withColumn("category_sub", F.element_at(category_parts, -1))
        .dropDuplicates(["event_time", "event_type", "product_id", "user_id", "user_session", "price"])
        .withColumn("event_date", F.to_date("event_time"))
        .withColumn("event_hour", F.hour("event_time"))
        .withColumn("day_of_week", F.date_format("event_time", "E"))     # Mon, Tue, ...
        .select("event_time", "event_date", "event_hour", "day_of_week", "event_type",
                "user_id", "user_session", "product_id", "category_id", "category_code",
                "category_main", "category_sub", "brand", "price")
    )


def build_products(events: DataFrame) -> DataFrame:
    """One row per product with its most recent attributes (price changes over time).
    max_by(x, event_time) = the value of x on the latest event: one aggregation, no window."""
    return events.groupBy("product_id").agg(
        F.max_by("category_id", "event_time").alias("category_id"),
        F.max_by("category_code", "event_time").alias("category_code"),
        F.max_by("category_main", "event_time").alias("category_main"),
        F.max_by("category_sub", "event_time").alias("category_sub"),
        F.max_by("brand", "event_time").alias("brand"),
        F.max_by("price", "event_time").alias("current_price"),
        F.min("event_time").alias("first_seen"),
        F.max("event_time").alias("last_seen"),
    )


def quality_report(bronze: DataFrame) -> dict[str, int]:
    """All rule violations counted in ONE pass over the data (one job, not one per rule)."""
    rules = quality_rules(bronze)
    row = bronze.agg(
        F.count("*").alias("total"),
        *[F.sum(rule.cast("int")).alias(name) for name, rule in rules.items()],
    ).first()
    return row.asDict()


def run(spark: SparkSession) -> None:
    print("\n=== SILVER ===")
    bronze = spark.read.parquet(str(BRONZE_DIR / "events"))

    report = quality_report(bronze)
    print(f"  data quality ({report['total']:,} bronze rows):")
    for name, value in report.items():
        if name != "total":
            print(f"    {name:<16} {value:>10,}")

    (clean_events(bronze).write.mode("overwrite")
        .partitionBy("event_date")              # date filters later skip whole folders
        .parquet(str(SILVER_DIR / "events")))
    events = spark.read.parquet(str(SILVER_DIR / "events"))
    kept = events.count()
    removed = report["total"] - kept
    print(f"  events   {report['total']:>12,} -> {kept:>12,}   removed {removed:,} "
          f"({removed / max(report['total'], 1):.2%}, incl. duplicates)")

    build_products(events).write.mode("overwrite").parquet(str(SILVER_DIR / "products"))
    print(f"  products {spark.read.parquet(str(SILVER_DIR / 'products')).count():>12,} rows")
