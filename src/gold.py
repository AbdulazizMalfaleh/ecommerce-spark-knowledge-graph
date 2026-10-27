"""GOLD layer: business-ready tables for dashboards, analysis and ML."""
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from src.config import GOLD_DIR, MIN_VIEWS_FOR_RANKING, SILVER_DIR, TOP_N_PER_CATEGORY


def _count_type(event_type: str) -> F.Column:
    return F.sum((F.col("event_type") == event_type).cast("int"))


def _revenue() -> F.Column:
    return F.round(F.sum(F.when(F.col("event_type") == "purchase", F.col("price")).otherwise(0)), 2)


def _ratio(numerator: str, denominator: str, digits: int) -> F.Column:
    """numerator / denominator, or null when the denominator is 0.
    Spark 4 runs in ANSI mode, where a plain "/" by zero fails the whole job
    (e.g. a category with no carts, or a product bought without a recorded view)."""
    return F.round(F.try_divide(F.col(numerator), F.col(denominator)), digits)


def daily_kpis(events: DataFrame) -> DataFrame:
    return (
        events.groupBy("event_date")
        .agg(
            _count_type("view").alias("views"),
            _count_type("cart").alias("carts"),
            _count_type("purchase").alias("purchases"),
            _revenue().alias("revenue"),
            F.countDistinct("user_id").alias("visitors"),
            F.countDistinct(F.when(F.col("event_type") == "purchase", F.col("user_id"))).alias("buyers"),
            F.countDistinct("user_session").alias("sessions"),
        )
        .withColumn("conversion_rate", _ratio("buyers", "visitors", 4))
        .withColumn("avg_order_value", _ratio("revenue", "purchases", 2))
        .orderBy("event_date")
    )


def funnel_by_category(events: DataFrame) -> DataFrame:
    return (
        events.groupBy("category_main")
        .agg(
            _count_type("view").alias("views"),
            _count_type("cart").alias("carts"),
            _count_type("purchase").alias("purchases"),
            _revenue().alias("revenue"),
        )
        .withColumn("view_to_cart", _ratio("carts", "views", 4))
        .withColumn("cart_to_purchase", _ratio("purchases", "carts", 4))
        .withColumn("view_to_purchase", _ratio("purchases", "views", 4))
        .orderBy(F.desc("revenue"))
    )


def product_performance(events: DataFrame, products: DataFrame, m: int = MIN_VIEWS_FOR_RANKING) -> DataFrame:
    """Per-product funnel plus a *smoothed* conversion rate (Bayesian / additive smoothing):

        smoothed_cr = (purchases + m * C) / (views + m)
        C = conversion rate across all products, m = smoothing strength

    A product with 2 views and 1 purchase (50%) should not beat one with 5,000 views
    converting at 8%. Smoothing pulls low-traffic products toward the global average.
    """
    per_product = events.groupBy("product_id").agg(
        _count_type("view").alias("views"),
        _count_type("cart").alias("carts"),
        _count_type("purchase").alias("purchases"),
        _revenue().alias("revenue"),
        F.countDistinct("user_id").alias("unique_users"),
    )
    # Global rate via a 1-row crossJoin instead of .collect(): keeps the plan fully lazy.
    global_cr = per_product.agg(F.try_divide(F.sum("purchases"), F.sum("views")).alias("C"))

    return (
        products.join(per_product, "product_id")
        .crossJoin(global_cr)
        .withColumn("conversion_rate", _ratio("purchases", "views", 4))
        .withColumn("smoothed_cr",
                    F.round((F.col("purchases") + F.lit(m) * F.col("C")) / (F.col("views") + F.lit(m)), 4))
        .drop("C")
    )


def top_products_by_category(perf: DataFrame, n: int = TOP_N_PER_CATEGORY) -> DataFrame:
    w = Window.partitionBy("category_main").orderBy(F.desc("smoothed_cr"), F.desc("purchases"))
    return (
        perf.filter(F.col("purchases") > 0)
        .withColumn("rank", F.row_number().over(w))
        .filter(F.col("rank") <= n)
        .select("category_main", "rank", "product_id", "brand", "category_code", "current_price",
                "views", "purchases", "conversion_rate", "smoothed_cr", "revenue")
    )


def sessions(events: DataFrame) -> DataFrame:
    return (
        events.groupBy("user_session", "user_id")
        .agg(
            F.min("event_time").alias("session_start"),
            F.max("event_time").alias("session_end"),
            F.count("*").alias("events"),
            _count_type("view").alias("views"),
            _count_type("cart").alias("carts"),
            _count_type("purchase").alias("purchases"),
            _revenue().alias("revenue"),
            F.countDistinct("product_id").alias("distinct_products"),
        )
        .withColumn("duration_sec",
                    F.unix_timestamp("session_end") - F.unix_timestamp("session_start"))
        .withColumn("converted", F.col("purchases") > 0)
    )


def hourly_heatmap(events: DataFrame) -> DataFrame:
    return events.groupBy("day_of_week", "event_hour").agg(
        F.count("*").alias("events"),
        _count_type("purchase").alias("purchases"),
        _revenue().alias("revenue"),
    )


def customer_rfm(events: DataFrame) -> DataFrame:
    """RFM segmentation of buyers.
    R = days since last purchase, F = number of orders (sessions with a purchase), M = total spend.
    R and M are scored 1-5 by quintile (approximate, computed in parallel). F uses fixed thresholds, because most buyers
    have exactly 1 order and quintiles would split identical customers arbitrarily."""
    purchases = events.filter(F.col("event_type") == "purchase")
    ref_date = purchases.agg(F.date_add(F.max("event_date"), 1).alias("ref_date"))

    rfm = (
        purchases.groupBy("user_id")
        .agg(
            F.max("event_date").alias("last_purchase"),
            F.countDistinct("user_session").alias("frequency"),
            F.round(F.sum("price"), 2).alias("monetary"),
        )
        .crossJoin(ref_date)
        .withColumn("recency_days", F.datediff("ref_date", "last_purchase"))
        .drop("ref_date")
    )
    # Quintile cut-points in ONE distributed aggregation, then score each row against them.
    # (ntile(5).over(Window.orderBy(...)) would give the same idea but has no partitionBy,
    #  so Spark moves every buyer into a single partition: one core does all the work.)
    cuts = rfm.agg(
        F.percentile_approx("recency_days", [0.2, 0.4, 0.6, 0.8]).alias("r_cuts"),
        F.percentile_approx("monetary", [0.2, 0.4, 0.6, 0.8]).alias("m_cuts"),
    )

    def quintile(col: str, cuts_col: str) -> F.Column:
        """1..5 = 1 + how many cut-points the value is above. Ties always get the same score."""
        return F.lit(1) + sum((F.col(col) > F.col(cuts_col)[i]).cast("int") for i in range(4))

    rfm = (
        rfm.crossJoin(cuts)
        .withColumn("r_score", 6 - quintile("recency_days", "r_cuts"))      # most recent = 5
        .withColumn("m_score", quintile("monetary", "m_cuts"))
        .drop("r_cuts", "m_cuts")
        .withColumn("f_score",
                    F.when(F.col("frequency") >= 6, 5).when(F.col("frequency") >= 4, 4)
                     .when(F.col("frequency") == 3, 3).when(F.col("frequency") == 2, 2).otherwise(1))
    )
    r, f = F.col("r_score"), F.col("f_score")
    return rfm.withColumn(
        "segment",
        F.when((r >= 4) & (f >= 4), "Champions")
         .when((r >= 3) & (f >= 3), "Loyal")
         .when((r >= 4) & (f <= 2), "New / Promising")
         .when((r <= 2) & (f >= 3), "At Risk")
         .when((r <= 2) & (f <= 2), "Hibernating")
         .otherwise("Needs Attention"),
    )


def run(spark: SparkSession) -> None:
    print("\n=== GOLD ===")
    events = spark.read.parquet(str(SILVER_DIR / "events"))
    products = spark.read.parquet(str(SILVER_DIR / "products"))

    product_performance(events, products).write.mode("overwrite").parquet(str(GOLD_DIR / "product_performance"))
    # Re-read the written result so top_products doesn't recompute the per-product aggregation.
    perf = spark.read.parquet(str(GOLD_DIR / "product_performance"))

    tables = {
        "daily_kpis": daily_kpis(events),
        "funnel_by_category": funnel_by_category(events),
        "product_performance": None,  # already written
        "top_products_by_category": top_products_by_category(perf),
        "sessions": sessions(events),
        "hourly_heatmap": hourly_heatmap(events),
        "customer_rfm": customer_rfm(events),
    }
    for name, df in tables.items():
        if df is not None:
            df.write.mode("overwrite").parquet(str(GOLD_DIR / name))
        print(f"  {name:<26} {spark.read.parquet(str(GOLD_DIR / name)).count():>10,} rows")
