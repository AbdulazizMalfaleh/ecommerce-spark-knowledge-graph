"""KNOWLEDGE GRAPH layer: turn Silver/Gold data into nodes and edges that obey ontology.yaml.

    data/graph/nodes/<Entity>/     one Parquet table per entity   (id + properties)
    data/graph/edges/<RELATION>/   one Parquet table per relation (src, dst + properties)

Two kinds of edges:
  * observed  - read straight from events (VIEWED, PURCHASED, MADE_BY, ...)
  * derived   - knowledge inferred from behaviour (BOUGHT_TOGETHER, SWITCHED_TO)
Products also get a PageRank score from the BOUGHT_TOGETHER network.
"""
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from src.config import (BOUGHT_TOGETHER_MIN_CUSTOMERS, GOLD_DIR, GRAPH_DIR, MAX_BASKET_PRODUCTS,
                        PAGERANK_DAMPING, PAGERANK_ITERATIONS, SILVER_DIR)
from src.ontology import Ontology


def _id(col: str) -> F.Column:
    return F.col(col).cast("string")


# ============================== nodes ==============================

def user_nodes(events: DataFrame) -> DataFrame:
    return events.select(_id("user_id").alias("id")).distinct()


def session_nodes(sessions: DataFrame) -> DataFrame:
    # group by session only: one node per session even if the raw data links it to >1 user
    return sessions.groupBy("user_session").agg(
        F.min("session_start").alias("session_start"),
        F.sum("events").cast("long").alias("events"),
        F.max("converted").alias("converted"),
    ).withColumnRenamed("user_session", "id")


def product_nodes(products: DataFrame, pagerank: DataFrame) -> DataFrame:
    return (
        products.select(_id("product_id").alias("id"), "current_price")
        .join(pagerank, "id", "left")
        .fillna({"pagerank": 0.0})
    )


def brand_nodes(products: DataFrame) -> DataFrame:
    return products.filter(F.col("brand").isNotNull()).select(F.col("brand").alias("id")).distinct()


def category_nodes(products: DataFrame) -> DataFrame:
    """'electronics.audio.headphone' -> three nodes: electronics, electronics.audio, electronics.audio.headphone"""
    parts = F.split("category_code", r"\.")
    prefixes = F.transform(F.sequence(F.lit(1), F.size(parts)),
                           lambda i: F.array_join(F.slice(parts, 1, i), "."))
    return (
        products.filter(F.col("category_code").isNotNull())
        .select(F.explode(prefixes).alias("id"))
        .distinct()
        .withColumn("name", F.element_at(F.split("id", r"\."), -1))
        .withColumn("level", F.size(F.split("id", r"\.")))
    )


# ============================== observed edges ==============================

def had_edges(events: DataFrame) -> DataFrame:
    return events.select(_id("user_id").alias("src"), F.col("user_session").alias("dst")).distinct()


def interaction_edges(events: DataFrame, event_type: str) -> DataFrame:
    df = (events.filter(F.col("event_type") == event_type)
          .groupBy(F.col("user_session").alias("src"), _id("product_id").alias("dst")))
    if event_type == "purchase":
        return df.agg(F.count("*").alias("items"), F.round(F.sum("price"), 2).alias("amount"))
    return df.agg(F.count("*").alias("times"))


def made_by_edges(products: DataFrame) -> DataFrame:
    return products.filter(F.col("brand").isNotNull()).select(_id("product_id").alias("src"), F.col("brand").alias("dst"))


def belongs_to_edges(products: DataFrame) -> DataFrame:
    return (products.filter(F.col("category_code").isNotNull())
            .select(_id("product_id").alias("src"), F.col("category_code").alias("dst")))


def subcategory_edges(categories: DataFrame) -> DataFrame:
    parts = F.split("id", r"\.")
    return (categories.filter(F.col("level") > 1)
            .select(F.col("id").alias("src"),
                    F.array_join(F.slice(parts, 1, F.col("level") - 1), ".").alias("dst")))


# ============================== derived edges ==============================

def bought_together_edges(events: DataFrame, min_customers: int = BOUGHT_TOGETHER_MIN_CUSTOMERS,
                          max_basket: int = MAX_BASKET_PRODUCTS) -> DataFrame:
    """Market-basket association between products, at customer level (whole period).

        customers(A,B) = buyers of both A and B
        confidence(A->B) = customers(A,B) / buyers(A)          "if you bought A, chance you bought B"
        lift(A->B) = confidence(A->B) / (buyers(B) / all_buyers)   >1 means more than chance
    """
    baskets = events.filter(F.col("event_type") == "purchase").select("user_id", "product_id").distinct()
    basket_size = baskets.groupBy("user_id").count()
    baskets = baskets.join(basket_size.filter(F.col("count") <= max_basket).select("user_id"), "user_id")

    buyers = baskets.groupBy("product_id").agg(F.count("*").alias("buyers"))
    total = baskets.agg(F.countDistinct("user_id").alias("all_buyers"))

    a, b = baskets.alias("a"), baskets.alias("b")
    pairs = (
        a.join(b, (F.col("a.user_id") == F.col("b.user_id")) & (F.col("a.product_id") != F.col("b.product_id")))
        .groupBy(F.col("a.product_id").alias("src_pid"), F.col("b.product_id").alias("dst_pid"))
        .agg(F.count("*").alias("customers"))
        .filter(F.col("customers") >= min_customers)
    )
    return (
        pairs
        .join(buyers.withColumnRenamed("product_id", "src_pid").withColumnRenamed("buyers", "src_buyers"), "src_pid")
        .join(buyers.withColumnRenamed("product_id", "dst_pid").withColumnRenamed("buyers", "dst_buyers"), "dst_pid")
        .crossJoin(total)
        .withColumn("confidence", F.round(F.col("customers") / F.col("src_buyers"), 4))
        .withColumn("lift", F.round(F.col("confidence") / (F.col("dst_buyers") / F.col("all_buyers")), 2))
        .select(_id("src_pid").alias("src"), _id("dst_pid").alias("dst"), "customers", "confidence", "lift")
    )


def switched_to_edges(events: DataFrame) -> DataFrame:
    """Brand A was viewed but brand B was bought instead, in the same session and category."""
    keys = ["user_session", "category_main", "brand"]
    branded = events.filter(F.col("brand").isNotNull() & (F.col("category_main") != "unknown"))
    viewed = branded.filter(F.col("event_type") == "view").select(*keys).distinct()
    bought = branded.filter(F.col("event_type") == "purchase").select(*keys).distinct()

    viewed_not_bought = viewed.join(bought, keys, "left_anti")        # bought anyway? then not a switch
    return (
        viewed_not_bought.alias("v")
        .join(bought.alias("p"), ["user_session", "category_main"])
        .groupBy(F.col("v.brand").alias("src"), F.col("p.brand").alias("dst"), "category_main")
        .agg(F.count("*").alias("sessions"))
        .select("src", "dst", "sessions", "category_main")
    )


# ============================== PageRank ==============================

def pagerank(spark: SparkSession, product_ids: DataFrame, edges: DataFrame,
             iterations: int = PAGERANK_ITERATIONS, d: float = PAGERANK_DAMPING) -> DataFrame:
    """Weighted PageRank on the BOUGHT_TOGETHER network, written with plain DataFrames.

    A product is important if important products lead to it. Scores are scaled so
    the average product = 1.0 (2.5 = two and a half times as central as average).

    Iterative algorithms + lazy evaluation = a trap: without a checkpoint, iteration k's
    plan contains all k previous iterations and planning time explodes. localCheckpoint()
    materialises the ranks each round and cuts that lineage.
    """
    ids = product_ids.select("id").distinct().localCheckpoint()
    n = ids.count()
    w = (edges.withColumn("w", F.col("confidence") / F.sum("confidence").over(Window.partitionBy("src")))
              .select("src", "dst", "w").localCheckpoint())
    has_out = w.select(F.col("src").alias("id")).distinct()

    ranks = ids.withColumn("rank", F.lit(1.0 / n))
    for _ in range(iterations):
        incoming = (w.join(ranks, w.src == ranks.id)
                     .groupBy(F.col("dst").alias("id")).agg(F.sum(F.col("rank") * F.col("w")).alias("inc")))
        dangling = ranks.join(has_out, "id", "left_anti").agg(F.coalesce(F.sum("rank"), F.lit(0.0)).alias("dm"))
        ranks = (
            ids.join(incoming, "id", "left").fillna({"inc": 0.0})
            .crossJoin(dangling)
            .select("id", (F.lit((1 - d) / n) + F.lit(d) * (F.col("inc") + F.col("dm") / n)).alias("rank"))
            .localCheckpoint()
        )
    return ranks.select("id", F.round(F.col("rank") * n, 4).alias("pagerank"))


# ============================== build ==============================

def build(spark: SparkSession) -> tuple[dict[str, DataFrame], dict[str, DataFrame]]:
    events = spark.read.parquet(str(SILVER_DIR / "events"))
    products = spark.read.parquet(str(SILVER_DIR / "products"))
    sessions = spark.read.parquet(str(GOLD_DIR / "sessions"))

    bought_together = bought_together_edges(events).localCheckpoint()
    ranks = pagerank(spark, products.select(_id("product_id").alias("id")), bought_together)
    categories = category_nodes(products)

    nodes = {
        "User": user_nodes(events),
        "Session": session_nodes(sessions),
        "Product": product_nodes(products, ranks),
        "Brand": brand_nodes(products),
        "Category": categories,
    }
    edges = {
        "HAD": had_edges(events),
        "VIEWED": interaction_edges(events, "view"),
        "CARTED": interaction_edges(events, "cart"),
        "PURCHASED": interaction_edges(events, "purchase"),
        "MADE_BY": made_by_edges(products),
        "BELONGS_TO": belongs_to_edges(products),
        "SUBCATEGORY_OF": subcategory_edges(categories),
        "BOUGHT_TOGETHER": bought_together,
        "SWITCHED_TO": switched_to_edges(events),
    }
    return nodes, edges


def load(spark: SparkSession) -> tuple[dict[str, DataFrame], dict[str, DataFrame]]:
    onto = Ontology.load()
    nodes = {e: spark.read.parquet(str(GRAPH_DIR / "nodes" / e)) for e in onto.entities}
    edges = {r: spark.read.parquet(str(GRAPH_DIR / "edges" / r)) for r in onto.relations}
    return nodes, edges


def run(spark: SparkSession) -> None:
    print("\n=== KNOWLEDGE GRAPH ===")
    onto = Ontology.load()
    print(f"  ontology: {len(onto.entities)} entities, {len(onto.relations)} relations")

    nodes, edges = build(spark)
    for name, df in nodes.items():
        df.write.mode("overwrite").parquet(str(GRAPH_DIR / "nodes" / name))
    for name, df in edges.items():
        df.write.mode("overwrite").parquet(str(GRAPH_DIR / "edges" / name))

    nodes, edges = load(spark)
    errors = onto.validate(nodes, edges)
    if errors:
        raise ValueError("Graph violates the ontology:\n  " + "\n  ".join(errors))
    print("  validation: graph obeys the ontology (columns, types, unique ids, no dangling edges)")

    for name, df in nodes.items():
        print(f"  node  {name:<16} {df.count():>10,}")
    for name, df in edges.items():
        tag = "  (derived)" if onto.relations[name].derived else ""
        print(f"  edge  {name:<16} {df.count():>10,}{tag}")

    from src import graph_viz            # imported here: graph_viz itself depends on this module
    graph_viz.build(spark)
