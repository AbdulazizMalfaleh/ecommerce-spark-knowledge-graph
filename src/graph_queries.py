"""Reason over the knowledge graph: every answer comes with the path that explains it.

    python -m src.graph_queries recommend                 # picks a user automatically
    python -m src.graph_queries recommend --user 512001234
    python -m src.graph_queries switching                 # which brands lose shoppers to which
    python -m src.graph_queries switching --category electronics
    python -m src.graph_queries product --id 1004856      # a product's neighbourhood
"""
import argparse

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from src.graph import load


def _user_products(nodes, edges, user_id: str, relation: str) -> DataFrame:
    """User -HAD-> Session -<relation>-> Product"""
    sessions = edges["HAD"].filter(F.col("src") == user_id).select(F.col("dst").alias("session"))
    return (edges[relation].join(sessions, F.col("src") == F.col("session"))
            .select(F.col("dst").alias("product")).distinct())


def _product_info(nodes, edges) -> DataFrame:
    """Product + its brand + category (following MADE_BY and BELONGS_TO)."""
    return (
        nodes["Product"]
        .join(edges["MADE_BY"].select(F.col("src").alias("id"), F.col("dst").alias("brand")), "id", "left")
        .join(edges["BELONGS_TO"].select(F.col("src").alias("id"), F.col("dst").alias("category")), "id", "left")
    )


def recommend(spark: SparkSession, user_id: str, k: int = 10) -> DataFrame:
    """Two graph paths, in order of strength:

    1. co-purchase:  User -> PURCHASED -> X -BOUGHT_TOGETHER-> Y
                     "N customers who bought X also bought Y"
    2. taxonomy:     User -> VIEWED/PURCHASED -> X -BELONGS_TO-> C <-BELONGS_TO- Y
                     "popular in category C, which you browsed"   (ranked by number of buyers)
    """
    nodes, edges = load(spark)
    bought = _user_products(nodes, edges, user_id, "PURCHASED")
    seen = bought.union(_user_products(nodes, edges, user_id, "VIEWED")).distinct()

    # path 1: co-purchase
    w = Window.partitionBy("dst").orderBy(F.desc("confidence"), F.desc("customers"))
    co = (
        edges["BOUGHT_TOGETHER"]
        .filter(F.col("lift") > 1)       # lift <= 1: the pair is just two popular items, not an association
        .join(bought, F.col("src") == F.col("product"))
        .join(bought.withColumnRenamed("product", "owned"), F.col("dst") == F.col("owned"), "left_anti")
        .withColumn("best", F.row_number().over(w))
        .groupBy(F.col("dst").alias("id"))
        .agg(F.sum("confidence").alias("score"),
             F.max_by(F.struct("src", "customers"), F.col("best") == 1).alias("why"))
        .select("id", F.col("score") + 1.0,   # +1 keeps co-purchase ahead of category picks
                F.format_string("%d customers who bought %s also bought this",
                                F.col("why.customers"), F.col("why.src")).alias("reason"))
        .toDF("id", "score", "reason")
    )

    # path 2: taxonomy fallback
    info = _product_info(nodes, edges)
    user_cats = seen.join(info, F.col("product") == F.col("id")).select("category").filter(F.col("category").isNotNull()).distinct()
    buyers = edges["PURCHASED"].groupBy(F.col("dst").alias("id")).agg(F.countDistinct("src").alias("buyers"))
    total_buyers = buyers.agg(F.max("buyers").alias("mx"))
    by_category = (
        info.join(user_cats, "category")
        .join(buyers, "id")
        .join(seen.withColumnRenamed("product", "id"), "id", "left_anti")
        .crossJoin(total_buyers)
        .select("id", (F.col("buyers") / F.col("mx")).alias("score"),
                F.format_string("popular in %s (%d buyers), which you browsed",
                                F.col("category"), F.col("buyers")).alias("reason"))
    )

    best = Window.partitionBy("id").orderBy(F.desc("score"))
    return (
        co.unionByName(by_category)
        .withColumn("r", F.row_number().over(best)).filter("r = 1").drop("r")
        .join(info, "id")
        .orderBy(F.desc("score"), F.desc("pagerank"))
        .select(F.col("id").alias("product_id"), "brand", "category", "current_price",
                F.round("score", 3).alias("score"), "reason")
        .limit(k)
    )


def pick_user(spark: SparkSession) -> str:
    """A user whose purchases have co-purchase links, so the demo shows both reasoning paths."""
    nodes, edges = load(spark)
    linked = edges["BOUGHT_TOGETHER"].select(F.col("src").alias("product")).distinct()
    row = (edges["PURCHASED"].join(linked, F.col("dst") == F.col("product"))
           .join(edges["HAD"].select(F.col("src").alias("user"), F.col("dst").alias("session")),
                 F.col("src") == F.col("session"))
           .groupBy("user").count().orderBy(F.desc("count"), "user").first())
    return row["user"]


def brand_switching(spark: SparkSession, category: str | None = None, top: int = 10) -> DataFrame:
    """Net flow between brand pairs: shoppers A lost to B minus shoppers B lost to A."""
    _, edges = load(spark)
    sw = edges["SWITCHED_TO"]
    if category:
        sw = sw.filter(F.col("category_main") == category)
    sw = sw.groupBy("src", "dst").agg(F.sum("sessions").alias("lost"))
    back = sw.select(F.col("src").alias("dst"), F.col("dst").alias("src"), F.col("lost").alias("won_back"))
    return (
        sw.join(back, ["src", "dst"], "left").fillna({"won_back": 0})
        .withColumn("net_loss", F.col("lost") - F.col("won_back"))
        .filter(F.col("net_loss") > 0)
        .orderBy(F.desc("net_loss"))
        .select(F.col("src").alias("brand"), F.col("dst").alias("loses_to"), "lost", "won_back", "net_loss")
        .limit(top)
    )


def product_neighbourhood(spark: SparkSession, product_id: str, top: int = 10) -> tuple[DataFrame, DataFrame]:
    nodes, edges = load(spark)
    info = _product_info(nodes, edges)
    me = info.filter(F.col("id") == product_id)
    together = (edges["BOUGHT_TOGETHER"].filter(F.col("src") == product_id)
                .join(info, F.col("dst") == F.col("id"))
                .orderBy(F.desc("confidence"))
                .select(F.col("dst").alias("product_id"), "brand", "category", "customers", "confidence", "lift")
                .limit(top))
    return me, together


def main() -> None:
    from src.spark_session import get_spark

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("recommend"); r.add_argument("--user"); r.add_argument("-k", type=int, default=10)
    s = sub.add_parser("switching"); s.add_argument("--category"); s.add_argument("--top", type=int, default=10)
    pr = sub.add_parser("product"); pr.add_argument("--id", required=True)
    a = p.parse_args()

    spark = get_spark("graph-queries")
    if a.cmd == "recommend":
        user = a.user or pick_user(spark)
        print(f"\nRecommendations for user {user}:")
        recommend(spark, user, a.k).show(a.k, truncate=False)
    elif a.cmd == "switching":
        print(f"\nBrand switching{' in ' + a.category if a.category else ''} (viewed brand, bought the other):")
        brand_switching(spark, a.category, a.top).show(a.top, truncate=False)
    else:
        me, together = product_neighbourhood(spark, a.id)
        me.show(truncate=False)
        print("Bought together with:")
        together.show(truncate=False)
    spark.stop()


if __name__ == "__main__":
    main()
