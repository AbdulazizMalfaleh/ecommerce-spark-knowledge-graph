"""Build an interactive HTML explorer of the knowledge graph.

    python -m src.graph_viz          -> data/graph/explorer.html  (open it in a browser)

Three views: the ontology itself, the brand-switching network, and the
co-purchase network of the most central products.
"""
import json

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

from src.config import GRAPH_DIR, PROJECT_ROOT
from src.graph import load
from src.graph_queries import brand_switching
from src.ontology import Ontology

TEMPLATE = PROJECT_ROOT / "src" / "templates" / "graph_explorer.html"
OUT = GRAPH_DIR / "explorer.html"


def collect_data(spark: SparkSession, top_products: int = 40, neighbours: int = 3) -> dict:
    onto = Ontology.load()
    nodes, edges = load(spark)

    counts = {
        "nodes": [[n, df.count()] for n, df in nodes.items()],
        "edges": [[r, df.count(), onto.relations[r].derived] for r, df in edges.items()],
    }
    ontology = {
        "entities": [{"name": e.name, "key": e.key, "description": e.description, "properties": e.properties}
                     for e in onto.entities.values()],
        "relations": [{"name": r.name, "from": r.source, "to": r.target, "derived": r.derived,
                       "description": r.description, "properties": r.properties}
                      for r in onto.relations.values()],
    }

    # ---- brands: size = items bought, colour = the category they sell most in ----
    cat_main = F.element_at(F.split("category", r"\."), 1)
    bought = (edges["PURCHASED"].select(F.col("dst").alias("pid"), "items")
              .join(edges["MADE_BY"].select(F.col("src").alias("pid"), F.col("dst").alias("brand")), "pid")
              .join(edges["BELONGS_TO"].select(F.col("src").alias("pid"), F.col("dst").alias("category")), "pid", "left")
              .withColumn("category", F.coalesce(cat_main, F.lit("unknown"))))
    brand_cat = bought.groupBy("brand", "category").agg(F.sum("items").alias("n"))
    brand_nodes = (brand_cat.groupBy("brand")
                   .agg(F.sum("n").alias("purchases"),
                        F.max_by("category", F.when(F.col("category") != "unknown", F.col("n")).otherwise(0)).alias("category"))
                   .collect())
    switch = brand_switching(spark, top=80).collect()

    # ---- products: top PageRank + their strongest associations ----
    bt = edges["BOUGHT_TOGETHER"].filter(F.col("lift") > 1)
    top = (nodes["Product"].join(bt.select(F.col("src").alias("id")).distinct(), "id")
           .orderBy(F.desc("pagerank")).limit(top_products).select("id"))
    w = Window.partitionBy("src").orderBy(F.desc("confidence"))
    picked = bt.join(top.withColumnRenamed("id", "src"), "src").withColumn("r", F.row_number().over(w)).filter(F.col("r") <= neighbours)
    ids = picked.select(F.col("src").alias("id")).union(picked.select(F.col("dst").alias("id"))).distinct()
    p_edges = picked.select("src", "dst", "customers", "confidence", "lift").collect()
    p_nodes = (nodes["Product"].join(ids, "id")
               .join(edges["MADE_BY"].select(F.col("src").alias("id"), F.col("dst").alias("brand")), "id", "left")
               .join(edges["BELONGS_TO"].select(F.col("src").alias("id"), F.col("dst").alias("category")), "id", "left")
               .withColumn("category_main", F.coalesce(F.element_at(F.split("category", r"\."), 1), F.lit("unknown")))
               .collect())

    return {
        "counts": counts,
        "ontology": ontology,
        "brands": {
            "nodes": [{"id": r.brand, "purchases": int(r.purchases), "category": r.category or "unknown"} for r in brand_nodes],
            "edges": [{"src": r.brand, "dst": r.loses_to, "lost": int(r.lost), "won_back": int(r.won_back),
                       "net_loss": int(r.net_loss)} for r in switch],
        },
        "products": {
            "nodes": [{"id": r.id, "brand": r.brand, "category": r.category, "category_main": r.category_main,
                       "price": r.current_price, "pagerank": r.pagerank} for r in p_nodes],
            "edges": [{"src": r.src, "dst": r.dst, "customers": int(r.customers),
                       "confidence": r.confidence, "lift": r.lift} for r in p_edges],
        },
    }


def render(data: dict) -> str:
    """The page body with the data embedded (no <html>/<head> wrapper)."""
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", payload)


def build(spark: SparkSession) -> None:
    body = render(collect_data(spark))
    page = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            f'</head><body>\n{body}\n</body></html>\n')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"  explorer -> {OUT}")


if __name__ == "__main__":
    from src.spark_session import get_spark
    s = get_spark("graph-viz")
    build(s)
    s.stop()
