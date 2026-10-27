"""Tests for the knowledge-graph layer.   Run:  pytest -q"""
import pytest
from pyspark.sql import functions as F

from src.graph import (bought_together_edges, category_nodes, pagerank, subcategory_edges,
                       switched_to_edges)
from src.ontology import Ontology
from src.silver import clean_events
from tests.test_pipeline import raw


def ev(t, etype, pid, code, brand, user, session, price=10.0):
    return (f"2019-10-01 10:{t:02d}:00 UTC", etype, pid, 9, code, brand, price, user, session)


def test_category_taxonomy(spark):
    products = spark.createDataFrame(
        [(1, "electronics.audio.headphone"), (2, "electronics.smartphone"), (3, None)],
        "product_id long, category_code string")
    cats = {r.id: (r.name, r.level) for r in category_nodes(products).collect()}
    assert cats == {
        "electronics": ("electronics", 1),
        "electronics.audio": ("audio", 2),
        "electronics.audio.headphone": ("headphone", 3),
        "electronics.smartphone": ("smartphone", 2),
    }
    parents = {(r.src, r.dst) for r in subcategory_edges(category_nodes(products)).collect()}
    assert parents == {("electronics.audio.headphone", "electronics.audio"),
                       ("electronics.audio", "electronics"),
                       ("electronics.smartphone", "electronics")}


def test_bought_together_confidence_and_lift(spark):
    # 4 buyers: u1,u2 buy A+B; u3 buys A only; u4 buys C only
    rows = [ev(0, "purchase", 1, None, None, 1, "s1"), ev(1, "purchase", 2, None, None, 1, "s1"),
            ev(0, "purchase", 1, None, None, 2, "s2"), ev(1, "purchase", 2, None, None, 2, "s2"),
            ev(0, "purchase", 1, None, None, 3, "s3"),
            ev(0, "purchase", 3, None, None, 4, "s4")]
    edges = {(r.src, r.dst): r for r in bought_together_edges(clean_events(raw(spark, rows)), min_customers=2).collect()}
    assert set(edges) == {("1", "2"), ("2", "1")}
    a_to_b = edges[("1", "2")]
    assert a_to_b.customers == 2
    assert a_to_b.confidence == pytest.approx(2 / 3, abs=1e-4)          # 2 of A's 3 buyers bought B
    assert a_to_b.lift == pytest.approx((2 / 3) / (2 / 4), abs=0.01)     # vs B's base rate 2/4
    assert edges[("2", "1")].confidence == 1.0                           # every B buyer bought A


def test_switched_to_ignores_brand_bought_anyway(spark):
    rows = [
        # s1: viewed samsung + apple, bought apple  -> samsung SWITCHED_TO apple
        ev(0, "view", 1, "electronics.smartphone", "samsung", 1, "s1"),
        ev(1, "view", 2, "electronics.smartphone", "apple", 1, "s1"),
        ev(2, "purchase", 2, "electronics.smartphone", "apple", 1, "s1"),
        # s2: viewed samsung, bought samsung AND apple -> samsung was bought, no switch
        ev(0, "view", 1, "electronics.smartphone", "samsung", 2, "s2"),
        ev(1, "purchase", 1, "electronics.smartphone", "samsung", 2, "s2"),
        ev(2, "purchase", 2, "electronics.smartphone", "apple", 2, "s2"),
        # s3: viewed a sofa brand, bought a phone -> different category, no switch
        ev(0, "view", 5, "furniture.sofa", "ikea", 3, "s3"),
        ev(1, "purchase", 2, "electronics.smartphone", "apple", 3, "s3"),
    ]
    out = [(r.src, r.dst, r.sessions, r.category_main) for r in switched_to_edges(clean_events(raw(spark, rows))).collect()]
    assert out == [("samsung", "apple", 1, "electronics")]


def test_pagerank_conserves_mass_and_ranks_hub_first(spark):
    ids = spark.createDataFrame([(str(i),) for i in range(1, 6)], "id string")
    # everyone points at 1; 1 points at 2; 5 has no out-links (dangling)
    edges = spark.createDataFrame(
        [("2", "1", 1.0), ("3", "1", 1.0), ("4", "1", 1.0), ("1", "2", 1.0)],
        "src string, dst string, confidence double")
    ranks = {r.id: r.pagerank for r in pagerank(spark, ids, edges, iterations=30).collect()}
    assert sum(ranks.values()) == pytest.approx(5.0, abs=0.01)       # scaled so the average is 1.0
    assert max(ranks, key=ranks.get) == "1"
    assert ranks["2"] > ranks["3"]                                    # endorsed by the hub


def test_ontology_rejects_a_broken_graph(spark):
    onto = Ontology.load()
    good = {e: spark.createDataFrame([], ", ".join(f"{c} {t if t != 'long' else 'bigint'}"
                                                  for c, t in onto.node_columns(e).items()))
            for e in onto.entities}
    good["Product"] = spark.createDataFrame([("1", 9.9, 1.0)], "id string, current_price double, pagerank double")
    good["Brand"] = spark.createDataFrame([("apple",)], "id string")
    edges = {r: spark.createDataFrame([], ", ".join(f"{c} {t if t != 'long' else 'bigint'}"
                                                   for c, t in onto.edge_columns(r).items()))
             for r in onto.relations}
    assert onto.validate(good, edges) == []

    # MADE_BY pointing at a brand that doesn't exist -> dangling edge
    edges["MADE_BY"] = spark.createDataFrame([("1", "no-such-brand")], "src string, dst string")
    # a property the ontology doesn't declare
    good["Brand"] = good["Brand"].withColumn("country", F.lit("US"))
    errors = onto.validate(good, edges)
    assert any("MADE_BY" in e and "dst is not a Brand" in e for e in errors)
    assert any("node Brand: columns" in e for e in errors)
