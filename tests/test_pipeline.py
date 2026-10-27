"""Unit tests for the Silver and Gold logic.   Run:  pytest -q"""
from datetime import datetime

import pytest

from src.gold import customer_rfm, funnel_by_category, product_performance
from src.silver import build_products, clean_events

RAW_COLS = ("event_time string, event_type string, product_id long, category_id long, "
            "category_code string, brand string, price double, user_id long, user_session string")


def raw(spark, rows):
    return spark.createDataFrame(rows, RAW_COLS)


def test_clean_events_parses_both_time_formats_and_drops_bad_rows(spark):
    ok = ("2019-10-01 10:00:00 UTC", "view", 1, 9, "electronics.audio.headphone", "Sony ", 50.0, 7, "s1")
    df = raw(spark, [
        ok,
        ok,                                                                           # exact duplicate
        ("2019-10-01 10:05:00+00:00", "PURCHASE", 1, 9, None, None, 50.0, 7, "s1"),  # other time format
        ("not-a-date", "view", 1, 9, None, None, 50.0, 7, "s1"),
        ("2019-10-01 10:00:00 UTC", "unknown", 1, 9, None, None, 50.0, 7, "s1"),
        ("2019-10-01 10:00:00 UTC", "view", 1, 9, None, None, -1.0, 7, "s1"),
        ("2019-10-01 10:00:00 UTC", "view", 1, 9, None, None, 50.0, 7, None),
    ])
    out = sorted(clean_events(df).collect(), key=lambda r: r.event_time)

    assert len(out) == 2
    view, purchase = out
    assert view.event_time == datetime(2019, 10, 1, 10, 0, 0)
    assert view.category_main == "electronics" and view.category_sub == "headphone"
    assert view.brand == "sony"
    assert purchase.event_type == "purchase"
    assert purchase.event_time == datetime(2019, 10, 1, 10, 5, 0)
    assert purchase.category_main == "unknown"


def test_products_keep_latest_price(spark):
    events = clean_events(raw(spark, [
        ("2019-10-01 10:00:00 UTC", "view", 1, 9, "kids.toys", "lego", 30.0, 7, "s1"),
        ("2019-10-05 10:00:00 UTC", "view", 1, 9, "kids.toys", "lego", 27.0, 8, "s2"),
    ]))
    (p,) = build_products(events).collect()
    assert p.current_price == 27.0


def test_funnel_and_smoothed_conversion(spark):
    rows = []
    # product 1: 1 view, 1 purchase (100%) — product 2: 100 views, 10 purchases (10%)
    rows.append(("2019-10-01 10:00:00 UTC", "view", 1, 9, "kids.toys", "lego", 10.0, 1, "a"))
    rows.append(("2019-10-01 10:01:00 UTC", "purchase", 1, 9, "kids.toys", "lego", 10.0, 1, "a"))
    for i in range(100):
        rows.append((f"2019-10-02 10:{i // 60:02d}:{i % 60:02d} UTC", "view", 2, 9, "kids.toys", "lego", 10.0, 100 + i, f"s{i}"))
    for i in range(10):
        rows.append((f"2019-10-02 11:00:{i:02d} UTC", "purchase", 2, 9, "kids.toys", "lego", 10.0, 100 + i, f"s{i}"))
    events = clean_events(raw(spark, rows))

    funnel = funnel_by_category(events).first()
    assert (funnel.views, funnel.purchases, funnel.revenue) == (101, 11, 110.0)
    assert funnel.cart_to_purchase is None          # no carts -> null, not a divide-by-zero crash

    # baseline traffic so the global conversion rate is realistic (~2%), as in real data
    for i in range(1000):
        rows.append((f"2019-10-03 {i // 3600:02d}:{i // 60 % 60:02d}:{i % 60:02d} UTC", "view", 3, 9, "kids.toys", "lego", 10.0, 5000 + i, f"t{i}"))
    for i in range(20):
        rows.append((f"2019-10-03 05:00:{i:02d} UTC", "purchase", 3, 9, "kids.toys", "lego", 10.0, 5000 + i, f"t{i}"))
    events = clean_events(raw(spark, rows))

    perf = {r.product_id: r for r in product_performance(events, build_products(events), m=20).collect()}
    assert perf[1].conversion_rate == 1.0
    # smoothing must rank the well-proven product above the 1-view fluke
    assert perf[2].smoothed_cr > perf[1].smoothed_cr


def test_rfm_frequency_counts_orders_not_items(spark):
    events = clean_events(raw(spark, [
        # user 1: two items in ONE session = 1 order; user 2: three separate orders
        ("2019-10-10 10:00:00 UTC", "purchase", 1, 9, None, None, 10.0, 1, "a"),
        ("2019-10-10 10:01:00 UTC", "purchase", 2, 9, None, None, 20.0, 1, "a"),
        ("2019-10-01 10:00:00 UTC", "purchase", 1, 9, None, None, 10.0, 2, "b1"),
        ("2019-10-05 10:00:00 UTC", "purchase", 1, 9, None, None, 10.0, 2, "b2"),
        ("2019-10-09 10:00:00 UTC", "purchase", 1, 9, None, None, 10.0, 2, "b3"),
    ]))
    rfm = {r.user_id: r for r in customer_rfm(events).collect()}
    assert (rfm[1].frequency, rfm[1].monetary, rfm[1].recency_days) == (1, 30.0, 1)
    assert (rfm[2].frequency, rfm[2].f_score, rfm[2].recency_days) == (3, 3, 2)
