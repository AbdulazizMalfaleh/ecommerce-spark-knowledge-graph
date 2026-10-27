"""Generate synthetic e-commerce events in the exact REES46 / Kaggle CSV format.

Simulates realistic shopping behaviour so every analysis has signal to find:
  * users have a favourite category and a long-tail activity level
  * sessions: a user browses several products in one category
  * funnel: view -> cart (~7%) -> purchase (~45% of carts), each step a few seconds/minutes later
  * traffic peaks in the evening; occasional discounted prices
It also injects "dirty" records (duplicates, bad prices, broken timestamps, unknown
event types, missing sessions) so the Silver layer has real cleaning to do.

Usage:
    python -m src.generate_sample_data                       # ~1M events
    python -m src.generate_sample_data --events 20000000     # bigger, for performance tests
"""
import argparse

import numpy as np
import pandas as pd

from src.config import RAW_DIR

# (category_code, relative traffic, median price, brands)
CATEGORIES = [
    ("electronics.smartphone",           30, 250, ["samsung", "apple", "xiaomi", "huawei", "oppo"]),
    ("electronics.audio.headphone",       8,  60, ["sony", "jbl", "apple", "xiaomi"]),
    ("electronics.video.tv",              6, 450, ["samsung", "lg", "sony"]),
    ("computers.notebook",                7, 700, ["lenovo", "hp", "asus", "acer", "apple"]),
    ("appliances.kitchen.refrigerators",  5, 500, ["samsung", "lg", "bosch", "beko"]),
    ("appliances.environment.vacuum",     6, 150, ["samsung", "dyson", "philips"]),
    ("apparel.shoes",                     9,  80, ["nike", "adidas", "puma"]),
    ("furniture.living_room.sofa",        3, 400, ["ikea", "hoff"]),
    ("kids.toys",                         5,  30, ["lego", "hasbro", "mattel"]),
    ("sport.bicycle",                     3, 300, ["stels", "forward"]),
    ("construction.tools.drill",          4, 100, ["bosch", "makita"]),
    ("accessories.bag",                   4,  40, ["samsonite", "xiaomi"]),
]
# share of traffic per hour of day (UTC): quiet nights, evening peak
HOUR_WEIGHTS = np.array([1, 1, 1, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6, 7, 8, 9, 10, 10, 9, 7, 4, 2], float)


def _session_ids(rng: np.random.Generator, n: int) -> np.ndarray:
    hi, lo = rng.integers(0, 2**63, n), rng.integers(0, 2**63, n)
    return np.array([f"{h:016x}{l:016x}" for h, l in zip(hi, lo)])


def generate(n_events: int, n_products: int, n_users: int, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    n_cat = len(CATEGORIES)
    cat_p = np.array([c[1] for c in CATEGORIES], float)
    cat_p /= cat_p.sum()

    # ---------------- products ----------------
    p_cat = rng.choice(n_cat, n_products, p=cat_p)
    p_brand = np.array([rng.choice(CATEGORIES[c][3]) for c in p_cat], dtype=object)
    p_brand[rng.random(n_products) < 0.10] = None                       # missing brand
    p_price = np.round([CATEGORIES[c][2] * rng.lognormal(0, 0.5) for c in p_cat], 2)
    p_code = np.array([CATEGORIES[c][0] for c in p_cat], dtype=object)
    p_code[rng.random(n_products) < 0.20] = None                        # missing category_code (common in the real data)
    p_pop = 1.0 / np.arange(1, n_products + 1) ** 0.9
    rng.shuffle(p_pop)
    p_appeal = rng.lognormal(0, 0.4, n_products)                        # how well a product converts
    cat_products = [np.flatnonzero(p_cat == c) for c in range(n_cat)]

    # ---------------- users & sessions ----------------
    u_activity = 1.0 / np.arange(1, n_users + 1) ** 0.7
    rng.shuffle(u_activity)
    u_activity /= u_activity.sum()
    u_fav = rng.choice(n_cat, n_users, p=cat_p)

    n_sessions = int(n_events / 4.6)
    s_user = rng.choice(n_users, n_sessions, p=u_activity)
    s_cat = np.where(rng.random(n_sessions) < 0.6, u_fav[s_user], rng.choice(n_cat, n_sessions, p=cat_p))
    day = rng.integers(0, 31, n_sessions)
    hour = rng.choice(24, n_sessions, p=HOUR_WEIGHTS / HOUR_WEIGHTS.sum())
    s_start = (np.datetime64("2019-10-01T00:00:00")
               + (day * 86400 + hour * 3600 + rng.integers(0, 3600, n_sessions)).astype("timedelta64[s]"))
    s_views = 1 + rng.geometric(0.27, n_sessions)

    # ---------------- view events ----------------
    v_sess = np.repeat(np.arange(n_sessions), s_views)
    v_prod = np.empty(len(v_sess), dtype=np.int64)
    v_cat = s_cat[v_sess]
    for c in range(n_cat):
        mask = v_cat == c
        prods = cat_products[c]
        w = p_pop[prods] / p_pop[prods].sum()
        v_prod[mask] = rng.choice(prods, mask.sum(), p=w)
    gaps = rng.exponential(45, len(v_sess)).astype(int)
    offset = pd.Series(gaps).groupby(v_sess).cumsum().to_numpy()
    v_time = s_start[v_sess] + offset.astype("timedelta64[s]")

    # ---------------- funnel: cart -> purchase ----------------
    carted = rng.random(len(v_sess)) < np.clip(0.07 * p_appeal[v_prod], 0, 0.5)
    c_idx = np.flatnonzero(carted)
    c_time = v_time[c_idx] + rng.exponential(30, len(c_idx)).astype("timedelta64[s]")
    bought = rng.random(len(c_idx)) < 0.45
    b_idx = c_idx[bought]
    b_time = c_time[bought] + rng.exponential(120, len(b_idx)).astype("timedelta64[s]")

    sess_ids = _session_ids(rng, n_sessions)

    def frame(idx, times, etype):
        prod = v_prod[idx]
        price = p_price[prod] * np.where(rng.random(len(idx)) < 0.05, 0.9, 1.0)   # 5% discounted
        return pd.DataFrame({
            "event_time": times,
            "event_type": etype,
            "product_id": 1_000_000 + prod,
            "category_id": 2_053_013_550_000_000_000 + p_cat[prod] * 1_000_003,
            "category_code": p_code[prod],
            "brand": p_brand[prod],
            "price": np.round(price, 2),
            "user_id": 512_000_000 + s_user[v_sess[idx]],
            "user_session": sess_ids[v_sess[idx]],
        })

    events = pd.concat([
        frame(np.arange(len(v_sess)), v_time, "view"),
        frame(c_idx, c_time, "cart"),
        frame(b_idx, b_time, "purchase"),
    ], ignore_index=True)
    events = events[events["event_time"] < np.datetime64("2019-11-01")]   # keep the month clean
    events["event_time"] =events["event_time"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    # ---------------- dirty records ----------------
    n_bad = max(10, len(events) // 5000)
    dupes = events.sample(frac=0.005, random_state=seed)
    broken = events.sample(n=n_bad * 4, random_state=seed + 1).copy()
    q = n_bad
    broken.iloc[:q, broken.columns.get_loc("price")] = -1.0                       # bad price
    broken.iloc[q:2 * q, broken.columns.get_loc("event_time")] = "not-a-date"     # broken timestamp
    broken.iloc[2 * q:3 * q, broken.columns.get_loc("event_type")] = "unknown"    # bad event type
    broken.iloc[3 * q:, broken.columns.get_loc("user_session")] = None            # missing session
    events = pd.concat([events, dupes, broken], ignore_index=True).sort_values("event_time", kind="stable")

    out = RAW_DIR / "events-sample.csv"
    events.to_csv(out, index=False)
    counts = events["event_type"].value_counts().to_dict()
    print(f"Wrote {len(events):,} events to {out}")
    print(f"  {counts}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--events", type=int, default=1_000_000)
    p.add_argument("--products", type=int, default=20_000)
    p.add_argument("--users", type=int, default=150_000)
    a = p.parse_args()
    generate(a.events, a.products, a.users)
