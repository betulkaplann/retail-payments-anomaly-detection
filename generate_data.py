"""
generate_data.py - create a SYNTHETIC dataset of retail card/Swish/cash transactions.

Real point-of-sale data is confidential, so this script simulates one year of
transactions for a small Swedish retail chain. All numbers are invented.

What is simulated:
  * 8 stores, about 60 cashiers, opening hours 09-21
  * payment methods: card, Swish, cash, gift card - Swish share grows and
    cash share falls during the year (a trend seen in Swedish payment data)
  * normal "exceptions": refunds, voids and discounts that honest staff make
  * FRAUD by a few cashiers, in three typical patterns:
      1. fake refunds  - refund often without receipt, often cash, round amounts
      2. sweethearting - very large discounts given to friends
      3. cash voids    - a cash sale is voided right after payment (cash kept)

Run:  python generate_data.py      -> writes data/transactions.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
N_STORES = 8
CASHIERS_PER_STORE = 7
FRAUD_CASHIERS = 6                     # how many dishonest cashiers
START, END = "2025-10-01", "2026-09-30"
OUT_FILE = Path(__file__).resolve().parent / "data" / "transactions.csv"


def payment_mix(day_share):
    """Probabilities of each payment method; day_share goes 0 -> 1 over the year."""
    swish = 0.04 + 0.06 * day_share            # 4% -> 10%
    cash = 0.10 - 0.04 * day_share             # 10% -> 6%
    gift = 0.03
    card = 1 - swish - cash - gift
    return ["card", "swish", "cash", "gift_card"], [card, swish, cash, gift]


def generate(seed=SEED):
    rng = np.random.default_rng(seed)
    days = pd.date_range(START, END, freq="D")

    # ---- Stores and cashiers ------------------------------------------------
    cashiers = pd.DataFrame({
        "cashier_id": [f"C{s:02d}{c}" for s in range(1, N_STORES + 1)
                       for c in "ABCDEFG"[:CASHIERS_PER_STORE]],
        "store_id": [f"S{s:02d}" for s in range(1, N_STORES + 1)
                     for _ in range(CASHIERS_PER_STORE)],
    })
    fraudsters = set(rng.choice(cashiers["cashier_id"], FRAUD_CASHIERS, replace=False))
    # each fraudster prefers one pattern
    patterns = ["fake_refund", "sweethearting", "cash_void"]
    style = {c: patterns[i % 3] for i, c in enumerate(sorted(fraudsters))}

    # store size: busier stores have more customers per day
    store_traffic = dict(zip(cashiers["store_id"].unique(),
                             rng.integers(70, 140, N_STORES)))

    rows = []
    for d_idx, day in enumerate(days):
        share = d_idx / (len(days) - 1)
        methods, probs = payment_mix(share)
        weekend = day.dayofweek >= 5
        december = day.month == 12
        for store, base in store_traffic.items():
            n = rng.poisson(base * (1.3 if weekend else 1.0) * (1.4 if december else 1.0))
            staff = cashiers.loc[cashiers.store_id == store, "cashier_id"].to_numpy()
            # busier around lunch and after work: mixture of two peaks
            hours = np.clip(np.where(rng.random(n) < 0.45,
                                     rng.normal(12.5, 1.5, n),
                                     rng.normal(17.5, 1.8, n)), 9, 20.99)
            for h in hours:
                ts = day + pd.Timedelta(hours=float(h))
                cashier = rng.choice(staff)
                amount = round(float(rng.lognormal(5.3, 0.8)), 2)     # median ~200 SEK
                method = rng.choice(methods, p=probs)
                rows.append([ts, store, cashier, "sale", method, amount, 0.0,
                             np.nan, np.nan, 0, ""])

                # Honest exceptions -------------------------------------------
                r = rng.random()
                if r < 0.020:                                         # refund
                    rows.append([ts + pd.Timedelta(minutes=int(rng.integers(5, 300))),
                                 store, cashier, "refund", method,
                                 round(amount * rng.uniform(0.3, 1), 2), 0.0,
                                 int(rng.random() < 0.88), np.nan, 0, ""])
                elif r < 0.032:                                       # void (scan error)
                    rows.append([ts + pd.Timedelta(minutes=1), store, cashier, "void",
                                 method, amount, 0.0, np.nan,
                                 float(rng.integers(0, 4)), 0, ""])
                elif r < 0.070:                                       # normal discount
                    rows[-1][6] = float(rng.choice([0.1, 0.15, 0.2, 0.25, 0.3, 0.5],
                                                   p=[0.3, 0.2, 0.25, 0.1, 0.1, 0.05]))

            # Fraud -----------------------------------------------------------------
            for cashier in staff:
                if cashier not in fraudsters or rng.random() > 0.35:  # not every day
                    continue
                kind = style[cashier]
                for _ in range(rng.integers(1, 3)):
                    # fraudsters are careful: often late, but not always
                    hour = rng.uniform(19, 20.99) if rng.random() < 0.6 else rng.uniform(10, 19)
                    ts = day + pd.Timedelta(hours=float(hour))
                    if kind == "fake_refund":
                        # round amounts are typical, and old receipts are sometimes reused
                        amt = (float(rng.choice([300, 400, 500, 600, 800, 1000]))
                               if rng.random() < 0.7 else round(float(rng.lognormal(5.8, 0.4)), 2))
                        rows.append([ts, store, cashier, "refund",
                                     rng.choice(["cash", "card"], p=[0.6, 0.4]),
                                     amt, 0.0, int(rng.random() < 0.4), np.nan, 1, kind])
                    elif kind == "sweethearting":
                        amt = round(float(rng.lognormal(6.0, 0.5)), 2)
                        rows.append([ts, store, cashier, "sale",
                                     rng.choice(["card", "swish"]), amt,
                                     float(rng.choice([0.3, 0.4, 0.5, 0.7, 0.9, 1.0])),
                                     np.nan, np.nan, 1, kind])
                    else:  # cash_void
                        amt = round(float(rng.lognormal(5.8, 0.5)), 2)
                        rows.append([ts, store, cashier, "void", "cash", amt, 0.0,
                                     np.nan, float(rng.integers(3, 15)), 1, kind])

    df = pd.DataFrame(rows, columns=[
        "timestamp", "store_id", "cashier_id", "txn_type", "payment_method",
        "amount_sek", "discount_pct", "has_receipt", "minutes_since_sale",
        "is_fraud", "fraud_type"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "txn_id", [f"T{i:07d}" for i in range(len(df))])
    return df


if __name__ == "__main__":
    data = generate()
    OUT_FILE.parent.mkdir(exist_ok=True)
    data.to_csv(OUT_FILE, index=False)
    print(f"Saved {len(data):,} synthetic transactions to {OUT_FILE.name}")
    print(data["txn_type"].value_counts())
    print("Fraud rows:", int(data["is_fraud"].sum()))
