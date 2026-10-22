"""
detection.py - find suspicious refunds, voids and discounts.

Steps:
    1. exceptions()       -> keep only "exception" transactions (refunds, voids,
                             discounted sales) - this is where fraud hides
    2. add_features()     -> describe each exception with numbers a model can use
    3. rule_flags()       -> simple business rules (the baseline)
    4. isolation_forest() -> unsupervised machine learning anomaly score
    5. savings_curve()    -> money saved after paying for investigations
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "log_amount", "hour", "late_hour", "is_cash", "no_receipt",
    "discount_pct", "round_amount", "is_refund", "is_void", "minutes_since_sale",
    "cashier_exception_rate_z", "cashier_late_share_z",
]


# ---------------------------------------------------------------------------
# 1. Exceptions
# ---------------------------------------------------------------------------
def exceptions(df, min_discount=0.2):
    """Refunds, voids and sales with a discount of at least min_discount."""
    mask = (df["txn_type"] != "sale") | (df["discount_pct"] >= min_discount)
    return df.loc[mask].copy()


# ---------------------------------------------------------------------------
# 2. Features
# ---------------------------------------------------------------------------
def add_features(exc, all_txns):
    """Add model features to the exceptions table.

    Two kinds of features:
      * transaction level - amount, time of day, payment method, receipt ...
      * cashier level     - does this cashier make MORE exceptions than
                            colleagues in the same store? (z-score)
    """
    exc = exc.copy()
    ts = pd.to_datetime(exc["timestamp"])

    # -- transaction level ----------------------------------------------------
    # For discounted sales, the value given away matters, not the price
    value_at_risk = np.where(exc["txn_type"] == "sale",
                             exc["amount_sek"] * exc["discount_pct"],
                             exc["amount_sek"])
    exc["value_at_risk"] = value_at_risk
    exc["log_amount"] = np.log1p(value_at_risk)
    exc["hour"] = ts.dt.hour + ts.dt.minute / 60
    exc["late_hour"] = (exc["hour"] >= 19).astype(int)        # last 2 hours
    exc["is_cash"] = (exc["payment_method"] == "cash").astype(int)
    exc["no_receipt"] = ((exc["txn_type"] == "refund")
                         & (exc["has_receipt"] == 0)).astype(int)
    exc["round_amount"] = (exc["amount_sek"] % 50 == 0).astype(int)
    exc["is_refund"] = (exc["txn_type"] == "refund").astype(int)
    exc["is_void"] = (exc["txn_type"] == "void").astype(int)
    exc["minutes_since_sale"] = exc["minutes_since_sale"].fillna(0)

    # -- cashier level --------------------------------------------------------
    sales = all_txns[all_txns["txn_type"] == "sale"].groupby("cashier_id").size()
    n_exc = exc.groupby("cashier_id").size()
    late = exc.groupby("cashier_id")["late_hour"].mean()
    cashier = pd.DataFrame({"rate": n_exc / sales * 100, "late": late})
    cashier["store_id"] = all_txns.drop_duplicates("cashier_id").set_index(
        "cashier_id")["store_id"]

    # z-score within store: (cashier value - store average) / store std
    for col, name in [("rate", "cashier_exception_rate_z"),
                      ("late", "cashier_late_share_z")]:
        grp = cashier.groupby("store_id")[col]
        cashier[name] = (cashier[col] - grp.transform("mean")) / grp.transform("std")
    exc = exc.merge(cashier[["cashier_exception_rate_z", "cashier_late_share_z"]],
                    left_on="cashier_id", right_index=True, how="left")
    return exc


# ---------------------------------------------------------------------------
# 3. Rule-based baseline
# ---------------------------------------------------------------------------
def rule_flags(exc):
    """Typical loss-prevention rules. Returns 1 = flagged, 0 = not flagged."""
    rule_refund = exc["no_receipt"] == 1                       # refund, no receipt
    rule_discount = exc["discount_pct"] >= 0.5                  # 50%+ discount
    rule_void = (exc["is_void"] == 1) & (exc["is_cash"] == 1) & (exc["late_hour"] == 1)
    return (rule_refund | rule_discount | rule_void).astype(int)


# ---------------------------------------------------------------------------
# 4. Isolation Forest
# ---------------------------------------------------------------------------
def isolation_forest(exc, seed=0):
    """Anomaly score: higher = more unusual.

    Isolation Forest builds many random decision trees. Unusual points are
    separated ("isolated") from the rest after only a few splits, so they get
    a high score. It needs NO fraud labels, which suits real life where most
    fraud is never confirmed.
    """
    X = StandardScaler().fit_transform(exc[FEATURES].fillna(0))
    model = IsolationForest(n_estimators=300, random_state=seed)
    model.fit(X)
    return -model.score_samples(X)          # flip sign so high = suspicious


# ---------------------------------------------------------------------------
# 5. Cost-sensitive evaluation
# ---------------------------------------------------------------------------
def savings_curve(exc, score, review_cost=150.0, recovery_rate=1.0):
    """Net savings if the team reviews the top-k cases, for every k.

    review_cost   : cost of investigating one case in SEK (staff time)
    recovery_rate : share of the fraud amount saved when fraud is found
    net savings(k) = recovered fraud value in top k - review_cost x k
    """
    order = np.argsort(-np.asarray(score))
    ranked = exc.iloc[order]
    caught = (ranked["is_fraud"] * ranked["value_at_risk"] * recovery_rate).cumsum()
    k = np.arange(1, len(ranked) + 1)
    return pd.DataFrame({
        "k": k,
        "fraud_found": ranked["is_fraud"].cumsum().to_numpy(),
        "value_recovered": caught.to_numpy(),
        "review_cost": k * review_cost,
        "net_savings": caught.to_numpy() - k * review_cost,
    })


def evaluate_flags(exc, flags, review_cost=150.0):
    """Precision, recall and net savings for a yes/no flag (e.g. the rules)."""
    flags = np.asarray(flags).astype(bool)
    tp = int(exc.loc[flags, "is_fraud"].sum())
    value = float((exc.loc[flags, "is_fraud"] * exc.loc[flags, "value_at_risk"]).sum())
    return pd.Series({
        "cases reviewed": int(flags.sum()),
        "fraud found": tp,
        "precision": tp / max(flags.sum(), 1),
        "recall": tp / exc["is_fraud"].sum(),
        "value recovered (SEK)": value,
        "review cost (SEK)": flags.sum() * review_cost,
        "net savings (SEK)": value - flags.sum() * review_cost,
    })
