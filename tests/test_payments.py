"""Tests for the payments anomaly-detection project. Run with:  pytest -q"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import detection as dt       # noqa: E402
import generate_data         # noqa: E402


@pytest.fixture(scope="module")
def transactions():
    """Six weeks of synthetic data: enough for every pattern, fast to build."""
    generate_data.START, generate_data.END = "2025-10-01", "2025-11-10"
    return generate_data.generate(seed=3)


def test_generated_data_has_expected_columns(transactions):
    expected = {"txn_id", "timestamp", "store_id", "cashier_id", "txn_type", "payment_method",
                "amount_sek", "discount_pct", "has_receipt", "minutes_since_sale", "is_fraud"}
    assert expected <= set(transactions.columns)
    assert transactions["txn_id"].is_unique
    assert transactions["is_fraud"].sum() > 0


def test_sql_features_match_pandas(transactions):
    """The SQL in sql/exception_features.sql must give the same numbers as detection.py."""
    con = duckdb.connect()
    con.register("transactions", transactions)
    sql = con.sql((ROOT / "sql" / "exception_features.sql").read_text()).df()
    pdf = dt.add_features(dt.exceptions(transactions), transactions)
    pdf = pdf.sort_values("txn_id").reset_index(drop=True)
    assert len(sql) == len(pdf)
    for col in dt.FEATURES:
        np.testing.assert_allclose(sql[col].astype(float), pdf[col].astype(float), atol=1e-9)


def test_business_queries_run(transactions):
    con = duckdb.connect()
    con.register("transactions", transactions)
    text = (ROOT / "sql" / "business_questions.sql").read_text()
    blocks = text.split("\n-- name: ")[1:]
    assert len(blocks) == 3
    for block in blocks:
        result = con.sql("\n".join(block.splitlines()[1:])).df()
        assert len(result) > 0


def test_savings_curve_on_a_toy_example():
    exc = pd.DataFrame({"is_fraud": [1, 0, 1], "value_at_risk": [500.0, 100.0, 300.0]})
    curve = dt.savings_curve(exc, score=[0.9, 0.1, 0.5], review_cost=100)
    # reviewed in order: fraud 500, fraud 300, honest 100
    assert curve["net_savings"].tolist() == [400.0, 600.0, 500.0]
    assert curve["fraud_found"].tolist() == [1, 2, 2]


def test_rules_flag_refund_without_receipt():
    exc = pd.DataFrame({"no_receipt": [1, 0], "discount_pct": [0.0, 0.1],
                        "is_void": [0, 0], "is_cash": [0, 0], "late_hour": [0, 0]})
    assert dt.rule_flags(exc).tolist() == [1, 0]
