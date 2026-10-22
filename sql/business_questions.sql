-- business_questions.sql
-- Three short queries a loss-prevention or payments team might ask.
-- Each query starts with its own "-- name:" line, so the notebook can run them one by one.

-- name: monthly_payment_mix
-- Share of sales value by payment method, per month
SELECT
    DATE_TRUNC('month', timestamp)                         AS month,
    payment_method,
    ROUND(100.0 * SUM(amount_sek)
          / SUM(SUM(amount_sek)) OVER (PARTITION BY DATE_TRUNC('month', timestamp)), 1)
                                                           AS share_pct
FROM transactions
WHERE txn_type = 'sale'
GROUP BY 1, 2
ORDER BY 1, 2;

-- name: refunds_without_receipt_by_store
-- Refunds without receipt: count, value and share of all refunds, per store
SELECT
    store_id,
    COUNT(*) FILTER (WHERE has_receipt = 0)                         AS refunds_no_receipt,
    ROUND(SUM(amount_sek) FILTER (WHERE has_receipt = 0), 0)        AS value_sek,
    ROUND(100.0 * COUNT(*) FILTER (WHERE has_receipt = 0) / COUNT(*), 1) AS pct_of_refunds
FROM transactions
WHERE txn_type = 'refund'
GROUP BY store_id
ORDER BY pct_of_refunds DESC;

-- name: top_cashier_per_store
-- The cashier with the highest exception rate in each store (window ranking)
WITH rates AS (
    SELECT
        store_id,
        cashier_id,
        COUNT(*) FILTER (WHERE txn_type <> 'sale' OR discount_pct >= 0.2) * 100.0
            / COUNT(*) FILTER (WHERE txn_type = 'sale')            AS exceptions_per_100_sales
    FROM transactions
    GROUP BY store_id, cashier_id
)
SELECT *
FROM (
    SELECT *, RANK() OVER (PARTITION BY store_id
                           ORDER BY exceptions_per_100_sales DESC) AS rank_in_store
    FROM rates
)
WHERE rank_in_store = 1
ORDER BY exceptions_per_100_sales DESC;
