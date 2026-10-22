-- exception_features.sql
-- Builds the model features for every exception transaction (refund, void,
-- or sale with a discount of at least 20%). Same logic as add_features() in
-- detection.py, written in SQL (DuckDB dialect, close to PostgreSQL).
-- Input table: transactions  (one row per transaction)

WITH exceptions AS (
    -- 1. Keep only refunds, voids and discounted sales
    SELECT *
    FROM transactions
    WHERE txn_type <> 'sale' OR discount_pct >= 0.2
),

txn_features AS (
    -- 2. Transaction-level features
    SELECT
        txn_id,
        store_id,
        cashier_id,
        is_fraud,
        -- value at risk: for a discounted sale, the value given away
        CASE WHEN txn_type = 'sale' THEN amount_sek * discount_pct
             ELSE amount_sek END                                   AS value_at_risk,
        HOUR(timestamp) + MINUTE(timestamp) / 60.0                 AS hour,
        CASE WHEN HOUR(timestamp) >= 19 THEN 1 ELSE 0 END          AS late_hour,
        CASE WHEN payment_method = 'cash' THEN 1 ELSE 0 END        AS is_cash,
        CASE WHEN txn_type = 'refund' AND has_receipt = 0
             THEN 1 ELSE 0 END                                     AS no_receipt,
        discount_pct,
        CASE WHEN amount_sek % 50 = 0 THEN 1 ELSE 0 END            AS round_amount,
        CASE WHEN txn_type = 'refund' THEN 1 ELSE 0 END            AS is_refund,
        CASE WHEN txn_type = 'void' THEN 1 ELSE 0 END              AS is_void,
        COALESCE(minutes_since_sale, 0)                            AS minutes_since_sale
    FROM exceptions
),

sales_per_cashier AS (
    SELECT cashier_id, COUNT(*) AS n_sales
    FROM transactions
    WHERE txn_type = 'sale'
    GROUP BY cashier_id
),

cashier_stats AS (
    -- 3. Cashier behaviour: exceptions per 100 sales, share in the last 2 hours
    SELECT
        f.cashier_id,
        f.store_id,
        COUNT(*) * 100.0 / s.n_sales   AS exception_rate,
        AVG(f.late_hour)               AS late_share
    FROM txn_features f
    JOIN sales_per_cashier s USING (cashier_id)
    GROUP BY f.cashier_id, f.store_id, s.n_sales
),

cashier_z AS (
    -- 4. Compare each cashier with colleagues in the SAME store (z-score),
    --    using window functions over the store
    SELECT
        cashier_id,
        (exception_rate - AVG(exception_rate) OVER w) / STDDEV_SAMP(exception_rate) OVER w
            AS cashier_exception_rate_z,
        (late_share - AVG(late_share) OVER w) / STDDEV_SAMP(late_share) OVER w
            AS cashier_late_share_z
    FROM cashier_stats
    WINDOW w AS (PARTITION BY store_id)
)

SELECT
    f.*,
    LN(1 + f.value_at_risk) AS log_amount,
    z.cashier_exception_rate_z,
    z.cashier_late_share_z
FROM txn_features f
JOIN cashier_z z USING (cashier_id)
ORDER BY f.txn_id;
