import wrds
import pandas as pd

# Connect to WRDS
db = wrds.Connection(wrds_username='mhu14')

# Choose your start date (past 2 months, adjust as needed)
start_date = "2025-02-01"

# SQL query to get data after a certain date
query = f"""
WITH trade_base AS (
    SELECT
        date,
        date_trunc('minute', time_m)::time AS minute,
        time_m,
        price,
        size,
        FIRST_VALUE(price) OVER (PARTITION BY date, date_trunc('minute', time_m) ORDER BY time_m) AS open,
        LAST_VALUE(price) OVER (PARTITION BY date, date_trunc('minute', time_m) ORDER BY time_m
                                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS close
    FROM taqm_2025.ctm_2025
    WHERE sym_root = 'SPY'
      AND date >= '{start_date}'
      AND time_m BETWEEN '09:30:00' AND '16:00:00'
),
trade_bars AS (
    SELECT
        date,
        minute,
        MIN(price) AS low,
        MAX(price) AS high,
        MAX(open) AS open,
        MAX(close) AS close,
        SUM(size) AS volume,
        AVG(size) AS avg_trade_size,
        COUNT(*) AS trade_count
    FROM trade_base
    GROUP BY date, minute
),
quote_bars AS (
    SELECT
        date,
        date_trunc('minute', time_m)::time AS minute,
        AVG(bid) AS avg_bid,
        AVG(ask) AS avg_ask,
        AVG((bid + ask)/2) AS avg_mid,
        AVG(ask - bid) AS avg_spread,
        AVG((bidsiz - asksiz) / NULLIF((bidsiz + asksiz), 0)) AS avg_imbalance
    FROM taqm_2025.cqm_2025
    WHERE sym_root = 'SPY'
      AND date >= '{start_date}'
      AND time_m BETWEEN '09:30:00' AND '16:00:00'
    GROUP BY date, minute
)

SELECT 
    t.*,
    q.avg_bid, q.avg_ask, q.avg_mid,
    q.avg_spread, q.avg_imbalance
FROM trade_bars t
LEFT JOIN quote_bars q
USING (date, minute)
ORDER BY date, minute
"""

# Run the query
df = db.raw_sql(query)

# Save to CSV
df.to_csv("spy_1min_from_feb2025.csv", index=False)
print("✅ Saved to spy_1min_from_feb2025.csv")

# Close connection
db.close()
