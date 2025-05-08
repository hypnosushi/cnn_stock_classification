import pandas as pd
import numpy as np

df = pd.read_csv("oct2024-apr2025.csv")

"""
Columns: date, minute, low, high, open, close, volume, avg_trade_size, trade_count, avg_bid,avg_ask, avg_mid,avg_spread, avg_imbalance

Considerations for features:
- Moving averages
- Returns
- Time-based features
- VWAP

Labels: future return over next 5 minutes using threshold and closing price
- 1 = strong upward movement 
- 0 = no upward movement
- -1 = strong downward movement
"""

# 5 minute forward return
df['future_close'] = df['close'].shift(-5)
df['future_return'] = (df['future_close'] / df['close']) - 1


# Want to have a balanced dataset but 0.00025 is too small of a change and may label random noise as a movement
# Instead of using a fixed threshold, we'll use a dynamic threshold based on the rolling standard deviation of the returns
def label_stddev(threshold):
    df['return_1min'] = df['close'].pct_change()
    df['rolling_vol'] = df['return_1min'].rolling(window=20).std()

    df['dynamic_thresh'] = threshold * df['rolling_vol']

    def label_row(row):
        if pd.isna(row['future_return']) or pd.isna(row['dynamic_thresh']):
            return np.nan 
        if row['future_return'] > row['dynamic_thresh']:
            return 1
        elif row['future_return'] < -row['dynamic_thresh']:
            return -1
        else:
            return 0

    df['label'] = df.apply(label_row, axis=1)
    return df

def label_static(threshold):
    def label_return(r):
        if r > threshold:
            return 1     
        elif r < -threshold:
            return -1   
        else:
            return 0   

    df['label'] = df['future_return'].apply(label_return)
    return df

# df = label_static(0.0005)
df = label_stddev(1)
df.dropna(subset=['label'], inplace=True)
df['label'] = df['label'].astype(int)

# Checking imbalance of labels
print(df['label'].value_counts(normalize=True))

df.to_csv("st.001labeled_oct2024-apr2025.csv", index=False)

"""
.0005 static
 0    0.575476
 1    0.212491
-1    0.212033

1 stdev 
 0    0.377187
 1    0.317691
-1    0.305121
"""

