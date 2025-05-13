import pandas as pd
import numpy as np
# import tensforflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

def plot_features_over_time(
    df,
    date_col='date',
    time_col='minute',
    features=None,
    normalize=False,
    group_size=4,
    title_prefix="Features"
):
    """
    Plots time series features in grouped subplots.

    Args:
        df (pd.DataFrame): Your raw data with date and time columns.
        date_col (str): Name of the date column.
        time_col (str): Name of the minute/time column.
        features (list): List of features to plot. If None, plots all.
        normalize (bool): If True, standardizes features.
        group_size (int): How many features to plot per figure.
        title_prefix (str): Title prefix for each plot group.
    """
    # Combine date and time into datetime
    date_time = pd.to_datetime(df[date_col] + ' ' + df[time_col], format='%Y-%m-%d %H:%M:%S')
    df = df.copy()
    df.index = date_time
    df = df.drop(columns=[date_col, time_col])

    # Select features
    if features is None:
        features = df.columns.tolist()

    # Normalize if requested
    if normalize:
        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])

    # Group features and plot
    for i in range(0, len(features), group_size):
        group = features[i:i + group_size]
        ax = df[group].plot(subplots=True, figsize=(12, 2.5 * len(group)), title=f"{title_prefix} {i // group_size + 1}")
        for a in ax:
            a.set_xlabel("Time")
        plt.tight_layout()
        plt.show()

plot_features_over_time(
    df,
    features=['close', 'volume', 'avg_spread', 'label'],
    normalize=False
)

# plot_features_over_time(
#     df,
#     normalize=True,
#     group_size=5
# )
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
    print(df.describe().transpose())
# df.to_csv("st.001labeled_oct2024-apr2025.csv", index=False)
spike_rows = df[df['volume'] > 8_000_000]  # or df['trade_count'] > threshold
print(spike_rows[['volume', 'trade_count']])
print(spike_rows.index)

def plot_feature_distributions(df, features=None, bins=50, kde=True, cols=3, figsize=(15, 10)):
    """
    Plots the distribution of each numeric feature in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame with your features.
        features (list, optional): List of columns to plot. If None, all numeric columns are used.
        bins (int): Number of bins for histograms.
        kde (bool): Whether to include a KDE (smoothed density) plot.
        cols (int): Number of columns per row in the subplot grid.
        figsize (tuple): Size of the entire figure.
    """
    if features is None:
        features = df.select_dtypes(include=np.number).columns.tolist()

    rows = int(np.ceil(len(features) / cols))
    plt.figure(figsize=figsize)

    for i, col in enumerate(features):
        plt.subplot(rows, cols, i + 1)
        sns.histplot(df[col].dropna(), bins=bins, kde=kde, color='steelblue')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

plot_feature_distributions(df)

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