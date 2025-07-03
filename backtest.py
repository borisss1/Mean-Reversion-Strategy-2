import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw()

print("Select your CSV file with OHLC data (must include columns: open, high, low, close, and a datetime column).")
file_path = askopenfilename(filetypes=[("CSV files", "*.csv")])

if not file_path:
    print("No file selected, exiting.")
    exit()

print(f'Loaded file "{file_path}"')
df = pd.read_csv(file_path)

datetime_col = 'date'
df[datetime_col] = pd.to_datetime(df[datetime_col])
df.set_index(datetime_col, inplace=True)
df.sort_index(inplace=True)

print(f"DataFrame shape: {df.shape}")
print(df.head())

def compute_ma_trend_score(df, ma_period=20, lookback=5, threshold=0.001):
    df = df.copy()
    
    df['ma'] = df['close'].rolling(window=ma_period).mean()

    df['ma_prev'] = df['ma'].shift(lookback)
    df['diff_pct'] = (df['ma'] - df['ma_prev']) / df['ma_prev']

    def score_func(x):
        if x > threshold:
            return 1
        elif x < -threshold:
            return -1
        else:
            return 0
    
    df['score'] = df['diff_pct'].apply(score_func)

    df['signal'] = df['score']

    df['daily_return'] = df['close'].pct_change()
    df['strategy_return'] = df['signal'].shift(1) * df['daily_return']

    df['cum_strategy_return'] = (1 + df['strategy_return']).cumprod()

    df['cum_max'] = df['cum_strategy_return'].cummax()
    df['drawdown'] = df['cum_strategy_return'] / df['cum_max'] - 1
    
    return df

strategy_df = compute_ma_trend_score(df)

plt.figure(figsize=(14,6))
(strategy_df['strategy_return'].cumsum().apply(np.exp)).plot(label='MA Trend Score Strategy', color='blue')
(strategy_df['daily_return'].cumsum().apply(np.exp)).plot(label='Buy & Hold', color='gray', alpha=0.5)
plt.title("Moving Average Trend Score vs Buy & Hold")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14,6))
strategy_df['drawdown'].plot(color='red', label='Strategy Drawdown')
plt.title("Strategy Drawdown")
plt.ylabel("Drawdown")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(14,6))
strategy_df['score'].plot(label='Trend Score', color='blue')
plt.axhline(1, color='green', linestyle='--', label='Uptrend')
plt.axhline(-1, color='red', linestyle='--', label='Downtrend')
plt.axhline(0, color='gray', linestyle=':')
plt.title("Moving Average Trend Score")
plt.legend()
plt.grid(True)
plt.show()

def print_stats(df):
    total_ret = df['cum_strategy_return'].iloc[-1] - 1
    sharpe = df['strategy_return'].mean() / df['strategy_return'].std() * np.sqrt(252)
    trades = df['signal'].diff().abs().sum() / 2
    max_dd = df['drawdown'].min()
    
    gross_profit = df.loc[df['strategy_return'] > 0, 'strategy_return'].sum()
    gross_loss = -df.loc[df['strategy_return'] < 0, 'strategy_return'].sum()  # Make positive
    
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf
    
    print(f"Total Return: {total_ret:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Number of Trades: {int(trades)}")

print_stats(strategy_df)
