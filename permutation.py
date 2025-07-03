import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_synthetic_ohlc(n_bars=1000, seed=42):
    np.random.seed(seed)
    log_returns = np.random.normal(loc=0, scale=0.002, size=n_bars)
    log_price = np.cumsum(log_returns)
    close = np.exp(log_price)

    open_ = close * (1 + np.random.normal(0, 0.001, size=n_bars))
    high = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, 0.002, size=n_bars)))
    low = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, 0.002, size=n_bars)))

    dt_index = pd.date_range(start="2020-01-01", periods=n_bars, freq="H")

    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close
    }, index=dt_index)

    return df

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

def get_permutation(df, start_index=0, seed=None):
    np.random.seed(seed)
    time_index = df.index
    n_bars = len(df)
    perm_index = start_index + 1
    perm_n = n_bars - perm_index

    log_bars = np.log(df[['open', 'high', 'low', 'close']])
    start_bar = log_bars.iloc[start_index].to_numpy()

    r_o = (log_bars['open'] - log_bars['close'].shift()).to_numpy()
    r_h = (log_bars['high'] - log_bars['open']).to_numpy()
    r_l = (log_bars['low'] - log_bars['open']).to_numpy()
    r_c = (log_bars['close'] - log_bars['open']).to_numpy()

    idx = np.arange(perm_n)
    perm_high = r_h[perm_index:][np.random.permutation(idx)]
    perm_low = r_l[perm_index:][np.random.permutation(idx)]
    perm_close = r_c[perm_index:][np.random.permutation(idx)]
    perm_open = r_o[perm_index:][np.random.permutation(idx)]

    perm_bars = np.zeros((n_bars, 4))
    perm_bars[:start_index] = log_bars.iloc[:start_index].to_numpy()
    perm_bars[start_index] = start_bar

    for i in range(perm_index, n_bars):
        k = i - perm_index
        perm_bars[i, 0] = perm_bars[i - 1, 3] + perm_open[k]
        perm_bars[i, 1] = perm_bars[i, 0] + perm_high[k]
        perm_bars[i, 2] = perm_bars[i, 0] + perm_low[k]
        perm_bars[i, 3] = perm_bars[i, 0] + perm_close[k]

    perm_bars = np.exp(perm_bars)
    perm_df = pd.DataFrame(perm_bars, index=time_index, columns=['open', 'high', 'low', 'close'])
    return perm_df

def print_stats(label, df):
    total_return = df['cum_strategy_return'].iloc[-1] - 1
    max_drawdown = df['drawdown'].min()
    sharpe = df['strategy_return'].mean() / df['strategy_return'].std() * np.sqrt(24 * 365)
    print(f"{label}:")
    print(f"  Total Return:   {total_return:.2%}")
    print(f"  Max Drawdown:   {max_drawdown:.2%}")
    print(f"  Sharpe Ratio:   {sharpe:.2f}\n")
    return total_return

if __name__ == "__main__":
    btc_real = generate_synthetic_ohlc()

    real_result = compute_ma_trend_score(btc_real)
    real_return = print_stats("REAL", real_result)

    single_perm = get_permutation(btc_real, seed=123)
    single_result = compute_ma_trend_score(single_perm)
    print_stats("PERMUTED (example)", single_result)

    plt.figure(figsize=(12, 5))
    real_result['cum_strategy_return'].plot(label='Real', color='lime')
    single_result['cum_strategy_return'].plot(label='Permuted (Example)', linestyle='--', color='red')
    plt.title("Cumulative Strategy Return: Real vs One Permuted")
    plt.ylabel("Cumulative Return")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    n_permutations = 1000
    perm_returns = []

    for i in range(n_permutations):
        perm_df = get_permutation(btc_real, seed=i)
        result = compute_ma_trend_score(perm_df)
        final_return = result['cum_strategy_return'].iloc[-1] - 1
        perm_returns.append(final_return)

    perm_returns = np.array(perm_returns)
    p_value = np.mean(perm_returns >= real_return)

    print(f"Real Strategy Final Return: {real_return:.2%}")
    print(f"Mean Permuted Return:       {perm_returns.mean():.2%}")
    print(f"Permutation P-Value:        {p_value:.3f}")

    plt.figure(figsize=(10, 4))
    plt.hist(perm_returns, bins=30, alpha=0.7, label="Permuted Returns")
    plt.axvline(real_return, color='red', linestyle='--', label="Real Strategy")
    plt.title("Distribution of Permuted Strategy Returns")
    plt.xlabel("Final Cumulative Return")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()
