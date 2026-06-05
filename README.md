A beginner ML project that predicts whether NVIDIA's stock price will go up or down the next day.
What it does:
Fetches real NVDA historical data via yfinance, engineers features (moving averages, RSI, daily return), and trains a Random Forest classifier to predict next-day price direction.
Results:
~58% accuracy on 2025-present data — slightly above the 50% random baseline.
Tech:
Python, yfinance, Pandas, scikit-learn
