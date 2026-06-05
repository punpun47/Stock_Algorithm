import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text

Nvidia_data = yf.download("NVDA", start = "2025-01-01")

# Create new column that stores 0 if close price tomorrow is greater than today and vice versa
Nvidia_data["Target"] = (Nvidia_data["Close"].shift(-1) > Nvidia_data["Close"]).astype(int)

# Create new column that stores mean of last 10 closing prices
Nvidia_data["MA_10"] = Nvidia_data["Close"].rolling(10).mean()

# Create new columnn that stores mean of last 50 closing prices
Nvidia_data["MA_50"] = Nvidia_data["Close"].rolling(50).mean()

# Create new column 
Nvidia_data["Daily_Return"] = Nvidia_data["Close"].pct_change()

# RSI
delta = Nvidia_data["Close"].diff()
gain = delta.clip(lower = 0)
loss = -delta.clip(upper = 0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

Nvidia_data["RSI"] = 100 - (100/(1 + avg_gain / avg_loss))

# Add Volume_Ratio - Tested Volume_Ratio but it lowered accuracy
Nvidia_data["Volume_Ratio"] = Nvidia_data["Volume"] / Nvidia_data["Volume"].rolling(10).mean()

# Drop null values caused by MA_50
Nvidia_data = Nvidia_data.dropna()
# Drop last row of Target becuase it will always be zero
Nvidia_data = Nvidia_data.iloc[:-1]

X = Nvidia_data[["Close", "MA_10", "MA_50", "Daily_Return", "RSI"]]
Y = Nvidia_data["Target"]

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

model = RandomForestClassifier(n_estimators = 100, random_state = 42)
model.fit(X_train, Y_train)

predictions = model.predict(X_test)
print(accuracy_score(Y_test, predictions))
