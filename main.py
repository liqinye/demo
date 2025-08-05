import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('synthetic_stock_data.csv')

print(df.head())

# Create a target variable for next-day direction
# Assuming 'Close' is the closing price column
# 1 if next day's close is higher, 0 otherwise
df['Next_Day_Up'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Feature engineering: adding moving average as a feature
df['MA_5'] = df['Close'].rolling(window=5).mean()
df['MA_10'] = df['Close'].rolling(window=10).mean()

# Drop NaN values created by rolling mean
df.dropna(inplace=True)

# Features: previous day's close, 5-day MA, 10-day MA
X = df[['Close', 'MA_5', 'MA_10']].shift(1).dropna()
y = df['Next_Day_Up'].dropna()

# Align X and y
X, y = X.align(y, join='inner', axis=0)

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')

