import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load data from CSV
data = pd.read_excel(r'C:\Users\Mahidhar\Downloads\New folder\minor project.xlsx')

# Select features (columns Open, High, Low, Close, Volume) for prediction
X = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Select target variable (column Adj Close)
y = data['Adj Close']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)
