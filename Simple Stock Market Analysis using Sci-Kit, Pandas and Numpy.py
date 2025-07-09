# To install the required libraries, run these commands in your terminal:
# pip install pandas numpy matplotlib yfinance scikit-learn

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import datetime

# --- 1. Configuration ---
TICKER = "AAPL"  # Example stock ticker (Apple Inc.)
START_DATE = "2010-01-01"
END_DATE = datetime.datetime.now().strftime("%Y-%m-%d") # Today's date
TRAINING_RATIO = 0.8 # 80% for training, 20% for testing
PREDICTION_FEATURES = [
    'Daily_Return',
    'SMA_20',    # Short-term Simple Moving Average
    'SMA_50',    # Medium-term Simple Moving Average
    'Volatility_10d', # 10-day rolling volatility
    'Volume'
]

# --- 2. Data Acquisition ---
def get_stock_data(ticker, start_date, end_date):
    """
    Downloads historical stock data using yfinance.
    """
    print(f"Downloading historical data for {ticker} from {start_date} to {end_date}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            print(f"No data found for {ticker} in the specified date range. Check ticker or dates.")
            return None
        print("Data downloaded successfully.")
        return data
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return None

# --- 3. Feature Engineering ---
def calculate_features(df):
    """
    Calculates various technical indicators and the target variable.
    Target: 1 if next day's close price is higher, 0 otherwise.
    """
    df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning

    # Daily Returns
    df_copy['Daily_Return'] = df_copy['Close'].pct_change()

    # Simple Moving Averages (SMAs)
    df_copy['SMA_20'] = df_copy['Close'].rolling(window=20).mean()
    df_copy['SMA_50'] = df_copy['Close'].rolling(window=50).mean()
    df_copy['SMA_200'] = df_copy['Close'].rolling(window=200).mean() # Longer term for context

    # Volatility (e.g., 10-day rolling standard deviation of daily returns)
    df_copy['Volatility_10d'] = df_copy['Daily_Return'].rolling(window=10).std()

    # Relative Strength Index (RSI) - A common momentum indicator
    # (Simplified calculation for demonstration)
    delta = df_copy['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    df_copy['RSI_14'] = 100 - (100 / (1 + RS))


    # Target Variable: Predict if the stock closes higher the next day
    # Shift(-1) moves the next day's close price to the current row
    df_copy['Target'] = (df_copy['Close'].shift(-1) > df_copy['Close']).astype(int)

    # Drop rows with NaN values created by rolling windows and shifting the target
    # This ensures our features and target are complete for all rows used in training
    df_copy = df_copy.dropna()

    return df_copy

# --- 4. Model Training and Evaluation ---
def train_and_evaluate_model(df_features, features, target_column='Target'):
    """
    Trains a RandomForestClassifier and evaluates its performance.
    """
    if df_features.empty:
        print("No data with features to train the model.")
        return None, None

    X = df_features[features]
    y = df_features[target_column]

    # Ensure no NaN values remain in features (e.g., if a new feature was added but not dropped)
    # This replaces any remaining NaNs with 0, which might not be ideal for all features,
    # but prevents model training errors. A better approach would be proper imputation.
    X = X.fillna(0)

    # Split data into training and testing sets
    # For time series, it's crucial to split chronologically to prevent data leakage
    split_index = int(len(X) * TRAINING_RATIO)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print(f"\nTraining data points: {len(X_train)}")
    print(f"Testing data points: {len(X_test)}")

    # Initialize and train the RandomForestClassifier
    # RandomForest is good for its simplicity and reasonable performance on tabular data
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') # Use class_weight for imbalanced targets
    print("Training the model...")
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    print(f"\nModel Accuracy on Test Set: {accuracy:.4f}")
    print("\nClassification Report:\n", report)

    return model, X_test.iloc[[-1]] # Return the trained model and the last test data point for next-day prediction

# --- 5. Visualization ---
def plot_stock_data(df, ticker):
    """
    Plots the closing price and selected moving averages.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(df['Close'], label='Close Price', color='blue')
    plt.plot(df['SMA_20'], label='20-Day SMA', color='orange')
    plt.plot(df['SMA_50'], label='50-Day SMA', color='green')
    plt.plot(df['SMA_200'], label='200-Day SMA', color='red', linestyle='--') # Optional longer term SMA
    plt.title(f'{ticker} Stock Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Main Execution ---
def main():
    print("--- Simple Stock Market Analyser ---")

    # Get stock data
    stock_data = get_stock_data(TICKER, START_DATE, END_DATE)

    if stock_data is None:
        return

    # Calculate features and target
    df_with_features = calculate_features(stock_data)

    if df_with_features.empty:
        print("Not enough data to calculate features. Exiting.")
        return

    # Display a sample of the data with features
    print("\nSample Data with Calculated Features:")
    print(df_with_features.tail()) # Show last few rows


    # Train and evaluate the model
    model, last_data_point_for_prediction = train_and_evaluate_model(df_with_features, PREDICTION_FEATURES)

    if model is None:
        print("Model could not be trained. Exiting.")
        return

    # --- Make a "Next Day" Prediction ---
    print("\n--- Making a 'Next Day' Prediction ---")
    if not last_data_point_for_prediction.empty:
        # We need to ensure the order of features is the same as training
        # And ensure it's a 2D array/DataFrame for prediction
        next_day_features = last_data_point_for_prediction[PREDICTION_FEATURES].fillna(0) # Fill NaNs for prediction if any
        
        # Make sure the index is correct for display
        last_date = next_day_features.index[0].strftime('%Y-%m-%d')
        
        predicted_direction = model.predict(next_day_features)[0]
        
        print(f"Based on data up to {last_date}:")
        if predicted_direction == 1:
            print(f"Predicted next day direction for {TICKER}: UP (price likely to increase)")
        else:
            print(f"Predicted next day direction for {TICKER}: DOWN (price likely to decrease or stay flat)")
        print("\nNOTE: This is a very basic prediction. Do NOT use for real financial decisions.")
    else:
        print("Could not get last data point for next-day prediction.")


    # --- Visualize the data ---
    print("\n--- Generating Stock Price Plot ---")
    plot_stock_data(df_with_features, TICKER)
    print("Plot displayed. Close the plot window to finish the script.")

if __name__ == "__main__":
    main()
