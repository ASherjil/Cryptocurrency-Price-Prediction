# Loading the data into dataframes
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
from tensorflow.keras.layers import GRU, Dropout, Dense, LSTM   
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, GRU, Dropout, Dense
from tensorflow.keras import optimizers
from kerastuner import HyperModel
from keras_tuner import HyperModel
from keras_tuner.tuners import Hyperband
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class GRUHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential()
        model.add(GRU(units=hp.Int('units', min_value=32, max_value=512, step=32),
                      return_sequences=True,
                      input_shape=self.input_shape))
        model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)))
        model.add(GRU(units=hp.Int('units', min_value=32, max_value=512, step=32)))
        model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
class BiGRUHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential()
        
        # First Bidirectional GRU layer with return_sequences=True since we will add another GRU layer after this
        model.add(Bidirectional(GRU(units=hp.Int('units_first_layer', min_value=32, max_value=512, step=32),
                                    return_sequences=True),
                                input_shape=self.input_shape))
        model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)))
        
        # Second Bidirectional GRU layer with return_sequences=False, which is the default and doesn't need to be specified
        model.add(Bidirectional(GRU(units=hp.Int('units_second_layer', min_value=32, max_value=512, step=32))))
        model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile the model
        model.compile(optimizer=optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                      loss='mean_squared_error')
        
        return model
    
class LSTMTuner(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential()
        # First LSTM layer
        model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32),
                       return_sequences=True,
                       input_shape=self.input_shape))
        model.add(Dropout(rate=hp.Float('dropout_1', min_value=0, max_value=0.5, step=0.1)))
        
        # Additional LSTM layer
        model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32), return_sequences=False))
        model.add(Dropout(rate=hp.Float('dropout_2', min_value=0, max_value=0.5, step=0.1)))
        
        # Output layer
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

def prepare_data(df, features, target, train_percent, val_percent):
    # Prepare the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    X = df[features]
    y = df[[target]]  # Keeping it as DataFrame
    
    # Scale features and target
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Reshape for LSTM [samples, time steps, features]
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    
    # Calculate the number of rows for training and validation
    total_rows = len(df)
    train_end = int(total_rows * train_percent)
    val_end = train_end + int(total_rows * val_percent)
    
    # Split data
    X_train_scaled = X_scaled[:train_end]
    y_train_scaled = y_scaled[:train_end]
    X_val_scaled = X_scaled[train_end:val_end]
    y_val_scaled = y_scaled[train_end:val_end]

    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled

def find_and_save_hyperparameter_GRU(df, features, target, train_percent, val_percent):

    X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled = prepare_data(df, features, target, train_percent, val_percent)

    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])

    tuner = Hyperband(
        GRUHyperModel(input_shape),
        objective='val_loss',
        max_epochs=20,
        directory='hyperband_gru_tuning',
        project_name='gru_tuning'
    )

    tuner.search(X_train_scaled, y_train_scaled, epochs=20, validation_data=(X_val_scaled, y_val_scaled))

    # Save the tuning results to a file
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model_metrics = {
        'units': best_hps.get('units'),
        'dropout_1': best_hps.get('dropout_1'),
        'dropout_2': best_hps.get('dropout_2')
    }

    with open('model_hyperparameter_tuning_GRU.txt', 'w') as f:
        json.dump(model_metrics, f)

def find_and_save_hyperparameter_biGRU(df, features, target, train_percent, val_percent):
    # Assuming prepare_data is a function that correctly splits and scales your data
    X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled = prepare_data(df, features, target, train_percent, val_percent)

    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])

    # Assuming BiGRUHyperModel is already defined and imported
    tuner = Hyperband(
        BiGRUHyperModel(input_shape),
        objective='val_loss',
        max_epochs=20,
        directory='hyperband_bigru_tuning',
        project_name='bigru_tuning'
    )

    tuner.search(X_train_scaled, y_train_scaled, epochs=20, validation_data=(X_val_scaled, y_val_scaled), verbose=1)

    # Extract the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Extract and save the best hyperparameters
    best_hyperparameters = {
        'Best Units (First Layer)': best_hps.get('units_first_layer'),
        'Best Dropout (First Layer)': best_hps.get('dropout_1'),
        'Best Units (Second Layer)': best_hps.get('units_second_layer'),
        'Best Dropout (Second Layer)': best_hps.get('dropout_2'),
        'Best Learning Rate': best_hps.get('learning_rate')
    }

    # Save the tuning results to a text file
    with open('bidirectional_GRU_hyperparameter_tuning_results.txt', 'w') as file:
        for key, value in best_hyperparameters.items():
            file.write(f"{key}: {value}\n")


def find_and_save_hyperparameter_LSTM(df, features, target, train_percent, val_percent):
    X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled = prepare_data(df, features, target, train_percent, val_percent)

    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])

    tuner = Hyperband(
        hypermodel=LSTMTuner(input_shape=input_shape),
        objective='val_loss',
        max_epochs=20,
        directory='hyperband_lstm_tuning',
        project_name='lstm_tuning'
    )

    tuner.search(X_train_scaled, y_train_scaled, epochs=20, validation_data=(X_val_scaled, y_val_scaled))

    # Save the tuning results to a file
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    with open('model_hyperparameter_tuning_LSTM.txt', "w") as f:
        f.write(f"Best LSTM units: {best_hps.get('units')}\n")
        f.write(f"Best dropout_1 rate: {best_hps.get('dropout_1')}\n")
        f.write(f"Best dropout_2 rate: {best_hps.get('dropout_2')}\n")


# List of cryptocurrencies
cryptos = ['bitcoin', 'litecoin', 'ethereum', 'monero', 'xrp']

# Dictionary to store the dataframes
dfs = {}

# Load each CSV file into a DataFrame and store it in the dictionary
for crypto in cryptos:
    df_name = 'raw_' + crypto + '_pd'
    dfs[df_name] = pd.read_csv(f'Top 100 Crypto Coins/{crypto}.csv')

    """""
    print(f"Top 5 rows of {df_name}:")
    print(dfs[df_name].head())
    print("\n")

    print(f"Statistics for {df_name}:")
    print(dfs[df_name].describe())
    print("\n")
    
    print(f"Missing values in {df_name}:")
    print(dfs[df_name].isnull().sum())
    print("\n")
    """""

# We need to add more features to the model to improve its accuracy
# Assuming dfs['raw_bitcoin_pd'] is your DataFrame for Bitcoin
df_bitcoin_2 = dfs['raw_bitcoin_pd'].copy()

# Calculate Moving Averages
df_bitcoin_2['SMA_50']  = df_bitcoin_2['Close'].rolling(window=50).mean()
df_bitcoin_2['SMA_200'] = df_bitcoin_2['Close'].rolling(window=200).mean()

# Calculate Exponential Moving Averages
df_bitcoin_2['EMA_50'] = df_bitcoin_2['Close'].ewm(span=50, adjust=False).mean()
df_bitcoin_2['EMA_200'] = df_bitcoin_2['Close'].ewm(span=200, adjust=False).mean()

# Calculate MACD
# MACD Line = 12-day EMA - 26-day EMA
# Signal Line = 9-day EMA of MACD Line
# MACD Histogram = MACD Line - Signal Line
EMA_12 = df_bitcoin_2['Close'].ewm(span=12, adjust=False).mean()
EMA_26 = df_bitcoin_2['Close'].ewm(span=26, adjust=False).mean()
df_bitcoin_2['MACD'] = EMA_12 - EMA_26
df_bitcoin_2['Signal_Line']    = df_bitcoin_2['MACD'].ewm(span=9, adjust=False).mean()

# Calculate RSI
delta = df_bitcoin_2['Close'].diff(1)
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
RS = gain / loss
df_bitcoin_2['RSI'] = 100 - (100 / (1 + RS))

# Calculate Volatility (as the standard deviation of daily returns)
#df_bitcoin_2['Daily_Return'] = df_bitcoin_2['Close'].pct_change()
df_bitcoin_2['Volatility']   = df_bitcoin_2['Close'].pct_change().rolling(window=50).std() * np.sqrt(50)

# If today's close is higher than yesterday's close, price direction is 1 (up), otherwise 0 (down or unchanged)
df_bitcoin_2['Price_Direction'] = (df_bitcoin_2['Close'] > df_bitcoin_2['Close'].shift(1)).astype(int)

# Display the head of the DataFrame to verify the new columns
print(df_bitcoin_2.iloc[199:205])


# Calculate the correlation matrix
correlation_matrix = df_bitcoin_2.corr()

# Focus on the 'High' and 'Low' columns
correlation_with_target = correlation_matrix[['Close']].sort_values(by='Close', ascending=False)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
#plt.show()
# Print the correlation values
print(correlation_with_target)

# Print the percentage for bitcoin price direction
counts = df_bitcoin_2['Price_Direction'].value_counts()
percentages = counts / len(df_bitcoin_2) * 100
print(f"\nPercentages:\n {percentages}")

def feature_importance_verifier(feature):
    # Assuming you're predicting the 'High' price. Adjust as necessary for 'Low'
    X = df_bitcoin_2.drop(['High', 'Low', 'Date', 'Currency', 'Open', 'Close'], axis=1).fillna(0)  # Example feature matrix
    y = df_bitcoin_2[feature].fillna(0)  # Example target variable

    # Initialize and train the random forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Extract feature importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print(f"Feature ranking for {feature}:")
    for f in range(X.shape[1]):
        print(f"{f + 1}. feature {X.columns[indices[f]]} ({importances[indices[f]]})")


#feature_importance_verifier('Close')
initial_row_count = len(df_bitcoin_2)
df_bitcoin_2_clean = df_bitcoin_2.dropna()
rows_removed = initial_row_count - len(df_bitcoin_2_clean)
#print(f"Number of rows removed: {rows_removed}")


features = ['EMA_50', 'EMA_200', 'SMA_200', 'SMA_50', 'MACD', 'Signal_Line']
target = 'Close'
train_split      = 0.80
validation_split = 0.10
find_and_save_hyperparameter_GRU(df_bitcoin_2_clean.copy(), features, target, train_split, validation_split)
find_and_save_hyperparameter_LSTM(df_bitcoin_2_clean.copy(), features, target, train_split, validation_split)
find_and_save_hyperparameter_biGRU(df_bitcoin_2_clean.copy(), features, target, train_split, validation_split)