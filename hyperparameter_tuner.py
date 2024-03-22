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
from sklearn.preprocessing import StandardScaler


class GRUHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential()
        # First GRU layer with units_first_layer
        model.add(GRU(units=hp.Int('units_first_layer', min_value=32, max_value=512, step=32),
                      return_sequences=True,
                      input_shape=self.input_shape))
        model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)))
        
        # Second GRU layer with units_second_layer
        model.add(GRU(units=hp.Int('units_second_layer', min_value=32, max_value=512, step=32)))
        model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)))
        
        model.add(Dense(1))
        
        # Compile the model with a hyperparameter choice for the learning rate
        model.compile(optimizer=optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                      loss='mean_squared_error')
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
        # First LSTM layer with units_first_layer
        model.add(LSTM(units=hp.Int('units_first_layer', min_value=32, max_value=512, step=32),
                       return_sequences=True,
                       input_shape=self.input_shape))
        model.add(Dropout(rate=hp.Float('dropout_1', min_value=0, max_value=0.5, step=0.1)))
        
        # Second LSTM layer with units_second_layer
        model.add(LSTM(units=hp.Int('units_second_layer', min_value=32, max_value=512, step=32), return_sequences=False))
        model.add(Dropout(rate=hp.Float('dropout_2', min_value=0, max_value=0.5, step=0.1)))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile the model with a hyperparameter choice for the learning rate
        model.compile(optimizer=optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                      loss='mean_squared_error')
        return model

def prepare_data(df, features, target, train_percent, val_percent):
    # Prepare the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    if df.index.name != 'Date':
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

def find_and_save_hyperparameter_GRU(df, features, target, train_percent, val_percent, _dir, file_name):
    X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled = prepare_data(df, features, target, train_percent, val_percent)

    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])

    tuner = Hyperband(
        GRUHyperModel(input_shape),
        objective='val_loss',
        max_epochs=20,
        directory=_dir,
        project_name='gru_tuning'
    )

    tuner.search(X_train_scaled, y_train_scaled, epochs=20, validation_data=(X_val_scaled, y_val_scaled))
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    best_val_loss = best_trial.score

    model_metrics = {
        'units_first_layer' : best_hps.get('units_first_layer'),
        'units_second_layer': best_hps.get('units_second_layer'),
        'dropout_1': best_hps.get('dropout_1'),
        'dropout_2': best_hps.get('dropout_2'),
        'learning_rate': best_hps.get('learning_rate'),
        'best_val_loss': best_val_loss
    }
    with open(f'{file_name}.txt', 'w') as f:
        json.dump(model_metrics, f, indent=4)

def find_and_save_hyperparameter_biGRU(df, features, target, train_percent, val_percent, _dir, file_name):

    X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled = prepare_data(df, features, target, train_percent, val_percent)
    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])

    tuner = Hyperband(
        BiGRUHyperModel(input_shape),
        objective='val_loss',
        max_epochs=20,
        directory=_dir,
        project_name='bigru_tuning'
    )

    tuner.search(X_train_scaled, y_train_scaled, epochs=20, validation_data=(X_val_scaled, y_val_scaled), verbose=1)

    # Get the best set of hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    best_val_loss = best_trial.score

    best_hyperparameters = {
        'Best Units (First Layer)': best_hps.get('units_first_layer'),
        'Best Dropout (First Layer)': best_hps.get('dropout_1'),
        'Best Units (Second Layer)': best_hps.get('units_second_layer'),
        'Best Dropout (Second Layer)': best_hps.get('dropout_2'),
        'Best Learning Rate': best_hps.get('learning_rate'),
        'Lowest Validation MSE Score': best_val_loss
    }
    with open(f'{file_name}.txt', 'w') as file:
        for key, value in best_hyperparameters.items():
            file.write(f"{key}: {value}\n")


def find_and_save_hyperparameter_LSTM(df, features, target, train_percent, val_percent, _dir, file_name):
    X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled = prepare_data(df, features, target, train_percent, val_percent)

    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])

    tuner = Hyperband(
        LSTMTuner(input_shape=input_shape),
        objective='val_loss',
        max_epochs=20,
        directory=_dir,
        project_name='lstm_tuning'
    )

    tuner.search(X_train_scaled, y_train_scaled, epochs=20, validation_data=(X_val_scaled, y_val_scaled))

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    best_val_loss = best_trial.score

    best_hyperparameters_info = {
        'Best Units (First Layer)': best_hps.get('units_first_layer'),
        'Best Dropout (First Layer)': best_hps.get('dropout_1'),
        'Best Units (Second Layer)': best_hps.get('units_second_layer'),
        'Best Dropout (Second Layer)': best_hps.get('dropout_2'),
        'Best Learning Rate': best_hps.get('learning_rate'),
        'Lowest Validation MSE Score': best_val_loss
    }
    with open(f'{file_name}.txt', "w") as f:
        for key, value in best_hyperparameters_info.items():
            f.write(f"{key}: {value}\n")


# List of cryptocurrencies
cryptos = ['bitcoin', 'litecoin', 'ethereum', 'xrp']

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

df_bitcoin_2 = dfs['raw_bitcoin_pd'].copy()
df_bitcoin_2['SMA_50']  = df_bitcoin_2['Close'].rolling(window=50).mean()
df_bitcoin_2['SMA_200'] = df_bitcoin_2['Close'].rolling(window=200).mean()
df_bitcoin_2['EMA_50'] = df_bitcoin_2['Close'].ewm(span=50, adjust=False).mean()
df_bitcoin_2['EMA_200'] = df_bitcoin_2['Close'].ewm(span=200, adjust=False).mean()
EMA_12 = df_bitcoin_2['Close'].ewm(span=12, adjust=False).mean()
EMA_26 = df_bitcoin_2['Close'].ewm(span=26, adjust=False).mean()
df_bitcoin_2['MACD'] = EMA_12 - EMA_26
df_bitcoin_2['Signal_Line']    = df_bitcoin_2['MACD'].ewm(span=9, adjust=False).mean()
delta = df_bitcoin_2['Close'].diff(1)
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
RS = gain / loss
df_bitcoin_2['RSI'] = 100 - (100 / (1 + RS))
df_bitcoin_2['Volatility']   = df_bitcoin_2['Close'].pct_change().rolling(window=50).std() * np.sqrt(50)
df_bitcoin_2['Price_Direction'] = (df_bitcoin_2['Close'] > df_bitcoin_2['Close'].shift(1)).astype(int)
print(df_bitcoin_2.iloc[199:205])


# Print the percentage for bitcoin price direction
counts = df_bitcoin_2['Price_Direction'].value_counts()
percentages = counts / len(df_bitcoin_2) * 100
print(f"\nPercentages:\n {percentages}")

initial_row_count = len(df_bitcoin_2)
df_bitcoin_2_clean = df_bitcoin_2.dropna()
print(df_bitcoin_2_clean.count())
rows_removed = initial_row_count - len(df_bitcoin_2_clean)
print(f"Number of rows removed: {rows_removed}")


features = ['EMA_50', 'EMA_200', 'SMA_200', 'SMA_50', 'MACD', 'Signal_Line']
target = 'Close'
train_split      = 0.80
validation_split = 0.10

find_and_save_hyperparameter_GRU(  df_bitcoin_2_clean.copy(), features, target, train_split, validation_split, 'gru_dir_1  ', 'gru_1')
find_and_save_hyperparameter_LSTM( df_bitcoin_2_clean.copy(), features, target, train_split, validation_split, 'lstm_dir_1 ', 'lstm_1')
find_and_save_hyperparameter_biGRU(df_bitcoin_2_clean.copy(), features, target, train_split, validation_split, 'bigru_dir_1', 'bigru_1')
#----------------------------------Check with new features from Ethereum, Litecoin, and XRP----------------------------------

df_bitcoin_features = df_bitcoin_2_clean.copy()
df_bitcoin_features = df_bitcoin_features.drop(['Currency', 'Open', 'High', 'Low'], axis=1)
df_bitcoin_features = df_bitcoin_features.rename(columns={'Close': 'bitcoin_Close'})
cryptos_prices_to_copy = ['litecoin', 'ethereum', 'xrp']

# Convert dates into indexes------------------------------------------------------------
df_bitcoin_features['Date'] = pd.to_datetime(df_bitcoin_features['Date'])
df_bitcoin_features.set_index('Date', inplace=True)
for df in dfs.values():
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
#----------------------------------------------------------------------------------------

latest_start_date = max(df.index.min() for df in [df_bitcoin_features] + [dfs[f'raw_{crypto}_pd'] for crypto in cryptos_prices_to_copy])
df_bitcoin_features = df_bitcoin_features[df_bitcoin_features.index >= latest_start_date]

for crypto in cryptos_prices_to_copy:
    # Filter each crypto DataFrame from the latest start date onwards and add 'Close' price to df_bitcoin_features
    df_bitcoin_features[f'{crypto}_Close'] = dfs[f'raw_{crypto}_pd'].loc[dfs[f'raw_{crypto}_pd'].index >= latest_start_date, 'Close']

print(df_bitcoin_features.count())


features = ['EMA_50', 'EMA_200', 'SMA_200', 'SMA_50', 'MACD', 'Signal_Line', 'litecoin_Close', 'ethereum_Close']
target   = 'bitcoin_Close'


# Perform hyperparameter tuning using the features from the other cryptocurrencies.
find_and_save_hyperparameter_GRU(  df_bitcoin_features.copy(), features, target, train_split, validation_split, 'gru_dir_2', 'gru_2')
find_and_save_hyperparameter_LSTM( df_bitcoin_features.copy(), features, target, train_split, validation_split, 'lstm_dir_2', 'lstm_2')
find_and_save_hyperparameter_biGRU(df_bitcoin_features.copy(), features, target, train_split, validation_split, 'bigru_dir_2', 'bigru_2')
#---------------------------------------------------------------------------------

features = ['EMA_50', 'EMA_200', 'SMA_200', 'SMA_50', 'MACD', 'Signal_Line']
target   = 'bitcoin_Close'

# Perform hyperparameter tuning using the features without the other cryptocurrencies on the reduced dataset to compare the 
# effect of the additional features.
find_and_save_hyperparameter_GRU(  df_bitcoin_features.copy(), features, target, train_split, validation_split, 'gru_dir_3', 'gru_3')
find_and_save_hyperparameter_LSTM( df_bitcoin_features.copy(), features, target, train_split, validation_split, 'lstm_dir_3', 'lstm_3')
find_and_save_hyperparameter_biGRU(df_bitcoin_features.copy(), features, target, train_split, validation_split, 'bigru_dir_3', 'bigru_3')
#---------------------------------------------------------------------------------