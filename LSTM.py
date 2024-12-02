#%% md
# # LSTM for S and P 500
#%%
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# For genetic algorithm
from deap import base, creator, tools, algorithms

# For performance metrics
import scipy.stats as scs

#%% md
# ## Collect S&P 500 Tickers
#%%
# Get the list of S&P 500 tickers
sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
sp500_table = pd.read_html(sp500_url)
sp500_tickers = sp500_table[0]['Symbol'].tolist()

# Clean up ticker symbols (e.g., replace '.' with '-' for tickers like 'BRK.B')
sp500_tickers = [ticker.replace('.', '-') for ticker in sp500_tickers]

# For demonstration, select a subset of tickers
tickers = sp500_tickers[:10]  # Adjust the number based on your computational capacity

#%% md
# ## Define Helper Functions
#%%
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def preprocess_data(data):
    prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    return scaled_prices, scaler

def create_sequences(data, dates, look_back):
    X, y, y_dates = [], [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i])
        y.append(data[i])
        y_dates.append(dates[i])
    return np.array(X), np.array(y), y_dates

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=False, input_shape=input_shape),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def generate_signals(predictions, dates, threshold):
    df_predictions = pd.DataFrame({
        'date': dates,
        'prediction': predictions.squeeze()
    })
    df_predictions['date'] = pd.to_datetime(df_predictions['date'])
    df_predictions.sort_values('date', inplace=True)
    df_predictions.reset_index(drop=True, inplace=True)
    df_predictions['next_day_prediction'] = df_predictions['prediction'].shift(-1)
    df_predictions['predicted_change'] = df_predictions['next_day_prediction'] - df_predictions['prediction']
    df_predictions['pct_change'] = df_predictions['predicted_change'] / df_predictions['prediction']
    df_predictions.dropna(subset=['next_day_prediction'], inplace=True)

    def get_signal(row):
        if row['pct_change'] > threshold:
            return 'buy'
        elif row['pct_change'] < -threshold:
            return 'sell'
        else:
            return 'hold'

    df_predictions['signal'] = df_predictions.apply(get_signal, axis=1)
    return df_predictions[['date', 'prediction', 'signal']]

#%% md
# ## Train LSTM Models and Generate Signals
#%%
# Define parameters
start_date = '2010-01-01'
split_date = '2023-12-31'
end_date = '2024-01-31'
look_back = 10
threshold = 0.005  # Adjust as needed

# Initialize dictionaries to store signals and actual prices
signals_dict = {}
prices_dict = {}

for ticker in tqdm(tickers):
    try:
        # Get data
        data = get_stock_data(ticker, start_date, end_date)
        if data.empty or len(data) < look_back + 1:
            continue  # Skip if insufficient data

        # Preprocess data
        scaled_prices, scaler = preprocess_data(data)
        dates_all = data.index

        # Create sequences
        X, y, y_dates = create_sequences(scaled_prices, dates_all, look_back)

        # Split into train and test based on dates
        split_index = np.where(pd.to_datetime(y_dates) >= pd.to_datetime(split_date))[0][0]
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        y_dates_train, y_dates_test = y_dates[:split_index], y_dates[split_index:]

        if len(X_test) == 0:
            continue  # Skip if no test data

        # Build and train model
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        early_stopping = EarlyStopping(monitor='loss', patience=5)
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, callbacks=[early_stopping])

        # Predict
        predictions = model.predict(X_test)
        predictions_unscaled = scaler.inverse_transform(predictions)

        # Generate signals
        signals = generate_signals(predictions_unscaled, y_dates_test, threshold)
        signals.set_index('date', inplace=True)
        signals.rename(columns={'signal': ticker}, inplace=True)

        # Store signals and actual prices
        signals_dict[ticker] = signals[[ticker]]
        prices_dict[ticker] = data.loc[signals.index, 'Close']
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        continue

#%% md
# ## Aggregate Signals and Prices
#%%
# Create a DataFrame for signals
signal_df = pd.concat(signals_dict.values(), axis=1)
signal_df.sort_index(inplace=True)

# Create a DataFrame for prices
price_df = pd.concat(prices_dict.values(), axis=1)
price_df.columns = prices_dict.keys()
price_df.sort_index(inplace=True)

# Align signal_df and price_df
common_dates = signal_df.index.intersection(price_df.index)
signal_df = signal_df.loc[common_dates]
price_df = price_df.loc[common_dates]

#%% md
# ##  Portfolio Optimization using Genetic Algorithm
#%%
# Define initial amount
initial_cash = 100000  # Starting with $100,000
max_positions = 10     # Maximum number of stocks to hold

def fitness(individual, signals, prices):
    total_cash = initial_cash
    portfolio = {}
    daily_returns = []

    for date in signals.index:
        # Get signals for the day
        daily_signals = signals.loc[date]
        daily_prices = prices.loc[date]

        # Determine stocks to buy/sell based on individual and signals
        buy_stocks = []
        sell_stocks = []
        for i, ticker in enumerate(signals.columns):
            if individual[i] == 1 and daily_signals[ticker] == 'buy':
                buy_stocks.append(ticker)
            elif ticker in portfolio and daily_signals[ticker] == 'sell':
                sell_stocks.append(ticker)

        # Sell stocks
        for ticker in sell_stocks:
            shares = portfolio.pop(ticker)
            total_cash += shares * daily_prices[ticker]

        # Buy stocks
        available_cash = total_cash / max_positions if max_positions > 0 else total_cash
        for ticker in buy_stocks:
            if len(portfolio) < max_positions:
                shares = available_cash / daily_prices[ticker]
                portfolio[ticker] = shares
                total_cash -= shares * daily_prices[ticker]

        # Calculate daily portfolio value
        portfolio_value = sum([shares * daily_prices[ticker] for ticker, shares in portfolio.items()])
        total_value = total_cash + portfolio_value
        daily_returns.append(total_value)

    # Calculate cumulative return
    cumulative_return = (daily_returns[-1] - initial_cash) / initial_cash
    return (cumulative_return,)

# Prepare data for optimization
signals = signal_df.copy()
prices = price_df.copy()

#%% md
# ### Set Up Genetic Algorithm
#%%
# Number of stocks
num_stocks = len(signals.columns)

# Create the toolbox
creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# Attribute generator
toolbox.register('attr_bool', np.random.randint, 2)
# Structure initializers
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_bool, n=num_stocks)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

# Register the evaluation function
toolbox.register('evaluate', fitness, signals=signals, prices=prices)
toolbox.register('mate', tools.cxUniform, indpb=0.5)
toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)
toolbox.register('select', tools.selTournament, tournsize=3)

#%% md
# ### Run Genetic Algorithm
#%%
population = toolbox.population(n=50)  # Population size
NGEN = 10  # Number of generations
for gen in range(NGEN):
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    fits = list(toolbox.map(toolbox.evaluate, offspring))
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

#%% md
# 
#%%
# Get the best individual
best_individual = tools.selBest(population, k=1)[0]
print('Best Individual:', best_individual)
selected_stocks = [ticker for i, ticker in enumerate(signals.columns) if best_individual[i] == 1]
print('Selected Stocks:', selected_stocks)

#%% md
# 
#%%
# Initialize variables for backtesting
total_cash = initial_cash
portfolio = {}
portfolio_values = []
dates = signals.index

for date in dates:
    # Get signals and prices for the day
    daily_signals = signals.loc[date]
    daily_prices = prices.loc[date]

    # Update portfolio based on signals
    for ticker in selected_stocks:
        if ticker in daily_signals.index:
            signal = daily_signals[ticker]
            price = daily_prices[ticker]
            if signal == 'buy' and ticker not in portfolio:
                # Buy stock
                shares = (total_cash / len(selected_stocks)) / price
                portfolio[ticker] = shares
                total_cash -= shares * price
            elif signal == 'sell' and ticker in portfolio:
                # Sell stock
                shares = portfolio.pop(ticker)
                total_cash += shares * price

    # Calculate total portfolio value
    portfolio_value = sum([shares * daily_prices[ticker] for ticker, shares in portfolio.items() if ticker in daily_prices.index])
    total_value = total_cash + portfolio_value
    portfolio_values.append({'date': date, 'total_value': total_value, 'cash': total_cash, 'portfolio_value': portfolio_value})

# Create DataFrame for portfolio values
portfolio_df = pd.DataFrame(portfolio_values)
portfolio_df.set_index('date', inplace=True)

#%% md
# 
#%%
# Calculate cumulative return
initial_value = portfolio_df['total_value'].iloc[0]
final_value = portfolio_df['total_value'].iloc[-1]
cumulative_return = (final_value - initial_value) / initial_value
print(f'Cumulative Return: {cumulative_return * 100:.2f}%')

#%% md
# 
#%%
# Ensure daily returns are calculated
portfolio_df['daily_return'] = portfolio_df['total_value'].pct_change()

# Drop the first NaN row caused by pct_change()
portfolio_df.dropna(subset=['daily_return'], inplace=True)

# Assume risk-free rate is 0 for simplicity
mean_return = portfolio_df['daily_return'].mean()
std_return = portfolio_df['daily_return'].std()

# Annualized Sharpe Ratio
sharpe_ratio = mean_return / std_return * np.sqrt(252)  # 252 trading days in a year
print(f'Sharpe Ratio: {sharpe_ratio:.2f}')

#%%
# Assume risk-free rate is 0 for simplicity
mean_return = portfolio_df['daily_return'].mean()
std_return = portfolio_df['daily_return'].std()
sharpe_ratio = mean_return / std_return * np.sqrt(252)  # Annualized Sharpe Ratio
print(f'Sharpe Ratio: {sharpe_ratio:.2f}')

#%% md
# 
#%%
# Plot portfolio value over time
plt.figure(figsize=(12, 6))
plt.plot(portfolio_df.index, portfolio_df['total_value'], label='Portfolio Value')
plt.title('Portfolio Value Over Time')
plt.xlabel('Date')
plt.ylabel('Total Value ($)')
plt.legend()
plt.grid(True)
plt.show()
