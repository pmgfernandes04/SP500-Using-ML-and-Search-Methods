def split_data(df):
    # Convert Date to datetime and sort by Date
    df_preprocess['Date'] = pd.to_datetime(df_preprocess['Date'])
    df_preprocess = df_preprocess.sort_values(by='Date')

    # Removing adjusted close from training data
    df_preprocess = df_preprocess.drop(columns=['AdjClose'])

    # Extract temporal features
    df_preprocess['Day'] = df_preprocess['Date'].dt.day
    df_preprocess['Month'] = df_preprocess['Date'].dt.month
    df_preprocess['Year'] = df_preprocess['Date'].dt.year

    # 2010 - 2023
    train_data = df_preprocess[(df_preprocess['Date'] >= '2010-01-01') & (df_preprocess['Date'] < '2024-01-01')]

    # Test with January 2024
    test_data = df_preprocess[(df_preprocess['Date'] >= '2024-01-01') & (df_preprocess['Date'] < '2024-02-01')]

    # Separates train_data into X and y
    X_train = train_data.drop(columns=['Date','Close', 'High', 'Low', 'Open'])
    #X_train = train_data[['Day','Month','Year']]
    y_train = train_data['Close']

    # Separates test_data into X and y
    X_test = test_data.drop(columns=['Date','Close', 'High', 'Low', 'Open'])
    #X_test = test_data[['Day','Month','Year']]
    y_test = test_data['Close']

    # Keep for reference during performance analysis
    y_test_dates = test_data['Date']  

    return X_train, y_train, X_test, y_test, y_test_dates