import os
import pandas as pd

def merge_result_csvs(input_folder, output_file):
    # Initialize an empty DataFrame to hold the aggregated data
    aggregated_data = pd.DataFrame()

    # Loop through all CSV files in the folder
    for file in os.listdir(input_folder):
        if file.endswith('.csv'):  # Ensure we only process the correct files
            file_path = os.path.join(input_folder, file)
            
            # Read the current CSV file
            df = pd.read_csv(file_path)
            
            # Select only the columns we need and pivot the data
            df = df[['date', 'ticker', 'prediction']]
            df_pivot = df.pivot(index='date', columns='ticker', values='prediction')
            
            # Merge the current DataFrame into the aggregated DataFrame
            aggregated_data = pd.concat([aggregated_data, df_pivot], axis=1)

        # Reset the index to make 'date' a column
        aggregated_data.reset_index(inplace=True)

        # Rename columns (optional, for better clarity)
        aggregated_data.rename(columns={'index': 'Date'}, inplace=True)

        # Save the aggregated data to a CSV file
        aggregated_data.to_csv(output_file, index=False)

        print(f"Aggregated data saved to {output_file}")

# Define the folder containing the CSV files and the output file

# input_folder = 'result_lstm_1pct'
# output_file = 'test_csv.csv'
# merge_result_csvs(input_folder, output_file)

#########################################################################################

def generate_forward_variation_csv(input_csv, output_csv):
    """
    Generate a CSV with percentage variations based on forward comparison.

    Parameters:
        input_csv (str): Path to the input CSV file containing predicted values.
        output_csv (str): Path to save the generated CSV file.
    """
    # Read the input CSV file
    df = pd.read_csv(input_csv)

    # Initialize a new DataFrame for the output
    output_data = pd.DataFrame()

    # Extract the dates (first column in the CSV)
    dates = df.iloc[:, 0]
    stocks = df.columns[1:]  # All other columns are stock tickers

    # Process each stock
    for stock in stocks:
        predicted_values = df[stock].values
        
        # Compute forward percentage variations
        variations = [
            f"{((predicted_values[i+1] - predicted_values[i]) / predicted_values[i]) * 100:.1f}%"
            if i < len(predicted_values) - 1 else "None"
            for i in range(len(predicted_values))
        ]
        
        # Add the variations as a row in the output
        output_data[stock] = variations

    # Add the dates as the header row
    output_data.insert(0, 'Date', dates)

    # Save the output to CSV
    output_data.to_csv(output_csv, index=False)

    print(f"Forward variation CSV saved to {output_csv}")

# Example usage:
# generate_forward_variation_csv('test_csv.csv', 'forward_variation_output.csv')


####################################################################################

def convert_to_signals(input_file, output_file, threshold=0.01):
    df = pd.read_csv(input_file)

    def get_signal(pct_change):
        if pct_change > threshold:
            return 'buy'
        elif pct_change < -threshold:
            return 'sell'
        else:
            return 'hold'

    for col in df.columns:
        if col != 'Date':

            df[col] = df[col].str.rstrip('%').astype(float) / 100
            df[col] = df[col].apply(get_signal)

    df.to_csv(output_file, index=False)

    print(f"Signals saved to {output_file}")

# Example usage
input_file = 'forward_variation_output.csv'  
output_file = 'trading_signals_05pct.csv' 

convert_to_signals(input_file, output_file, threshold = 0.005)

#print(os.getcwd())


def standardize_casing(file_path, output_path):
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Apply title casing to all string elements in the DataFrame
        df = df.applymap(lambda x: x.title() if isinstance(x, str) else x)

        # Save the modified DataFrame to a new file
        df.to_csv(output_path, index=False)
        return f"Standardized file saved to {output_path}."
        
    except Exception as e:
        return str(e)


# File paths
input_file = "trading_signals_05pct.csv"
output_file = "trading_signals_05pct.csv"

# Apply the function
standardize_message = standardize_casing(input_file, output_file)
standardize_message