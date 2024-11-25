import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def read_and_process_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert all relevant columns from microseconds to milliseconds
    for column in df.columns[1:]:
        df[column] = df[column] / 1000
    
    return df

def plot_data_with_average(df, save_plot, output_file=None):
    columns_to_plot = df.columns[1:]  # Skip the Timestamp column
    sample_index = np.arange(len(df))

    fig, axes = plt.subplots(len(columns_to_plot), 1, figsize=(10, 15), sharex=True)

    for ax, column in zip(axes, columns_to_plot):
        data = df[column].to_numpy()
        ax.plot(sample_index, data, label=column)

        # Calculate fixed average
        fixed_average = np.mean(data)
        ax.axhline(fixed_average, color='red', linestyle='-', label=f'Average: {fixed_average:.2f} ms')

        ax.set_ylabel(f'{column} (ms)')
        ax.legend()

    plt.xlabel('Sample Index')
    plt.suptitle('Performance Metrics Over Time')

    if save_plot:
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Plot saved as {output_file}")
    else:
        plt.show()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Plot CSV data with average.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file')
    parser.add_argument('-s', '--save', action='store_true', help='Save the plot instead of displaying it')
    
    args = parser.parse_args()
    
    # Read and process CSV file
    df = read_and_process_csv(args.file_path)
    
    # Determine the output file name
    output_file = None
    if args.save:
        output_file = os.path.splitext(args.file_path)[0] + '.png'
    
    # Plot data with average
    plot_data_with_average(df, args.save, output_file)

if __name__ == '__main__':
    main()
