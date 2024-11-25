import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

roll_limit = 30   # degrees: above nothing is considered
airspeed_limit = 5  # m/s limit: below nothing is considered
airspeed_threshold = 3  # m/s threshold to stop plotting if below for too long

def main(input_file, window_size, save_plot):
    # Define the column names for the data
    columns = ['Timestamp', 'DistanceGround', 'Zero', 'FilteredAvgAltitude',
               'Vx', 'Vy', 'Vz', 'Airspeed', 'RollAngle', 'PitchAngle']

    # Read the data into a DataFrame
    df = pd.read_csv(input_file, sep=',', header=None, names=columns)

    # Convert relevant columns to numeric, coercing errors to NaN
    df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
    df['DistanceGround'] = pd.to_numeric(df['DistanceGround'], errors='coerce')
    df['FilteredAvgAltitude'] = pd.to_numeric(df['FilteredAvgAltitude'], errors='coerce')
    df['Vx'] = pd.to_numeric(df['Vx'], errors='coerce')
    df['Vy'] = pd.to_numeric(df['Vy'], errors='coerce')
    df['Vz'] = pd.to_numeric(df['Vz'], errors='coerce')
    df['Airspeed'] = pd.to_numeric(df['Airspeed'], errors='coerce')
    df['RollAngle'] = pd.to_numeric(df['RollAngle'], errors='coerce')
    df['PitchAngle'] = pd.to_numeric(df['PitchAngle'], errors='coerce')

    # Apply rolling mean to relevant columns, excluding FilteredAvgAltitude
    rolling_columns = ['DistanceGround', 'Vx', 'Vy', 'Vz', 'Airspeed', 'RollAngle', 'PitchAngle']
    df[rolling_columns] = df[rolling_columns].rolling(window=window_size).mean()

    # Remove NaN values resulting from rolling window and type conversion
    df.dropna(inplace=True)

    # Identify the last index where airspeed is above the threshold
    valid_airspeed_indices = df.index[df['Airspeed'] > airspeed_threshold].tolist()
    if valid_airspeed_indices:
        last_valid_index = valid_airspeed_indices[-1]
        df = df.loc[:last_valid_index]

    # Identify invalid intervals
    invalid_intervals = (df['Airspeed'] <= airspeed_limit) | (df['RollAngle'] >= roll_limit) | (df['RollAngle'] <= -roll_limit)

    # Group contiguous invalid intervals
    invalid_regions = []
    start_idx = None

    for i in range(len(invalid_intervals)):
        if invalid_intervals.iloc[i]:
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                invalid_regions.append((df.iloc[start_idx]['Timestamp'], df.iloc[i - 1]['Timestamp']))
                start_idx = None
    if start_idx is not None:
        invalid_regions.append((df.iloc[start_idx]['Timestamp'], df.iloc[len(invalid_intervals) - 1]['Timestamp']))

    # Plot everything on the same graph without x-labels
    fig, axs = plt.subplots(2, 1, figsize=(14, 18), sharex=True)

    # Plot 1: Altitude comparison
    axs[0].plot(df['Timestamp'].values, df['DistanceGround'].values, label='Ground Truth')
    axs[0].plot(df['Timestamp'].values, df['FilteredAvgAltitude'].values, label='Filtered Avg Altitude')

    # Highlight invalid intervals
    for start, end in invalid_regions:
        axs[0].axvspan(start, end, color='red', alpha=0.3)

    axs[0].set_ylabel('Altitude (m)')
    axs[0].legend(loc='upper left')
    axs[0].grid(True)

    # Plot 2: Roll and Pitch angles
    axs[1].plot(df['Timestamp'].values, df['RollAngle'].values, label='Roll Angle')
    axs[1].plot(df['Timestamp'].values, df['PitchAngle'].values, label='Pitch Angle')
    axs[1].set_ylabel('Angle (degrees)')
    axs[1].legend(loc='upper left')
    axs[1].grid(True)

    for ax in axs:
        ax.label_outer()
        ax.set_xlabel('')

    plt.tight_layout()

    if save_plot:
        # Construct the output file name
        output_file = os.path.splitext(input_file)[0] + '.png'
        plt.savefig(output_file)
        print(f"Plot saved as {output_file}")
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot flight data.')
    parser.add_argument('input_file', type=str, help='Path to the input file containing flight data')
    parser.add_argument('-w', '--window_size', type=int, default=1, required=False, help='Window size for rolling average')
    parser.add_argument('-s', '--save', action='store_true', help='Save the plot instead of displaying it')

    args = parser.parse_args()
    main(args.input_file, args.window_size, args.save)
