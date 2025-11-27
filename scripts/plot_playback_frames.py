import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.signal import medfilt
from matplotlib.gridspec import GridSpec

roll_limit = 1000   # degrees: above nothing is considered
airspeed_limit = 5  # m/s limit: below nothing is considered
airspeed_threshold = 3  # m/s threshold to stop plotting if below for too long
ground_truth_threshold = 0  # m: limit to consider ground truth values

legendsize = 25
ylabelsize = 25
xlabelsize = 25
ticksize = 25

meanWindow = 3
medianWindow = 3
curvesize = 4
alpha = 0.4
colorfill = 'red'
gt_color = 'black'
estimated_color = 'gray'
width_fig_inch=13.7
height_fig_inch=6


rows_skipped = 50

def synchronize_timestamps(df):
    # Identify the last valid timestamp before the abrupt change
    timestamp_diffs = df['timestamp'].diff()
    
    # Assuming the first large negative difference indicates the abrupt timestamp change
    abrupt_change_index = timestamp_diffs[timestamp_diffs < 0]

    if abrupt_change_index.empty:
        return df
    else :
        abrupt_change_index = abrupt_change_index.index[0]
    
    # Get the last valid timestamp before the change and the first timestamp after the change
    last_valid_timestamp = df.loc[abrupt_change_index - 1, 'timestamp']
    first_invalid_timestamp = df.loc[abrupt_change_index, 'timestamp']
    
    # Calculate the offset to adjust subsequent timestamps
    offset = last_valid_timestamp - first_invalid_timestamp
    
    # Adjust the subsequent timestamps to maintain continuity
    for i in range(abrupt_change_index, len(df)):
        df.at[i, 'timestamp'] = df.at[i, 'timestamp'] + offset
    
    return df

def main(input_file, save, median):


    col1 = ['timestamp','frameID','opticalFlow','featureDetection','totalProcessingTime','unfilteredAltitude','filteredAltitude','lidarData','distanceGround','rollAngle','pitchAngle','airspeed','groundspeed','vx','vy']

    col2 = ['Timestamp','frameID','estimatedAltitude','unfilteredAltitude','lidarData','Vx_body_FRD','Vy_body_FRD','Vz_body_FRD','Airspeed','Groundspeed','RollAngle','PitchAngle']
    
    col3 = ['timestamp','frameID','opticalFlow','featureDetection','totalProcessingTime','unfilteredAltitude','filteredAltitude','lidarData','liveUnfilteredAltitude','rollAngle','pitchAngle','airspeed','groundspeed','vx','vy']


    with open(input_file, 'r') as f:
        first_line = f.readline().strip()
    
    required_columns = ['rollAngle', 'distanceGround', 'filteredAltitude']

    if first_line == ','.join(col1):
        columns = col1
        df = pd.read_csv(input_file, names=columns, header=0)
        print("Using first header")

    elif first_line == ','.join(col2):
        columns = col2
        df = pd.read_csv(input_file, names=columns, header=0)
        print("Using second header")

        #change name to the same as the first header
        df.rename(columns={'Timestamp':'timestamp', 'RollAngle':'rollAngle', 'lidarData':'distanceGround', 'unfilteredAltitude':'filteredAltitude'}, inplace=True)
    
    elif first_line == ','.join(col3):
        columns = col3
        df = pd.read_csv(input_file, names=columns, header=0)
        print("Using third header")

        #change name to the same as the first header
        df.rename(columns={'lidarData':'distanceGround'}, inplace=True)
 
    else:
        print('Invalid header')
        return

    df = synchronize_timestamps(df)
    df['timestamp'] = df['timestamp'] / 1e6 - df['timestamp'].iloc[0] / 1e6

    #skip rows here
    df = df.iloc[rows_skipped:]
    
    # Apply median filter to smooth the data
    if median != None and columns == col1:
        df['filteredAltitude'] = df['filteredAltitude'].rolling(window=meanWindow).mean()

    if median != None and (columns == col2 or columns == col3):
        df['filteredAltitude'] = medfilt(df['filteredAltitude'], kernel_size=medianWindow)

    for col in required_columns:
        if col not in df.columns:
            print(f"Missing expected column: {col}")
            return

    df[required_columns] = df[required_columns].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=required_columns, inplace=True)
    df.dropna(inplace=True)

    valid_intervals = (df['rollAngle'].abs() < roll_limit) & (df['distanceGround'] > ground_truth_threshold)
    ground_truth = df['distanceGround']
    predictions = df['filteredAltitude']

    #predictions[ground_truth < 2] = ground_truth[ground_truth < 2] * np.random.uniform(1.0, 1.5, len(ground_truth[ground_truth < 2]))
    #predictions[ground_truth < 2] = medfilt(predictions[ground_truth < 2], kernel_size=51)

    relative_error = (np.abs(df['filteredAltitude'] - ground_truth) / ground_truth)
    filtered_valid_df = df[relative_error <= 0.5]

    fig, ax1 = plt.subplots(figsize=(width_fig_inch,height_fig_inch))


    ax1.plot(df['timestamp'].to_numpy(), ground_truth.to_numpy(), label='Ground Truth', linestyle='-', color=gt_color, linewidth=curvesize)
    ax1.plot(df['timestamp'].to_numpy(), predictions.to_numpy(), label='Estimate', linestyle='-', color=estimated_color, linewidth=curvesize)
    ax1.set_ylim([0, 35])

    # Fill the error area between Ground Truth and Prediction
    ax1.fill_between(df['timestamp'].to_numpy(), ground_truth.to_numpy(), predictions, color=colorfill, alpha=alpha)


    plt.tight_layout()

    if save:
        name = input_file.split('.')[0]
        fig.savefig(name + '.svg', format='svg')
        fig.savefig(name + '.png', format='png')
        fig.savefig(name + '.pdf', format='pdf')
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot flight data and compute errors.')
    parser.add_argument('input_file', type=str, help='Path to the input file containing flight data')
    parser.add_argument('-s, --save', dest='save', action='store_true', help='Save the plots to disk')
    parser.add_argument('-m, --median', dest='median', action='store_true',default=None, help='Use median filter to smooth the data')

    args = parser.parse_args()
    main(args.input_file, args.save, args.median)
