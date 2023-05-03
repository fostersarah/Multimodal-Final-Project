import pandas as pd
import numpy as np
import statistics


def get_statistics(data):
    mean = statistics.mean(data)
    median = statistics.median(data)
    mode = statistics.mode(data)
    sd = statistics.stdev(data)
    var = statistics.variance(data)
    statistics_list = [mean, median, mode, sd, var]
    return statistics_list


# Import heart rate data frame
hr = pd.read_csv('Aw-Soma Data.csv')

# Find difference: HR - resting
phys_data = hr['Set BPM'] - hr['Resting']

# RPE
rpe_data = hr['RPE']

# remove NaN values
phys_data = phys_data[:76]
rpe_data = rpe_data[:76]


# Statistics
phys_statistics = get_statistics(phys_data)
rpe_statistics = get_statistics(rpe_data)

data = {'Physiological Data': phys_statistics,
        'RPE Data': rpe_statistics}

df = pd.DataFrame(data)
df.index = ['Mean', 'Median', 'Mode', 'Standard Deviation', 'Variance']
print(df)
