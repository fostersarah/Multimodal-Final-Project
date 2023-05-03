import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
import operator as op
from scipy.stats.stats import pearsonr


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

# Data visualization

# plot RPE
RPE_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
RPE_counts = [op.countOf(rpe_data, 1), op.countOf(rpe_data, 2), op.countOf(rpe_data, 3), op.countOf(rpe_data, 4),
              op.countOf(rpe_data, 5), op.countOf(rpe_data, 6), op.countOf(rpe_data, 7), op.countOf(rpe_data, 8),
              op.countOf(rpe_data, 9), op.countOf(rpe_data, 10)]

plt.bar(RPE_labels, RPE_counts)
plt.title('RPE Distribution')
plt.xlabel('Rating of Perceived Exertion')
plt.ylabel('Count')
plt.show()

# Correlation between RPE and phys data
plt.scatter(phys_data, rpe_data)
plt.title('RPE vs Adjusted Heart Rate')
plt.xlabel('Adjusted Heart Rate')
plt.ylabel('Rating of Perceived Exertion')
plt.show()

print(pearsonr(phys_data, rpe_data))