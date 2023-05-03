import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Import heart rate data frame
hr = pd.read_csv('Aw-Soma Data.csv')

# Find difference: HR - resting
phys_data = hr['Set BPM'] - hr['Resting']

# remove NaN values
phys_data = phys_data[:76]

# sklearn scaling on data
scaler = MinMaxScaler((-1, 1))
phys_data = [[x] for x in phys_data]
scaled_phys_data = scaler.fit_transform(phys_data)

# create df and export csv file
df = pd.DataFrame(scaled_phys_data)
df.to_csv('PhysiologicalFeatures/physiologicalData.csv', header=False)