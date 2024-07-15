
import os
import pandas as pd
import matplotlib.pyplot as plt

# Function to load and parse a single .dat file
def load_dat_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            label, concentration = parts[0].split(';')
            features = {int(k): float(v) for k, v in (item.split(':') for item in parts[1:])}
            features['label'] = label
            features['concentration'] = concentration
            data.append(features)
    return pd.DataFrame(data)

# Define the path to the directory containing .dat files
data_dir = "data"

# List all .dat files in the directory
dat_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.dat')]

# Load and parse all .dat files
dataframes = [load_dat_file(file) for file in dat_files]

# Plotting each file's data
for i, df in enumerate(dataframes):
    # Convert 'label' and 'concentration' to numeric
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df['concentration'] = pd.to_numeric(df['concentration'], errors='coerce')

    # Drop any rows with NaN values
    df = df.dropna()
    
    # Extract the filename for title
    filename = os.path.basename(dat_files[i])
    
    # Plotting
    plt.figure(figsize=(14, 7))
    
    # Plot label distribution
    plt.subplot(1, 2, 1)
    df['label'].hist(bins=50)
    plt.title(f'Label Distribution in {filename}')
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    
    # Plot concentration distribution
    plt.subplot(1, 2, 2)
    df['concentration'].hist(bins=50)
    plt.title(f'Concentration Distribution in {filename}')
    plt.xlabel('Concentration')
    plt.ylabel('Frequency')
    
    plt.suptitle(f'Summary Plots for {filename}')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
