import numpy as np
import matplotlib.pyplot as plt

# Function to read sequences from the provided file
def read_lengths(file_path):
    lengths = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 3:
                start = int(parts[1])
                end = int(parts[2])
                lengths.append(end - start)
    return lengths


# Paths to the files
file_path_hek = '../pos_bed_files/HEK_iM_high_confidence_peaks.bed'
file_path_wdlps = '../pos_bed_files/WDLPS_iM_high_confidence_peaks.bed'

# Read sequences from files
lengths_hek = read_lengths(file_path_hek)
lengths_wdlps = read_lengths(file_path_wdlps)

# Calculate lengths
median_length_hek = np.median(lengths_hek)
median_length_wdlps = np.median(lengths_wdlps)

# Plotting the histogram with more bins and proper x-axis label
plt.figure(figsize=(10, 6))

# Histogram for HEK
plt.hist(lengths_hek, bins=200, color='skyblue', range=(0, 800), alpha=0.5, label=f'Hek high confidence peaks\n(n={len(lengths_hek)})')

# Histogram for WDLPS
plt.hist(lengths_wdlps, bins=200, color='orange', range=(0, 800), alpha=0.5, label=f'Wdlps high confidence peaks\n(n={len(lengths_wdlps)})')

# Highlight the medians
plt.axvline(median_length_hek, color='blue', linestyle='dashed', linewidth=1, label=f'Median: {median_length_hek} (hek)')
plt.axvline(median_length_wdlps, color='darkorange', linestyle='dashed', linewidth=1, label=f'Median: {median_length_wdlps} (wdlps)')

# Add legend
plt.legend(loc='upper right')

# Add labels
plt.xlabel('iM length (base pairs)')
plt.ylabel('Frequency')
# Set y-axis limits
plt.ylim(0, 600)

# Show plot without the title
plt.tight_layout()
#plt.show()
plt.savefig('iM_length_histogram.png')
