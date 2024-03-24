import pandas as pd
import matplotlib.pyplot as plt
#upload the coverage_profile.bedgrph into a dataframe:
file_path ='atac_files/coverage_profile.bedgraph'
atac_seq_df = pd.read_csv(file_path, sep='\t', header=None, names=['Chr', 'Start', 'End', 'Score'])

# Assuming the DataFrame is already loaded correctly as atac_seq_df

# Define motif start and end based on the provided area information
motif_start = 10250
motif_end = 10374
# Filter only the rows from chromosome 1 and within the motif range
filtered_df = atac_seq_df[(atac_seq_df['Chr'] == 'chr1') &
                           (atac_seq_df['Start'] < motif_end) &
                           (atac_seq_df['End'] > motif_start)]

# Plotting
plt.figure(figsize=(10, 6))  # Adjust the figure size to be more proportional

# Set a fixed width for each bar to ensure they don't overlap
bar_width = 1

# Plot all bars with a light grey color and a fixed width
plt.bar(filtered_df['Start'], filtered_df['Score'],
        width=bar_width, color='lightgrey', edgecolor='black')

# Highlight the motif area by plotting bars in the motif region with a pastel blue color
motif_df = filtered_df[(filtered_df['Start'] >= motif_start) &
                       (filtered_df['End'] <= motif_end)]
plt.bar(motif_df['Start'], motif_df['Score'],
        width=bar_width, color='#aec6cf', edgecolor='black', label='Motif')

# Set the title and labels with improved aesthetics
plt.title('ATAC-seq Chromatin Accessibility Profile', fontsize=14)
plt.xlabel('Genomic Position (chr1)', fontsize=12)
plt.ylabel('Accessibility Score', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()

# Set the limits of the x-axis to the range of interest and the y-axis to the actual score range
plt.xlim(motif_start, motif_end)
plt.ylim(0, filtered_df['Score'].max() + 2)  # Add a small buffer above the max score

# Improve the aesthetics of the spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Show the final plot
plt.tight_layout()  # Adjust the padding of the plot
plt.show()
