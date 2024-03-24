import pandas as pd
import numpy as np
from collections import Counter
import scipy.stats as stats  # Import the stats module from scipy

# Parameters
sequence_length = 124
total_genome_length = 248956422
mean = sequence_length / 2
std_dev = sequence_length / 8  # Standard deviation can be adjusted as needed
x = np.linspace(1, sequence_length, sequence_length)
weights = stats.norm.pdf(x, mean, std_dev)
weights /= np.sum(weights)  # Normalize the weights
# Initialize arrays and data structures
nucleotide_scores = np.zeros(total_genome_length)
nucleotide_counts = np.zeros(total_genome_length)
#nucleotide_labels = [Counter() for _ in range(total_genome_length)]  # List of Counters for labels
# Read the CSV file in chunks
chunk_size = 100000
total_chunks = 248803288 // chunk_size

for chunk_index, chunk in enumerate(pd.read_csv('part_sequence_predictions_HEK_iM.csv', chunksize=chunk_size)):
    print(f"Processing chunk {chunk_index + 1} of {total_chunks}")

    for idx, row in chunk.iterrows():
        # Calculate the genomic position of the sequence
        genomic_position = max(idx - (sequence_length // 2), 0)
        end_position = min(genomic_position + sequence_length, total_genome_length)
        central_nucleotide_pos = genomic_position + sequence_length // 2

        # Update scores, counts, and labels
        for pos in range(genomic_position, end_position):
            position_in_sequence = pos - genomic_position
            nucleotide_scores[pos] += row['Prediction'] * weights[position_in_sequence]
            nucleotide_counts[pos] += weights[position_in_sequence]
            if pos == central_nucleotide_pos:
                nucleotide_labels[pos][row['Label']] += 1

    print(f"Chunk {chunk_index + 1} processed.")

# Calculate the weighted average score for each nucleotide
weighted_averages = nucleotide_scores / nucleotide_counts
weighted_averages[np.isnan(weighted_averages)] = 0

# Determine the most frequent label for each nucleotide
final_labels = [label_counter.most_common(1)[0][0] if label_counter else np.nan for label_counter in nucleotide_labels]

# Save the results to a CSV file
results_df = pd.DataFrame({
    'Nucleotide_Position': np.arange(total_genome_length),
    'Label': final_labels,
    'Score': weighted_averages
})

results_df.to_csv('nucleotide_scores_HEK_iM.csv', index=False)
print("All data processed and saved to nucleotide_scores.csv.")
